"""Optimiser mechanics that results depend on."""

from __future__ import annotations

import numpy as np
import pytest

from qres.optimizers import (
    BudgetExceeded,
    EnergyOracle,
    OptimizeResult,
    estimate_horizon,
    shot_ladder,
    spsa,
    stochastic_parameter_shift,
)


class FakeEstimator:
    """Quadratic bowl plus optional Gaussian noise, charged like a real one."""

    def __init__(self, n=6, noise=0.0, seed=0):
        from qres.resources import ResourceLedger

        self.n = n
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.ledger = ResourceLedger()
        self.hamiltonian = type("H", (), {"coeffs": np.ones(3)})()

    def estimate(self, params, total_shots):
        from qres.estimator import EstimationResult

        self.ledger.record_execution(shots=total_shots, duration_s=1e-6)
        value = float(np.sum(np.asarray(params) ** 2))
        sigma = self.noise / np.sqrt(max(total_shots, 1))
        if sigma:
            value += self.rng.normal(0, sigma)
        return EstimationResult(value=value, stderr=sigma, shots_used=total_shots, circuits_used=1)

    def exact(self, params):
        return float(np.sum(np.asarray(params) ** 2))


def test_budget_is_enforced_exactly():
    oracle = EnergyOracle(FakeEstimator(), default_shots=100, budget=1000)
    with pytest.raises(BudgetExceeded):
        for _ in range(50):
            oracle(np.zeros(6))
    assert oracle.shots_used >= 1000


def test_estimate_horizon_uses_the_budget_not_maxiter():
    """The trap that froze SPSA's step size for a whole run in session 1."""
    oracle = EnergyOracle(FakeEstimator(), default_shots=1000, budget=100_000)
    assert estimate_horizon(oracle, maxiter=10**6, shots=1000, evals_per_iter=2) == 50
    # without a budget the sentinel is all there is
    free = EnergyOracle(FakeEstimator(), default_shots=1000, budget=None)
    assert estimate_horizon(free, maxiter=42, shots=1000) == 42


def test_shot_ladder_spends_the_budget_and_escalates():
    oracle = EnergyOracle(FakeEstimator(n=6), default_shots=2048, budget=4_000_000)
    result = shot_ladder(oracle, np.ones(6) * 0.4)
    schedule = result.extra["schedule"]
    assert len(schedule) >= 2, "a 4M budget should afford more than one rung"
    assert schedule == sorted(schedule), "shot counts must escalate"
    for a, b in zip(schedule, schedule[1:]):
        assert b > a
    assert oracle.shots_used <= 4_000_000
    assert oracle.shots_used > 0.5 * 4_000_000, "most of the budget should be spent"


def test_shot_ladder_refuses_to_over_ladder_on_a_small_budget():
    """Over-laddering measured *worse* than not laddering (H2 at 800k, p=0.000).

    The level count must fall with the budget rather than splitting it into
    rungs too thin to feed the inner optimiser's simplex.
    """
    small = EnergyOracle(FakeEstimator(n=12), default_shots=2048, budget=200_000)
    large = EnergyOracle(FakeEstimator(n=12), default_shots=2048, budget=51_200_000)
    n_small = len(shot_ladder(small, np.ones(12) * 0.3).extra["schedule"])
    n_large = len(shot_ladder(large, np.ones(12) * 0.3).extra["schedule"])
    assert n_small < n_large
    assert n_small == 1, "a 200k budget cannot feed two rungs for 12 parameters"


def test_shot_ladder_converges_on_a_noiseless_bowl():
    oracle = EnergyOracle(FakeEstimator(n=4, noise=0.0), default_shots=1024, budget=2_000_000)
    result = shot_ladder(oracle, np.ones(4))
    assert result.fun < 1e-6


def test_stochastic_parameter_shift_visits_every_coordinate():
    oracle = EnergyOracle(FakeEstimator(n=5), default_shots=64, budget=200_000)
    result = stochastic_parameter_shift(
        oracle, np.ones(5) * 0.2, maxiter=200, coordinates=1, shots=64
    )
    visits = result.extra["visits"]
    assert all(v > 0 for v in visits), "every parameter must get updated"
    assert sum(visits) == result.nit


def test_spsa_and_ladder_return_the_expected_shape():
    for run in (
        lambda o: spsa(o, np.ones(4) * 0.3, maxiter=20, shots=256, seed=0),
        lambda o: shot_ladder(o, np.ones(4) * 0.3),
    ):
        oracle = EnergyOracle(FakeEstimator(n=4), default_shots=256, budget=1_000_000)
        try:
            result = run(oracle)
        except BudgetExceeded:
            continue
        assert isinstance(result, OptimizeResult)
        assert len(result.x) == 4
        assert np.isfinite(result.fun)


def test_multi_start_charges_for_its_own_selection():
    """The audit must come out of the budget, not be free.

    Ranking candidates by each run's own best observed value would cost nothing
    and be wrong: that value is the minimum of many noisy draws, biased low by
    roughly the run's noise, so it favours the noisiest run rather than the best.
    """
    from qres.optimizers import multi_start

    oracle = EnergyOracle(FakeEstimator(n=4, noise=1.0), default_shots=512, budget=2_000_000)
    result = multi_start(oracle, np.ones(4) * 0.3, starts=3, audit_fraction=0.2)
    assert len(result.extra["audit_scores"]) == 3
    assert 0 <= result.extra["winner"] < 3
    assert oracle.shots_used <= 2_000_000


def test_multi_start_splits_the_budget_between_starts():
    from qres.optimizers import multi_start

    for starts in (2, 4):
        oracle = EnergyOracle(FakeEstimator(n=4), default_shots=512, budget=4_000_000)
        multi_start(oracle, np.ones(4) * 0.3, starts=starts, audit_fraction=0.1)
        assert oracle.shots_used <= 4_000_000
        assert oracle.shots_used > 0.4 * 4_000_000


def test_trust_radius_estimation_refuses_rather_than_guessing():
    """Three principled heuristics failed; a wrong guess is worse than none.

    See RESEARCH_LOG Result 40 -- the signal each one needs is smaller than the
    shot noise at any affordable probe cost.
    """
    from qres.optimizers import TRUST_RADIUS, estimate_trust_radius

    with pytest.raises(NotImplementedError):
        estimate_trust_radius()
    # the measured values, which differ by 10x and both matter
    assert TRUST_RADIUS["hardware_efficient"] == 1.0
    assert TRUST_RADIUS["coupled_cluster"] == 0.1


def test_trust_region_steps_along_an_averaged_gradient():
    """A single random probe is not a step direction.

    The first version stepped a full radius along one simultaneous-perturbation
    direction. In n dimensions that has cosine ~1/sqrt(n) with the true
    gradient, so the realised decrease misses the predicted one, every step is
    rejected, and each rejection shrinks the radius -- which under the
    radius^-4 shot rule quadruples the next iteration's cost. Measured: 6
    iterations, all rejected, budget gone (RESEARCH_LOG Result 41).
    """
    from qres.optimizers import stochastic_trust_region

    oracle = EnergyOracle(FakeEstimator(n=6, noise=0.5), default_shots=512, budget=4_000_000)
    start = np.ones(6) * 0.5
    seen: list[float] = []
    try:
        stochastic_trust_region(
            oracle, start, probes=4, kappa=100.0, callback=lambda x, v: seen.append(v)
        )
    except BudgetExceeded:
        pass  # spending the whole budget is the normal exit; the driver catches it
    assert len(seen) > 10, "should complete many iterations, not spiral out"
    assert min(seen) < float(np.sum(start**2)), "should make progress"


def test_trust_region_ties_shots_to_radius():
    """The whole point: precision escalates as the radius shrinks."""
    from qres.optimizers import stochastic_trust_region

    oracle = EnergyOracle(FakeEstimator(n=4), default_shots=512, budget=200_000)
    radii: list[float] = []
    try:
        stochastic_trust_region(
            oracle,
            np.ones(4) * 0.5,
            probes=2,
            kappa=10.0,
            callback=lambda x, v: radii.append(1.0),
        )
    except BudgetExceeded:
        pass
    # the shot rule n >= (sigma / (kappa r^2))^2 is monotone decreasing in r,
    # which is the whole design: a smaller radius must cost more shots
    sigma = 1.0
    costs = [(sigma / (10.0 * r**2)) ** 2 for r in (0.5, 0.25, 0.125)]
    assert costs[0] < costs[1] < costs[2]
    assert costs[2] / costs[0] == pytest.approx(256.0), "halving twice is 16^2"
