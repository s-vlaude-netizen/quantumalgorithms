"""The gauge-fixed parameterisation (Result 80) and the budget bookkeeping.

Two things here can fail silently and did not have to. The gauge objective adds a
chain rule through a normalisation, which is exactly the shape of the error that
was wrong in Result 78 and caught only by finite differences; and the milestone
snapshot has to reproduce what a shorter run would have produced, or the whole
"one fit contains both budgets" saving is unsound.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from experiments.exp023_penalised_thc import _objective
from experiments.exp024_convergence_budget import (
    OLD_BUDGET,
    gauge_objective,
    run_fit,
    unpack_gauge,
    unpack_plain,
)
from experiments.exp022_optimised_thc import thc_tensor


def _pack(chi, coupling):
    return np.concatenate([chi.ravel(), coupling.ravel()])


@pytest.mark.parametrize("alpha", [0.0, 1e-4])
def test_gauge_gradient_matches_finite_differences(alpha):
    """Including the projection term, which is where a chain rule goes wrong."""
    rng = np.random.default_rng(11)
    orbitals, rank = 4, 5

    target = rng.normal(size=(orbitals,) * 4)
    target = 0.5 * (target + target.transpose(2, 3, 0, 1))
    target = 0.5 * (target + target.transpose(1, 0, 2, 3))
    target = 0.5 * (target + target.transpose(0, 1, 3, 2))

    # deliberately *not* unit norm, or the division by |chi| is untested
    chi = rng.normal(size=(orbitals, rank)) * np.array([0.3, 1.0, 2.5, 0.7, 1.4])
    coupling = rng.normal(size=(rank, rank))
    packed = _pack(chi, coupling)

    _, gradient = gauge_objective(packed, orbitals, rank, target, alpha)

    epsilon = 1e-7
    worst = 0.0
    for index in range(len(packed)):
        step = np.zeros_like(packed)
        step[index] = epsilon
        high, _ = gauge_objective(packed + step, orbitals, rank, target, alpha)
        low, _ = gauge_objective(packed - step, orbitals, rank, target, alpha)
        numeric = (high - low) / (2 * epsilon)
        worst = max(worst, abs(numeric - gradient[index]) / max(abs(numeric), 1e-9))

    assert worst < 1e-4, f"alpha={alpha}: gauge gradient off by {worst:.2e}"


def test_the_gauge_really_is_a_gauge():
    """Scaling a column and its Z entries must change nothing at all.

    This is the premise of the whole arm: if the direction were not exactly flat,
    removing it would be a change of model rather than of coordinates.
    """
    rng = np.random.default_rng(2)
    orbitals, rank = 4, 5
    chi = rng.normal(size=(orbitals, rank))
    coupling = rng.normal(size=(rank, rank))
    coupling = 0.5 * (coupling + coupling.T)
    tensor = thc_tensor(chi, coupling)

    scale = rng.uniform(0.3, 3.0, size=rank)
    rescaled = thc_tensor(chi * scale, coupling / np.outer(scale**2, scale**2))
    np.testing.assert_allclose(tensor, rescaled, atol=1e-10)


def test_the_gauge_gradient_has_no_radial_component():
    """The point of the parameterisation: the optimiser never moves along a column.

    If this fails the arm is not gauge-fixed and its iteration count means
    nothing.
    """
    rng = np.random.default_rng(7)
    orbitals, rank = 4, 5
    target = rng.normal(size=(orbitals,) * 4)
    target = 0.5 * (target + target.transpose(2, 3, 0, 1))
    target = 0.5 * (target + target.transpose(1, 0, 2, 3))
    target = 0.5 * (target + target.transpose(0, 1, 3, 2))

    chi = rng.normal(size=(orbitals, rank))
    coupling = rng.normal(size=(rank, rank))

    _, gradient = gauge_objective(_pack(chi, coupling), orbitals, rank, target, 0.0)
    grad_chi = gradient[: orbitals * rank].reshape(orbitals, rank)

    radial = np.sum(grad_chi * (chi / np.linalg.norm(chi, axis=0)), axis=0)
    assert np.max(np.abs(radial)) < 1e-9, (
        f"gradient still has a radial component of {np.max(np.abs(radial)):.2e}"
    )


def test_unpack_gauge_returns_unit_columns():
    chi = np.array([[3.0, 0.0], [4.0, 5.0]])
    coupling = np.eye(2)
    unit, _ = unpack_gauge(_pack(chi, coupling), 2, 2)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=0), 1.0, atol=1e-12)


def test_unpack_plain_symmetrises_the_coupling():
    """The fit symmetrises Z, so an assessment that did not would read a different
    model than the one that was optimised."""
    chi = np.ones((2, 2))
    coupling = np.array([[1.0, 4.0], [0.0, 1.0]])
    _, symmetric = unpack_plain(_pack(chi, coupling), 2, 2)
    np.testing.assert_allclose(symmetric, [[1.0, 2.0], [2.0, 1.0]], atol=1e-12)


def test_the_milestone_snapshot_equals_a_short_run():
    """One fit must contain both budgets, or the experiment measures two things.

    L-BFGS's trajectory does not depend on `maxiter`, so the iterate passing
    iteration `k` of a long run must equal the final iterate of a run capped at
    `k`. That is the assumption the saving rests on and it is cheap to check.
    """
    rng = np.random.default_rng(4)
    orbitals, rank, milestone = 4, 4, 12

    target = rng.normal(size=(orbitals,) * 4)
    target = 0.5 * (target + target.transpose(2, 3, 0, 1))
    target = 0.5 * (target + target.transpose(1, 0, 2, 3))
    target = 0.5 * (target + target.transpose(0, 1, 3, 2))
    start = rng.normal(size=(orbitals, rank))

    long_run = run_fit("plain", orbitals, rank, target, 0.0, start,
                       cap=200, milestone=milestone)
    assert long_run["reached_milestone"], "the fit must actually pass the milestone"

    packed = np.concatenate([start.ravel(), np.zeros(rank * rank)])
    short = minimize(
        _objective, packed, args=(orbitals, rank, target, 0.0), jac=True,
        method="L-BFGS-B",
        options={"maxiter": milestone, "maxfun": milestone * 2,
                 "ftol": 1e-14, "gtol": 1e-10},
    )
    np.testing.assert_allclose(long_run["milestone_x"], short.x, atol=1e-10)


def test_a_fit_that_hits_the_cap_is_not_counted_as_converged():
    """The bookkeeping error that produced Result 79's open question.

    A fit stopped by its budget is not a measurement, whatever its residual
    reads, and `scipy` reports success=False for it -- but the guard has to be
    explicit because a future tolerance change could make success=True at the cap.
    """
    rng = np.random.default_rng(5)
    orbitals, rank = 4, 6
    target = rng.normal(size=(orbitals,) * 4)
    target = 0.5 * (target + target.transpose(2, 3, 0, 1))
    target = 0.5 * (target + target.transpose(1, 0, 2, 3))
    target = 0.5 * (target + target.transpose(0, 1, 3, 2))

    starved = run_fit("plain", orbitals, rank, target, 0.0,
                      rng.normal(size=(orbitals, rank)), cap=3, milestone=2)
    assert starved["iterations"] == 3
    assert not starved["converged"]
    assert starved["status"] == 1


def test_the_old_budget_is_the_one_result_79_used():
    """A constant, pinned, because the whole comparison is against that number."""
    assert OLD_BUDGET == 20000
