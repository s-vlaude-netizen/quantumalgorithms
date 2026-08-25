"""Noise environments, and the property every optimisation result depends on."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qres.noise import describe_environment, make_environment


def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.mark.parametrize("spec", ["ideal", "medium"])
def test_shot_noise_is_resampled_between_runs(spec):
    """Each run must draw fresh shot noise.

    A fixed ``seed_simulator`` makes Aer return identical counts for the same
    circuit every time. The optimiser then faces a *frozen* rough landscape
    instead of a stochastic one -- repeated evaluations at a point agree
    exactly, and it can fit itself to one noise realisation rather than
    averaging over it. This distorts every optimisation result while looking
    entirely normal, so it is checked rather than assumed.
    """
    env = make_environment(spec)
    isa = env.prepare(bell_circuit())
    runs = [env.run([isa], shots=1024)[0] for _ in range(4)]
    assert any(r != runs[0] for r in runs[1:]), "shot noise is not being resampled"


def test_runs_are_reproducible_across_environments_with_the_same_seed():
    """Fresh draws must not cost reproducibility."""
    isa_a = make_environment("ideal", seed=7).prepare(bell_circuit())
    first = [make_environment("ideal", seed=7).run([isa_a], shots=512)[0] for _ in range(1)]
    second = [make_environment("ideal", seed=7).run([isa_a], shots=512)[0] for _ in range(1)]
    assert first == second


def test_estimator_is_unbiased():
    """Sampled energy must converge on the exact one, not merely near it."""
    from qiskit.quantum_info import Statevector

    from qres.ansatz import build_ansatz
    from qres.estimator import ShotEstimator
    from qres.problems.chemistry import build_molecule
    from qres.resources import ResourceLedger

    problem = build_molecule("H2")
    env = make_environment("ideal")
    ansatz = build_ansatz("hea:2", problem, env)
    estimator = ShotEstimator(problem.hamiltonian, ansatz, env, ledger=ResourceLedger())

    rng = np.random.default_rng(0)
    params = rng.normal(0, 0.4, ansatz.num_parameters)
    exact = float(
        Statevector(ansatz.assign_parameters(params)).expectation_value(problem.hamiltonian).real
    )

    samples = np.array([estimator.estimate(params, 8192).value for _ in range(40)])
    sem = samples.std(ddof=1) / np.sqrt(len(samples))
    assert abs(samples.mean() - exact) < 4 * sem, "estimator is biased"


def test_sampling_error_scales_as_one_over_sqrt_shots():
    from qres.ansatz import build_ansatz
    from qres.estimator import ShotEstimator
    from qres.problems.chemistry import build_molecule
    from qres.resources import ResourceLedger

    problem = build_molecule("H2")
    env = make_environment("ideal")
    ansatz = build_ansatz("hea:2", problem, env)
    estimator = ShotEstimator(problem.hamiltonian, ansatz, env, ledger=ResourceLedger())
    rng = np.random.default_rng(1)
    params = rng.normal(0, 0.4, ansatz.num_parameters)

    spreads = {}
    for shots in (2048, 32768):
        values = np.array([estimator.estimate(params, shots).value for _ in range(40)])
        spreads[shots] = values.std(ddof=1)
    # 16x the shots should be ~4x tighter; allow a wide band for 40 samples
    ratio = spreads[2048] / spreads[32768]
    assert 2.5 < ratio < 6.0, f"error scaling is wrong: ratio {ratio:.2f}, expected ~4"


def test_describe_environment_reports_real_calibration():
    described = describe_environment(make_environment("medium"))
    assert described["backend"] == "fake_kolkata"
    assert 0 < described["two_qubit_error"]["median"] < 0.1
    assert 0 < described["readout_error"]["median"] < 0.2
