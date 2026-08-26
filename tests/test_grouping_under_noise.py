"""The domain boundary of every variance-based result in this project.

RESEARCH_LOG Result 55: the measurement machinery optimises estimator
*variance*, and on a device noise model at realistic depths the error is
*bias* instead -- so the ranking those improvements are built on inverts.  Two
facts hold that conclusion up, and both are cheap enough to keep as tests:

* general-commuting grouping needs entangling gates to reach its measurement
  basis while qubit-wise commuting grouping needs none, which is the mechanism
* on an ideal simulator more shots buy accuracy at the usual `1/sqrt(n)`, and
  under device noise they buy almost nothing

If either stops holding, the domain caveat in the README and in Results 38, 54
and 55 has to be re-derived rather than quietly inherited.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qres.ansatz import hartree_fock_state
from qres.estimator import ShotEstimator
from qres.noise import make_environment
from qres.problems.chemistry import build_molecule

TWO_QUBIT_GATES = {"cx", "cz", "ecr", "cy", "swap", "rzz"}


def two_qubit_count(circuit) -> int:
    return sum(n for name, n in circuit.count_ops().items() if name in TWO_QUBIT_GATES)


def estimator_for(problem, scheme: str, environment: str, seed: int = 0):
    ansatz = hartree_fock_state(problem.hf_bitstring)
    return ShotEstimator(
        problem.hamiltonian,
        ansatz,
        make_environment(environment, seed=seed),
        grouping=scheme,
        allocation="uniform",
    )


def exact_energy(problem) -> float:
    state = Statevector(hartree_fock_state(problem.hf_bitstring))
    return float(np.real(state.expectation_value(problem.hamiltonian)))


def test_qwc_basis_changes_need_no_entangling_gates():
    """The mechanism behind Result 55's inversion.

    Qubit-wise commuting terms share a single-qubit measurement basis, so the
    basis change is a layer of rotations.  If this ever stops being true the
    inversion has a different cause than the one recorded.
    """
    problem = build_molecule("LiH")
    estimator = estimator_for(problem, "qwc", "ideal")
    assert sum(two_qubit_count(c) for c in estimator._isa) == 0


def test_general_commuting_basis_changes_do_need_entangling_gates():
    """The other half: general commuting groups pay a Clifford diagonalisation."""
    problem = build_molecule("LiH")
    estimator = estimator_for(problem, "commuting", "ideal")
    assert max(two_qubit_count(c) for c in estimator._isa) > 0


def test_general_commuting_uses_far_fewer_groups():
    """The advantage it buys with those gates, so the trade is visible here."""
    problem = build_molecule("LiH")
    commuting = estimator_for(problem, "commuting", "ideal")
    qwc = estimator_for(problem, "qwc", "ideal")
    assert len(commuting.groups) < len(qwc.groups) / 3


@pytest.mark.parametrize("scheme", ["commuting", "qwc"])
def test_ideal_error_falls_with_shots(scheme):
    """Shot-noise-limited: 16x the shots must buy a clear improvement."""
    problem = build_molecule("H4")
    exact = exact_energy(problem)

    def median_error(shots: int) -> float:
        errors = []
        for seed in range(8):
            estimator = estimator_for(problem, scheme, "ideal", seed=3000 + seed)
            errors.append(abs(estimator.estimate([], shots).value - exact))
        return float(np.median(errors))

    assert median_error(400_000) < 0.6 * median_error(25_000)


def test_noisy_error_is_bias_and_shots_barely_help():
    """The finding that bounds every variance-based result here.

    Under a device noise model the median error tracks the bias, so more shots
    do almost nothing -- measured at 4.3% for 16x the shots.  Asserted loosely
    (no better than 30%) so ordinary run-to-run variation does not fail it,
    while a return to `1/sqrt(n)` behaviour still would.
    """
    problem = build_molecule("H4")
    exact = exact_energy(problem)

    def measure(shots: int):
        values = [
            estimator_for(problem, "commuting", "heron", seed=4000 + seed)
            .estimate([], shots)
            .value
            for seed in range(8)
        ]
        values = np.array(values)
        return float(np.median(np.abs(values - exact))), float(np.mean(values) - exact)

    small_error, small_bias = measure(25_000)
    large_error, _ = measure(400_000)

    # the error is the bias, not the statistics
    assert abs(small_bias) == pytest.approx(small_error, rel=0.2)
    # and 16x the shots does not rescue it
    assert large_error > 0.7 * small_error
