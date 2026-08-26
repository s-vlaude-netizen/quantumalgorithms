"""Correctness of the error-mitigation machinery.

Mitigation is the one part of this package that can make an answer *worse* while
looking like it is helping, so the properties below are the ones that keep it
honest:

* folding must leave the ideal state exactly unchanged -- if it does not, the
  extrapolation is fitting a curve through a moving target
* extrapolation must be exact on data of the order it claims to handle
* the readout correction must produce a probability distribution, since raw
  matrix inversion produces negative entries that make expectation values worse
  than no correction at all
* calibration must be measured on the qubits the estimator actually uses.  At
  optimisation level 2 the transpiler picked physical qubits [66, 5, 87, 81, 60,
  51] on a 133-qubit target, whose readout fidelity is 0.95-0.98 against
  0.68-0.75 on qubits 0-5.  Calibrating on the wrong ones corrects with numbers
  that are off by that much, confidently.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qres.ansatz import hartree_fock_state
from qres.estimator import ShotEstimator
from qres.mitigation import (
    _project_to_simplex,
    assignment_matrix_on,
    correct_counts,
    extrapolate,
    fold_global,
    measured_physical_qubits,
)
from qres.noise import make_environment
from qres.problems.chemistry import build_molecule


@pytest.mark.parametrize("scale", [1, 3, 5, 7])
def test_folding_preserves_the_ideal_state_exactly(scale):
    """`C (C^dag C)^k` is `C`. Not approximately -- exactly."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.7, 2)
    circuit.cx(1, 2)

    folded = fold_global(circuit, scale)
    overlap = abs(Statevector(circuit).inner(Statevector(folded)))
    assert overlap == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("scale", [1, 3, 5])
def test_folding_multiplies_the_gate_count(scale):
    """The point of folding is more gates, hence more noise, on real hardware."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    base = sum(circuit.count_ops().values())
    assert sum(fold_global(circuit, scale).count_ops().values()) == scale * base


def test_folding_rejects_even_scales():
    """An even fold does not return to the original unitary."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    for bad in (0, 2, 4, -1):
        with pytest.raises(ValueError):
            fold_global(circuit, bad)


def test_linear_extrapolation_is_exact_on_linear_data():
    scales = [1, 3, 5]
    assert extrapolate(scales, [2.0 + 0.5 * s for s in scales], "linear") == pytest.approx(2.0)


def test_richardson_is_exact_on_quadratic_data():
    """Three points determine a quadratic, so Richardson must nail it."""
    scales = [1, 3, 5]
    values = [2.0 + 0.5 * s + 0.1 * s**2 for s in scales]
    assert extrapolate(scales, values, "richardson") == pytest.approx(2.0)
    # and the line is biased on the same data -- the trade this offers
    assert extrapolate(scales, values, "linear") != pytest.approx(2.0, abs=0.1)


def test_extrapolation_needs_two_points():
    with pytest.raises(ValueError):
        extrapolate([1], [1.0])


def test_simplex_projection_returns_a_distribution():
    rng = np.random.default_rng(0)
    for _ in range(20):
        vector = rng.normal(0, 1, 8)
        projected = _project_to_simplex(vector)
        assert projected.sum() == pytest.approx(1.0)
        assert (projected >= -1e-12).all()


def test_simplex_projection_leaves_a_valid_distribution_alone():
    distribution = np.array([0.5, 0.25, 0.25])
    assert _project_to_simplex(distribution) == pytest.approx(distribution)


def test_correct_counts_inverts_a_known_assignment_matrix():
    """Apply a synthetic readout error, then undo it."""
    truth = np.array([0.7, 0.1, 0.15, 0.05])
    matrix = np.array([
        [0.90, 0.05, 0.05, 0.01],
        [0.05, 0.90, 0.01, 0.05],
        [0.04, 0.01, 0.90, 0.05],
        [0.01, 0.04, 0.04, 0.89],
    ])
    matrix /= matrix.sum(axis=0)

    observed = matrix @ truth
    counts = {format(i, "02b"): observed[i] * 100_000 for i in range(4)}

    corrected = correct_counts(counts, matrix, 2)
    total = sum(corrected.values())
    recovered = np.array([corrected.get(format(i, "02b"), 0.0) / total for i in range(4)])

    assert recovered == pytest.approx(truth, abs=1e-6)
    # and the uncorrected distribution was genuinely wrong, so this is a real test
    assert np.abs(observed - truth).max() > 0.05


def test_correct_counts_survives_empty_input():
    assert correct_counts({}, np.eye(4), 2) == {}


def test_calibration_follows_the_transpiler_layout():
    """The estimator's qubits, not 0..n-1 -- and they are much better ones."""
    problem = build_molecule("H4")
    ansatz = hartree_fock_state(problem.hf_bitstring)
    environment = make_environment("heron", seed=1)
    estimator = ShotEstimator(
        problem.hamiltonian, ansatz, environment,
        grouping="qwc", allocation="uniform", optimization_level=2,
    )

    physical = measured_physical_qubits(estimator)
    assert len(physical) == ansatz.num_qubits
    assert physical != list(range(ansatz.num_qubits)), "level 2 should relocate"

    chosen = assignment_matrix_on(environment, physical, shots=512)
    trivial = assignment_matrix_on(environment, list(range(ansatz.num_qubits)), shots=512)

    assert np.allclose(chosen.sum(axis=0), 1.0)
    # the transpiler picks qubits that read out better; that gap is exactly why
    # calibrating on the wrong ones corrupts the correction
    assert np.diag(chosen).mean() > np.diag(trivial).mean()


def test_tensored_calibration_uses_two_circuits_and_agrees_with_exact():
    """The approximation must be close, and it must be cheap.

    Independent per-qubit readout is an assumption, not a fact -- crosstalk and
    shared readout lines break it.  On this noise model the two matrices agree
    to ~0.01 elementwise, which is what makes the 2^n -> 2 circuit saving worth
    taking (Result 57: the approximation beats the exact method 1.7x at a fixed
    budget, because it gets 2^(n-1) times more shots per calibration point).
    """
    from qres.mitigation import tensored_assignment_matrix

    environment = make_environment("heron", seed=1)
    qubits = [66, 5, 87, 81, 60, 51]

    exact = assignment_matrix_on(environment, qubits, shots=4096)
    tensored = tensored_assignment_matrix(environment, qubits, shots=4096)

    assert tensored.shape == exact.shape
    assert np.allclose(tensored.sum(axis=0), 1.0)
    assert np.abs(exact - tensored).max() < 0.05
    assert np.diag(tensored).mean() == pytest.approx(np.diag(exact).mean(), abs=0.02)


def test_tensored_matrix_factorises_in_the_right_qubit_order():
    """It must be a Kronecker product, with qubit 0 as the *last* factor.

    Qubit 0 is the least significant bit of the basis index.  Reversing the
    factor order produces a matrix that still has unit columns and still looks
    like a plausible assignment matrix, while attributing every qubit's readout
    error to a different qubit -- so this checks the per-qubit error the
    tensored matrix implies against the same quantity measured independently by
    the exact calibration.
    """
    from qres.mitigation import tensored_assignment_matrix

    environment = make_environment("heron", seed=1)
    # deliberately mismatched qubits: 0 has poor readout, 66 good, so getting
    # the order backwards moves a large error onto the wrong one
    qubits = [0, 66]

    tensored = tensored_assignment_matrix(environment, qubits, shots=8192)
    exact = assignment_matrix_on(environment, qubits, shots=8192)

    def implied_error_ratio(matrix, qubit_index):
        """A_k[1,0] / A_k[0,0], read off the prepared-all-zeros column."""
        return matrix[1 << qubit_index, 0] / matrix[0, 0]

    for qubit_index in range(len(qubits)):
        assert implied_error_ratio(tensored, qubit_index) == pytest.approx(
            implied_error_ratio(exact, qubit_index), abs=0.03
        ), f"qubit {qubit_index} error landed on the wrong factor"

    # and the two qubits must genuinely differ, or the check above is vacuous
    ratios = [implied_error_ratio(tensored, i) for i in range(len(qubits))]
    assert max(ratios) > 3 * min(ratios), "qubits too similar to detect a swap"
