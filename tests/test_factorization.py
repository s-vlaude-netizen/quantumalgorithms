"""Double factorisation: correctness, and the measurement it was built to settle."""

from __future__ import annotations

import numpy as np
import pytest

from qres.factorization import (
    double_factorize,
    molecular_integrals,
    pauli_one_norm,
    variance_comparison,
)
from qres.problems.chemistry import build_molecule


def test_factorisation_reconstructs_the_two_body_tensor():
    """Exact reconstruction, or every downstream number is meaningless."""
    problem = build_molecule("H4")
    one_body, two_body, _ = molecular_integrals(problem)
    factorised = double_factorize(one_body, two_body)

    reconstructed = np.zeros_like(two_body)
    for t in range(factorised.num_factors):
        rotation = factorised.factor_rotations[t]
        matrix = rotation @ np.diag(factorised.factor_diagonals[t]) @ rotation.T
        reconstructed += factorised.factor_weights[t] * np.einsum("pq,rs->pqrs", matrix, matrix)

    assert np.abs(reconstructed - two_body).max() < 1e-10


def test_integrals_come_back_as_a_four_index_tensor():
    """qiskit-nature returns these 8-fold-symmetry packed; PySCF's are unpacked.

    Reading the packed form as a tensor is a silent shape error -- 55 numbers
    for 4 orbitals rather than 4^4 -- so the source matters.
    """
    problem = build_molecule("H4")
    one_body, two_body, nuclear = molecular_integrals(problem)
    n = one_body.shape[0]
    assert two_body.shape == (n, n, n, n)
    assert nuclear > 0
    # chemist notation is symmetric under (pq) <-> (rs)
    np.testing.assert_allclose(two_body, np.transpose(two_body, (2, 3, 0, 1)), atol=1e-10)


def test_truncation_removes_only_small_factors():
    problem = build_molecule("LiH")
    one_body, two_body, _ = molecular_integrals(problem)
    full = double_factorize(one_body, two_body)
    cut = full.truncated(1e-2)
    assert cut.num_factors < full.num_factors
    assert cut.one_norm() == pytest.approx(full.one_norm(), rel=0.05)


def test_double_factorisation_is_worse_than_pauli_grouping_for_sampling():
    """The measurement this module exists to make, kept as a regression guard.

    Double factorisation reduces the *number* of measurement settings, and is
    the standard 1-norm reduction for qubitisation. For a VQE **sampling**
    estimator it is worse, because each factor is a squared one-body operator
    and squaring inflates the variance. Measured at Hartree-Fock:

        H2   4.00x    H4  10.05x    LiH  6.82x   worse than commuting grouping

    If this ever inverts, the finding in RESEARCH_LOG Result 35 is wrong and
    the direction is worth reopening.
    """
    result = variance_comparison(build_molecule("H4"))
    assert result["factorised_variance"] > result["pauli_variance"]
    assert result["ratio"] > 2.0


def test_pauli_one_norm_excludes_the_identity():
    from qiskit.quantum_info import SparsePauliOp

    operator = SparsePauliOp(["II", "XZ", "ZY"], np.array([5.0, 1.0, -2.0], dtype=complex))
    assert pauli_one_norm(operator) == pytest.approx(3.0)
