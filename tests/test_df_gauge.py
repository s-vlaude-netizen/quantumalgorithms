"""The double-factorisation gauge group (Result 87).

Two things here can look right while being wrong, and both bit during
development. The 1-norm formula used for the optimisation has to be *this
repository's* 1-norm, or the reduction is against a definition nobody uses. And
a witness for non-invariance can accidentally sit on an invariant orbit -- the
first one chosen did, reported "invariant", and contradicted the numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp032_df_gauge import (
    absorbed_factors,
    mix,
    orthogonal_from,
    reconstruct,
    symbolic_gauge,
    two_body_one_norm,
)
from qres.factorization import double_factorize, molecular_integrals
from qres.problems.chemistry import build_molecule


def _h4():
    problem = build_molecule("H4")
    one_body, two_body, _ = molecular_integrals(problem)
    return one_body, two_body, double_factorize(one_body, two_body)


def test_the_one_norm_is_the_repositorys_own():
    """Load-bearing: optimise the wrong functional and the gain means nothing.

    `DoubleFactorization.one_norm` is written as
    `sum_t |lambda_t| (spin sum_p |d_tp|)^2 / 2` over unit-norm factors; this
    file works with `A_t = sqrt(lambda_t) L^t` and writes the same thing as
    `2 sum_t ||A_t||_*^2`. If the substitution were wrong the whole experiment
    would be minimising a quantity of its own invention.
    """
    for name in ("H2", "H4", "H6"):
        problem = build_molecule(name)
        one_body, two_body, _ = molecular_integrals(problem)
        factorisation = double_factorize(one_body, two_body)

        mine = two_body_one_norm(absorbed_factors(factorisation))
        one_body_part = 2 * float(np.abs(np.linalg.eigvalsh(factorisation.one_body)).sum())
        theirs = factorisation.one_norm() - one_body_part
        assert mine == pytest.approx(theirs, rel=1e-10), name


def test_the_absorbed_factors_rebuild_the_tensor():
    """`W = sum_t vec(A_t) vec(A_t)^T` must be the actual two-electron tensor."""
    _, two_body, factorisation = _h4()
    np.testing.assert_allclose(
        reconstruct(absorbed_factors(factorisation)), two_body, atol=1e-10
    )


def test_a_gauge_rotation_leaves_the_tensor_alone():
    """The gate on every result in the experiment."""
    _, two_body, factorisation = _h4()
    factors = absorbed_factors(factorisation)

    rng = np.random.default_rng(0)
    count = factors.shape[0]
    parameters = rng.normal(size=count * (count - 1) // 2)
    rotated = mix(factors, orthogonal_from(parameters, count))

    np.testing.assert_allclose(reconstruct(rotated), two_body, atol=1e-10)


def test_a_gauge_rotation_does_move_the_one_norm():
    """The asymmetry with THC, which is the point of the result.

    Result 86 proved THC's λ is invariant under its gauge. If DF's were too,
    there would be nothing to optimise and this experiment would be empty.
    """
    _, _, factorisation = _h4()
    factors = absorbed_factors(factorisation)
    count = factors.shape[0]

    base = two_body_one_norm(factors)
    rng = np.random.default_rng(1)
    moved = [
        two_body_one_norm(mix(factors, orthogonal_from(
            rng.normal(size=count * (count - 1) // 2), count)))
        for _ in range(5)
    ]
    assert not any(value == pytest.approx(base, rel=1e-6) for value in moved), (
        "the DF 1-norm is invariant under the gauge; if that is real the whole "
        "experiment has no object and should be withdrawn, not adjusted"
    )


def test_orthogonal_from_is_orthogonal():
    for count in (2, 5, 9):
        rng = np.random.default_rng(count)
        rotation = orthogonal_from(rng.normal(size=count * (count - 1) // 2), count)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(count), atol=1e-10)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-10)


def test_zero_parameters_give_the_identity():
    """So the first restart really does start from the unrotated factorisation."""
    np.testing.assert_allclose(orthogonal_from(np.zeros(10), 5), np.eye(5), atol=1e-14)


def test_absorbed_factors_refuses_a_negative_weight():
    """O(T) is the gauge group only when the weights are non-negative.

    Every molecule here is measured to have strictly positive weights (the ERI
    matrix is a Gram matrix), but on an indefinite one the group is the
    signature-preserving subgroup and this code would silently be wrong.
    """
    from qres.factorization import DoubleFactorization

    broken = DoubleFactorization(
        one_body=np.eye(2),
        factor_weights=np.array([1.0, -0.5]),
        factor_rotations=np.stack([np.eye(2), np.eye(2)]),
        factor_diagonals=np.ones((2, 2)),
    )
    with pytest.raises(ValueError, match="negative weights"):
        absorbed_factors(broken)


def test_the_symbolic_witness_is_not_on_an_invariant_orbit():
    """The bug that made the proof contradict the measurement.

    The first witness was Z = diag(1,-1) and X = [[0,1],[1,0]]; their real
    rotations are unit Bloch vectors with eigenvalues ±1 at every angle, so the
    nuclear norm never moved and the script reported invariance. This pins both
    halves of the corrected statement.
    """
    proof = symbolic_gauge()
    assert proof["tensor_invariant"], "the tensor must be exactly invariant"
    assert not proof["one_norm_invariant"], "the 1-norm must NOT be invariant"
    assert abs(proof["one_norm_difference_at_pi_over_4"]) > 1e-9

    # and the degenerate pair really is degenerate, which is why it was useless
    rotations = np.linspace(0.0, np.pi / 2, 7)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    norms = [
        np.abs(np.linalg.eigvalsh(np.cos(t) * pauli_z + np.sin(t) * pauli_x)).sum()
        for t in rotations
    ]
    assert np.allclose(norms, 2.0, atol=1e-12), (
        "the Z/X pair is supposed to be the degenerate case this test explains"
    )
