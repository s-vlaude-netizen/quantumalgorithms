"""Covariance machinery: verified against Qiskit and dense matrices."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

from qres.covariance import (
    covariance_grouping,
    covariance_matrix,
    group_variance,
    pauli_expectations,
    predicted_total_variance,
)
from qres.measurement import group_paulis
from qres.problems.chemistry import build_molecule


def random_state(n: int, seed: int) -> Statevector:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return Statevector(vec / np.linalg.norm(vec))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pauli_expectations_match_qiskit(seed):
    """The bitmask fast path must agree with Qiskit exactly.

    Guards the phase convention: ``Pauli.phase`` is relative to ``(-i)^(x.z)``,
    so ``Pauli("Y").phase`` is 0 while its group phase is 1.  Using ``.phase``
    directly mis-signs every term containing a Y and is invisible on
    Y-free Hamiltonians.
    """
    n = 4
    state = random_state(n, seed)
    rng = np.random.default_rng(seed + 100)
    labels = sorted({"".join(rng.choice(list("IXYZ"), size=n)) for _ in range(40)})
    paulis = [Pauli(label) for label in labels]

    mine = pauli_expectations(paulis, state)
    theirs = np.array([state.expectation_value(p).real for p in paulis])
    np.testing.assert_allclose(mine, theirs, atol=1e-12)


def test_pauli_expectation_handles_y_terms():
    """Explicit Y coverage -- the case the phase convention breaks on."""
    state = random_state(3, 7)
    for label in ["YII", "IYI", "YYI", "YZY", "XYZ", "YYY"]:
        p = Pauli(label)
        got = pauli_expectations([p], state)[0]
        assert got == pytest.approx(state.expectation_value(p).real, abs=1e-12)


def test_covariance_matrix_matches_dense_computation():
    n = 3
    state = random_state(n, 3)
    psi = state.data
    rng = np.random.default_rng(11)
    labels = sorted({"".join(rng.choice(list("IXYZ"), size=n)) for _ in range(10)})
    ham = SparsePauliOp(labels, rng.normal(size=len(labels)).astype(complex))

    cov = covariance_matrix(ham, state)
    paulis = list(ham.paulis)
    for i, pi in enumerate(paulis):
        for j, pj in enumerate(paulis):
            mi, mj = pi.to_matrix(), pj.to_matrix()
            want = np.real(
                psi.conj() @ (mi @ mj) @ psi
                - (psi.conj() @ mi @ psi) * (psi.conj() @ mj @ psi)
            )
            assert cov[i, j] == pytest.approx(want, abs=1e-12)


def test_covariance_matrix_is_symmetric():
    state = random_state(3, 5)
    rng = np.random.default_rng(5)
    labels = sorted({"".join(rng.choice(list("IXYZ"), size=3)) for _ in range(12)})
    ham = SparsePauliOp(labels, rng.normal(size=len(labels)).astype(complex))
    cov = covariance_matrix(ham, state)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_group_variance_matches_direct_sampling_variance():
    """Var_g from the covariance matrix must equal the per-shot value variance."""
    problem = build_molecule("H4")
    coeffs = np.asarray(problem.hamiltonian.coeffs).real
    matrix = problem.hamiltonian.to_matrix()
    _, vectors = np.linalg.eigh(matrix)
    state = Statevector(np.ascontiguousarray(vectors[:, 0]))
    cov = covariance_matrix(problem.hamiltonian, state)

    groups, _ = group_paulis(problem.hamiltonian, "commuting")
    n = problem.num_qubits
    idx = np.arange(2**n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)

    for group in groups[:4]:
        rotated = state.evolve(group.basis_change) if group.basis_change else state
        probs = np.abs(rotated.data) ** 2
        values = group.shot_values(bits)
        mean = probs @ values
        want = probs @ (values - mean) ** 2
        got = group_variance(group.indices, coeffs, cov)
        assert got == pytest.approx(want, rel=1e-8, abs=1e-10)


def test_covariance_grouping_covers_every_term_exactly_once():
    problem = build_molecule("H4")
    reference = Statevector(np.ascontiguousarray(np.linalg.eigh(problem.hamiltonian.to_matrix())[1][:, 0]))
    groups, identity = covariance_grouping(problem.hamiltonian, reference)
    covered = sorted(i for g in groups for i in g.indices)
    expected = sorted(
        i for i, p in enumerate(problem.hamiltonian.paulis) if p.x.any() or p.z.any()
    )
    assert covered == expected
    for g in groups:
        for a in g.paulis:
            for b in g.paulis:
                assert a.commutes(b)


def test_grouping_is_deterministic_across_calls():
    """Near-ties in |coefficient| must not make the partition wobble.

    PySCF coefficients differ in the last few ulp between processes; without a
    stable tie-break the greedy takes a different path and a headline variance
    ratio moves by 0.15.
    """
    problem = build_molecule("LiH")
    signatures = []
    for _ in range(3):
        groups, _ = group_paulis(problem.hamiltonian, "commuting")
        signatures.append(tuple(tuple(sorted(g.indices)) for g in groups))
    assert signatures[0] == signatures[1] == signatures[2]

    # and stable under a perturbation below the rounding tolerance
    perturbed = SparsePauliOp(
        problem.hamiltonian.paulis,
        problem.hamiltonian.coeffs + 1e-15,
    )
    other, _ = group_paulis(perturbed, "commuting")
    assert tuple(tuple(sorted(g.indices)) for g in other) == signatures[0]


def test_predicted_variance_is_positive_and_matches_manual_sum():
    problem = build_molecule("H4")
    coeffs = np.asarray(problem.hamiltonian.coeffs).real
    state = Statevector(np.ascontiguousarray(np.linalg.eigh(problem.hamiltonian.to_matrix())[1][:, 0]))
    cov = covariance_matrix(problem.hamiltonian, state)
    groups, _ = group_paulis(problem.hamiltonian, "commuting")

    total = predicted_total_variance(groups, cov, coeffs)
    manual = sum(np.sqrt(max(group_variance(g.indices, coeffs, cov), 0.0)) for g in groups) ** 2
    assert total == pytest.approx(manual)
    assert total > 0
