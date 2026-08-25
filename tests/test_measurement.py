"""Correctness of the measurement machinery.

The dangerous failure mode here is a basis change that is *almost* right: the
energies stay plausible and every downstream benchmark is quietly wrong.  So we
check the grouping algebraically against exact statevector expectation values,
with no sampling noise anywhere in the comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qres.measurement import (
    allocate_shots,
    group_paulis,
    qwc_compatible,
    VarianceModel,
)


def random_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        for q in range(n):
            qc.rx(rng.uniform(0, 2 * np.pi), q)
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def exact_group_expectations(group, state: Statevector, n: int) -> np.ndarray:
    """<P_i> for every Pauli in the group, computed through the basis change.

    This is the exact analogue of what the sampler does: rotate, then read
    Z-parities off the computational-basis probabilities.
    """
    rotated = state.evolve(group.basis_change) if group.basis_change else state
    probs = np.abs(rotated.data) ** 2
    idx = np.arange(2**n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    par = (bits @ group.diagonal_z.T.astype(np.int8)) & 1
    signed = np.where(par == 1, -1.0, 1.0) * group.signs[None, :]
    return probs @ signed


@pytest.mark.parametrize("method", ["none", "qwc", "commuting"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_grouping_reproduces_exact_expectations(method, seed):
    """Every grouped measurement must reproduce <P> to machine precision."""
    n = 4
    rng = np.random.default_rng(seed)
    n_terms = 20
    labels, coeffs = set(), []
    while len(labels) < n_terms:
        labels.add("".join(rng.choice(list("IXYZ"), size=n)))
    labels = sorted(labels)
    coeffs = rng.normal(size=len(labels))
    ham = SparsePauliOp(labels, coeffs.astype(complex))

    groups, identity = group_paulis(ham, method=method)
    state = Statevector(random_circuit(n, 3, seed))

    # every term is covered exactly once
    covered = sorted(i for g in groups for i in g.indices)
    expected = sorted(i for i, l in enumerate(labels) if set(l) != {"I"})
    assert covered == expected

    total = identity
    for g in groups:
        got = exact_group_expectations(g, state, n)
        want = np.array([state.expectation_value(p).real for p in g.paulis])
        np.testing.assert_allclose(got, want, atol=1e-10, err_msg=f"{method} group {g.indices}")
        total += float(got @ g.coeffs)

    assert total == pytest.approx(state.expectation_value(ham).real, abs=1e-9)


@pytest.mark.parametrize("method", ["qwc", "commuting"])
def test_grouping_members_actually_commute(method):
    n = 5
    rng = np.random.default_rng(7)
    labels = sorted({"".join(rng.choice(list("IXYZ"), size=n)) for _ in range(60)})
    ham = SparsePauliOp(labels, rng.normal(size=len(labels)).astype(complex))
    groups, _ = group_paulis(ham, method=method)
    for g in groups:
        for a in g.paulis:
            for b in g.paulis:
                assert a.commutes(b)
                if method == "qwc" and g.kind == "qwc":
                    assert qwc_compatible(a, b)


def test_commuting_grouping_is_not_worse_than_qwc():
    """General commuting must produce at most as many groups as QWC."""
    from qres.problems.chemistry import build_molecule

    problem = build_molecule("H4")
    qwc, _ = group_paulis(problem.hamiltonian, method="qwc")
    gc, _ = group_paulis(problem.hamiltonian, method="commuting")
    assert len(gc) <= len(qwc)


def test_allocate_shots_respects_total_and_floor():
    for total in (100, 1000, 12345):
        for g in (1, 3, 17):
            w = np.abs(np.random.default_rng(g).normal(size=g)) + 1e-9
            counts = allocate_shots(total, w, min_shots=8)
            assert counts.sum() == total
            assert (counts >= 1).all()
            assert len(counts) == g


def test_allocate_shots_is_monotone_in_variance():
    counts = allocate_shots(100_000, np.array([1.0, 4.0, 9.0]), min_shots=8)
    # n ∝ sqrt(var) = 1 : 2 : 3
    ratios = counts / counts[0]
    np.testing.assert_allclose(ratios, [1, 2, 3], rtol=0.02)


def test_variance_model_shrinks_towards_empirical():
    model = VarianceModel(prior_variance=np.array([100.0]), prior_strength=64.0)
    assert model.variances()[0] == pytest.approx(100.0)
    rng = np.random.default_rng(0)
    model.update(0, rng.normal(0, 1, size=64))
    # halfway between prior and empirical after n == prior_strength
    assert 40 < model.variances()[0] < 60
    model.update(0, rng.normal(0, 1, size=10_000))
    assert model.variances()[0] < 5.0
