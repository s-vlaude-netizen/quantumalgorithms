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
            # tolerance follows COVARIANCE_DECIMALS: the matrix is rounded to
            # 1e-9 so the partition cannot depend on ulp-scale noise
            assert cov[i, j] == pytest.approx(want, abs=1e-9)


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
        # Var_g sums up to ~900 rounded covariances, so the 1e-9 rounding can
        # accumulate to ~1e-6 here.  Against group variances of order 0.01-1
        # that is 5+ orders below anything physical.
        assert got == pytest.approx(want, rel=1e-5, abs=1e-6)


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


def test_refinement_never_increases_predicted_variance():
    """Local search is steepest-descent, so the objective can only fall."""
    from qres.covariance import refine_partition

    problem = build_molecule("H4")
    coeffs = np.asarray(problem.hamiltonian.coeffs).real
    state = Statevector(
        np.ascontiguousarray(np.linalg.eigh(problem.hamiltonian.to_matrix())[1][:, 0])
    )
    cov = covariance_matrix(problem.hamiltonian, state)
    paulis = list(problem.hamiltonian.paulis)
    groups, _ = group_paulis(problem.hamiltonian, "commuting")
    start = [list(g.indices) for g in groups]

    def objective(parts):
        return float(sum(np.sqrt(max(group_variance(p, coeffs, cov), 0.0)) for p in parts) ** 2)

    before = objective(start)
    refined, moves = refine_partition(
        start, coeffs, cov, lambda a, b: paulis[a].commutes(paulis[b]), max_sweeps=400
    )
    assert objective(refined) <= before + 1e-12
    assert moves > 0


def test_refinement_preserves_the_partition_and_commutation():
    from qres.covariance import refined_covariance_grouping
    from qres.ansatz import hartree_fock_state

    problem = build_molecule("H4")
    reference = Statevector(hartree_fock_state(problem.hf_bitstring))
    groups, _, _ = refined_covariance_grouping(problem.hamiltonian, reference, "commuting")

    covered = sorted(i for g in groups for i in g.indices)
    expected = sorted(
        i for i, p in enumerate(problem.hamiltonian.paulis) if p.x.any() or p.z.any()
    )
    assert covered == expected, "refinement must not lose or duplicate a term"
    assert all(len(g.indices) > 0 for g in groups), "no empty groups"
    for g in groups:
        for a in g.paulis:
            for b in g.paulis:
                assert a.commutes(b), "refinement must not break simultaneous measurability"


def test_refinement_is_deterministic_across_calls():
    from qres.covariance import refined_covariance_grouping
    from qres.ansatz import hartree_fock_state

    problem = build_molecule("H4")
    reference = Statevector(hartree_fock_state(problem.hf_bitstring))
    signatures = []
    for _ in range(2):
        groups, _, moves = refined_covariance_grouping(problem.hamiltonian, reference, "commuting")
        signatures.append((moves, tuple(tuple(sorted(g.indices)) for g in groups)))
    assert signatures[0] == signatures[1]


def test_shot_estimator_accepts_covariance_groupings():
    """The covariance groupings must be reachable from the normal estimator path."""
    from qres.ansatz import build_ansatz
    from qres.estimator import ShotEstimator
    from qres.noise import make_environment

    problem = build_molecule("H4")
    env = make_environment("ideal")
    ansatz = build_ansatz("hea:1:linear:cry", problem, env)

    reports = {}
    for method in ("commuting", "covariance", "covariance+refine"):
        est = ShotEstimator(problem.hamiltonian, ansatz, env, grouping=method)
        reports[method] = est.report
        assert est.report["method"] == method
        assert est.report["num_terms"] == len(problem.hamiltonian) - 1  # identity dropped
        # the estimator must still produce the right energy
        params = np.zeros(ansatz.num_parameters)
        assert est.exact(params) == pytest.approx(problem.hartree_fock_energy, abs=1e-9)

    assert reports["covariance+refine"]["refinement_moves"] > 0
    assert reports["commuting"]["refinement_moves"] == 0


def test_covariance_grouping_refuses_rather_than_silently_degrading():
    """Beyond the simulable size it must raise, not quietly fall back."""
    from qres.ansatz import hardware_efficient
    from qres.estimator import ShotEstimator
    from qres.noise import make_environment

    n = ShotEstimator.MAX_COVARIANCE_QUBITS + 2
    ham = SparsePauliOp(["Z" * n, "X" * n], np.array([1.0, 0.5], dtype=complex))
    ansatz = hardware_efficient(n, reps=1)
    env = make_environment("ideal")
    with pytest.raises(ValueError, match="MAX_COVARIANCE_QUBITS"):
        ShotEstimator(ham, ansatz, env, grouping="covariance")


def test_sampled_energy_agrees_with_exact_for_every_grouping():
    """A grouping change must never move the energy, only its variance."""
    from qres.ansatz import build_ansatz
    from qres.estimator import ShotEstimator
    from qres.noise import make_environment

    problem = build_molecule("H4")
    env = make_environment("ideal")
    ansatz = build_ansatz("hea:1:linear:cry", problem, env)
    rng = np.random.default_rng(0)
    params = rng.normal(0, 0.3, ansatz.num_parameters)

    for method in ("commuting", "covariance", "covariance+refine"):
        est = ShotEstimator(problem.hamiltonian, ansatz, env, grouping=method)
        exact = est.exact(params)
        result = est.estimate(params, 400_000)
        # within 5 sigma of the exact value, and sigma itself must be sane
        assert result.stderr > 0
        assert abs(result.value - exact) < 5 * result.stderr + 1e-9


def test_grouping_survives_a_numerically_equivalent_reference():
    """Two references that are the same state must give the same partition.

    The Hartree-Fock determinant built as a bare circuit and the UCCSD ansatz
    evaluated at theta = 0 are the same state -- overlap 1.000000, max amplitude
    difference 2.6e-12. Before covariances were rounded, those two produced
    **11 and 19 groups** on H4, because the ~1e-12 difference in the covariance
    matrix flipped near-ties in the greedy.

    That is the same failure mode as the coefficient tie-break, one level down,
    and it silently changed which partition an end-to-end experiment measured.
    """
    from qres.ansatz import build_ansatz, hartree_fock_state
    from qres.covariance import refined_covariance_grouping

    problem = build_molecule("H4")
    bare = Statevector(hartree_fock_state(problem.hf_bitstring))
    ansatz = build_ansatz("uccsd", problem)
    at_zero = Statevector(ansatz.assign_parameters(np.zeros(ansatz.num_parameters)))

    assert abs(bare.inner(at_zero)) == pytest.approx(1.0, abs=1e-9)
    assert 0 < np.abs(bare.data - at_zero.data).max() < 1e-9, "should differ at ulp scale"

    first, _, moves_a = refined_covariance_grouping(problem.hamiltonian, bare, "commuting")
    second, _, moves_b = refined_covariance_grouping(problem.hamiltonian, at_zero, "commuting")

    assert [sorted(g.indices) for g in first] == [sorted(g.indices) for g in second]
    assert moves_a == moves_b


def test_covariance_matrix_is_rounded_below_the_physical_scale():
    """A perturbation far below anything physical must not change the matrix."""
    from qres.covariance import COVARIANCE_DECIMALS

    state = random_state(3, 9)
    rng = np.random.default_rng(9)
    labels = sorted({"".join(rng.choice(list("IXYZ"), size=3)) for _ in range(10)})
    ham = SparsePauliOp(labels, rng.normal(size=len(labels)).astype(complex))

    perturbed = Statevector(state.data + 1e-13 * rng.normal(size=state.data.shape))
    perturbed = Statevector(perturbed.data / np.linalg.norm(perturbed.data))

    np.testing.assert_array_equal(
        covariance_matrix(ham, state), covariance_matrix(ham, perturbed)
    )
    assert COVARIANCE_DECIMALS <= 12
