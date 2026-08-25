"""QAOA circuit construction and scoring."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from qres.problems.optimization import maxcut, portfolio_optimization, random_regular_maxcut
from qres.qaoa import _default_angles, _ising_terms, qaoa_circuit, run_qaoa


def test_ising_terms_round_trip():
    problem = random_regular_maxcut(8, 3, seed=2)
    singles, pairs = _ising_terms(problem.hamiltonian)
    assert len(singles) + len(pairs) == len(problem.hamiltonian)
    # a 3-regular MaxCut has only quadratic terms
    assert singles == []
    assert len(pairs) == 12


@pytest.mark.parametrize("reps", [1, 2])
def test_qaoa_circuit_matches_the_intended_unitary(reps):
    """Build exp(-i g H_C) exp(-i b H_M) by matrix exponential and compare.

    Qiskit's rotation convention (``r*(t) = exp(-i t/2 P)``) is exactly the sort
    of factor-of-two that produces a circuit which optimises to *something*
    while implementing the wrong operator.
    """
    problem = maxcut([(0, 1), (1, 2), (0, 2), (2, 3)])
    n = problem.num_qubits
    cost = problem.hamiltonian.to_matrix()
    mixer = sum(
        _pauli_matrix("X", q, n) for q in range(n)
    )

    rng = np.random.default_rng(0)
    gammas = rng.uniform(-1, 1, reps)
    betas = rng.uniform(-1, 1, reps)

    qc = qaoa_circuit(problem.hamiltonian, reps=reps)
    got = Statevector(qc.assign_parameters(np.concatenate([gammas, betas]))).data

    want = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    for layer in range(reps):
        want = expm(-1j * gammas[layer] * cost) @ want
        want = expm(-1j * betas[layer] * mixer) @ want

    # global phase is unobservable
    overlap = abs(np.vdot(want, got))
    assert overlap == pytest.approx(1.0, abs=1e-9)


def _pauli_matrix(letter: str, qubit: int, n: int) -> np.ndarray:
    single = {"X": np.array([[0, 1], [1, 0]], dtype=complex)}[letter]
    eye = np.eye(2, dtype=complex)
    out = np.array([[1]], dtype=complex)
    for q in range(n):  # qubit 0 is the least significant factor
        out = np.kron(single if q == qubit else eye, out)
    return out


def test_higher_order_terms_are_rejected_not_silently_dropped():
    from qiskit.quantum_info import SparsePauliOp

    ham = SparsePauliOp(["ZZZI", "IIZZ"], np.array([1.0, 1.0], dtype=complex))
    with pytest.raises(ValueError, match="higher-order"):
        qaoa_circuit(ham, reps=1)


def test_non_diagonal_hamiltonian_is_rejected():
    from qiskit.quantum_info import SparsePauliOp

    ham = SparsePauliOp(["XZ"], np.array([1.0], dtype=complex))
    with pytest.raises(ValueError, match="diagonal"):
        qaoa_circuit(ham, reps=1)


def test_default_angles_are_an_adiabatic_ramp():
    angles = _default_angles(4)
    gammas, betas = angles[:4], angles[4:]
    assert np.all(np.diff(gammas) > 0)  # cost weight grows
    assert np.all(np.diff(betas) < 0)  # mixer weight shrinks


def test_deeper_qaoa_widens_the_spread_faster_than_it_helps():
    """Depth costs reliability, and at this budget it does not buy a median.

    Measured on an 8-node 3-regular instance over 8 seeds:

        p=1  median AR 0.492, range [0.474, 0.506]
        p=2  median AR 0.513, range [0.188, 0.599]
        p=3  median AR 0.584, range [0.010, 0.706]

    Three of eight seeds at p=3 fail almost completely, and over seeds 0-5 the
    p=3 median (0.343) is *below* p=1's (0.492) -- the failure rate grows with
    depth faster than the reachable ratio improves at a fixed shot budget.

    So the robust claim is about spread, not about the median. An earlier
    version asserted single-seed monotonicity and passed only because shot noise
    was frozen; with noise resampled it is false, and so is the median version.
    """
    problem = random_regular_maxcut(8, 3, seed=5)

    def ratios(reps):
        return np.array([
            run_qaoa(
                problem,
                reps=reps,
                environment="ideal",
                optimizer="cobyla",
                shots=2048,
                shot_budget=60_000,
                maxiter=1000,
                seed=s,
            ).approximation_ratio
            for s in range(6)
        ])

    shallow, deep = ratios(1), ratios(3)
    assert deep.std() > 2 * shallow.std(), "depth should widen the spread"
    assert deep.max() > shallow.max(), "depth should raise the ceiling"


def test_scoring_is_consistent_with_brute_force():
    problem = portfolio_optimization(8, budget=4, seed=1)
    run = run_qaoa(
        problem,
        reps=2,
        environment="ideal",
        optimizer="cobyla",
        shots=2048,
        shot_budget=40_000,
        maxiter=200,
        seed=0,
    )
    # the best bitstring actually sampled can never beat the true optimum
    assert run.best_sampled >= run.optimal_value - 1e-9
    assert run.best_sampled_ratio <= 1.0 + 1e-9
    assert 0.0 <= run.optimal_probability <= 1.0
