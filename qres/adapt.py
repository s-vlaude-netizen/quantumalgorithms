"""ADAPT-VQE: grow the ansatz one operator at a time from measured gradients.

This is the only direction the measurements support (RESEARCH_LOG Result 43).
The shot cost scales as ``(Σ|c|)² · n / ε²`` with ``Σ|c| ~ N^2.78``, so

    shots  ~  N^5.55 · n / ε²

and ``n`` is the one factor nothing else here touches.  UCCSD's ``n ~ N⁴`` gives
``N^9.6``, against CCSD(T)'s ``N⁷`` in operations.  An ansatz reaching the same
accuracy with ``n ~ N²`` would give ``N^7.6`` -- the same order as the classical
competitor, which is where a comparison would begin to be interesting.

ADAPT-VQE (Grimsley et al. 2019) builds the ansatz greedily: at each step it
measures ``∂E/∂θ_k = <ψ|[H, A_k]|ψ>`` for every operator in a pool, appends the
one with the largest gradient, and re-optimises everything.  It reaches a given
accuracy with far fewer parameters than a fixed UCCSD -- how many fewer is an
empirical question, and the first thing to measure before building the
shot-based version.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector


@dataclass
class AdaptStep:
    """One growth step: which operator was added and what it bought."""

    index: int
    operator_label: str
    gradient: float
    energy: float
    parameters: int
    seconds: float


@dataclass
class AdaptResult:
    energy: float
    parameters: np.ndarray
    operators: list[int]
    steps: list[AdaptStep] = field(default_factory=list)
    converged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_parameters(self) -> int:
        return len(self.operators)


def excitation_pool(problem, kind: str = "sd") -> list[SparsePauliOp]:
    """Anti-Hermitian excitation generators, mapped to qubit operators.

    The pool is what ADAPT chooses from, so its content sets both the reachable
    accuracy and the cost of each growth step (one gradient per operator).
    ``"sd"`` is the standard singles-and-doubles pool; ``"d"`` drops the
    singles, which measurements here show contribute almost nothing to the
    correlation energy (Result 16) while costing a gradient each.
    """
    from qiskit_nature.second_q.circuit.library import UCC

    from .ansatz import _mapper_for

    mapper = _mapper_for(problem)
    ansatz = UCC(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        excitations=kind,
        qubit_mapper=mapper,
    )
    # UCC exposes its generators as the operators it would exponentiate
    operators = []
    for operator in ansatz.operators:
        simplified = operator.simplify(atol=1e-12)
        if len(simplified) > 0:
            operators.append(simplified)
    return operators


def pauli_action(pauli) -> tuple[int, int, complex]:
    """A Pauli string as ``(z_mask, x_mask, phase)`` -- three integers, no matrix.

    Applying ``P`` to a statevector is an index permutation with signs, not a
    matrix product::

        (P psi)[k] = phase * (-1)^popcount(z & k) * psi[k XOR x]

    Storing sparse matrices instead costs 2^n entries *per Pauli term*, which is
    what drove the H8 memory blow-ups; this costs three integers.

    The phase convention is Qiskit's and was determined **empirically** rather
    than recalled: ``(-i)^(pauli.phase + |z AND x|)``, the second term because
    ``Z X = iY`` on every qubit carrying both.  A first version omitted it and
    was wrong by ``-i`` on every operator containing a Y.  Verified against
    ``to_matrix`` on 200 random Paulis with group phases, to exactly zero.
    """
    z_mask = int(sum(int(bit) << i for i, bit in enumerate(pauli.z)))
    x_mask = int(sum(int(bit) << i for i, bit in enumerate(pauli.x)))
    both = int(np.sum(pauli.z & pauli.x))
    phase = (-1j) ** (int(pauli.phase) + both)
    return z_mask, x_mask, phase


def apply_pauli(vector: np.ndarray, action: tuple[int, int, complex]) -> np.ndarray:
    """``P |psi>`` from a prepared :func:`pauli_action`."""
    z_mask, x_mask, phase = action
    indices = np.arange(vector.shape[0])
    signs = np.where(np.bitwise_count(indices & z_mask) & 1, -1.0, 1.0)
    permuted = vector[indices ^ x_mask] if x_mask else vector
    return phase * signs * permuted


def prepare_operator(operator: SparsePauliOp) -> list[tuple[float, tuple[int, int, complex]]]:
    """One generator as ``[(coefficient, pauli action), ...]``."""
    return [
        (float(np.real(coefficient)), pauli_action(pauli))
        for coefficient, pauli in zip(operator.coeffs, operator.paulis)
    ]


def apply_operator(vector: np.ndarray, terms) -> np.ndarray:
    """``A |psi>`` for ``A = sum_j c_j P_j``."""
    out = np.zeros_like(vector)
    for coefficient, action in terms:
        out += coefficient * apply_pauli(vector, action)
    return out


class LazyPool:
    """Prepared generators, built on first use and memoised.

    Preparing the *whole* pool eagerly is the same O(pool x 2^n) mistake the
    commutator cache made, and it is worse here because a Pauli matrix is dense
    in its nonzero count: at 14 qubits, ~4 000 operators x ~8 terms x 16 384
    entries is several gigabytes, which is what an H8 run was climbing towards.

    ADAPT only ever evolves with the operators it has **chosen** -- at most
    ``max_operators``, so ~150 rather than ~4 000. Preparing on demand makes the
    memory scale with the answer instead of with the search space.
    """

    __slots__ = ("_pool", "_prepared")

    def __init__(self, pool: list[SparsePauliOp]) -> None:
        self._pool = pool
        self._prepared: dict[int, list[tuple[float, Any]]] = {}

    def __getitem__(self, index: int) -> list[tuple[float, Any]]:
        terms = self._prepared.get(index)
        if terms is None:
            terms = prepare_operator(self._pool[index])
            self._prepared[index] = terms
        return terms

    def __len__(self) -> int:
        return len(self._pool)

    @property
    def prepared_count(self) -> int:
        """How many were actually built -- the point of the class."""
        return len(self._prepared)


def prepared_pool(pool: list[SparsePauliOp]) -> LazyPool:
    """Lazy prepared generators for ``pool``. See :class:`LazyPool`."""
    return LazyPool(pool)


def apply_evolution(vector: np.ndarray, terms: list[tuple[float, Any]], theta: float) -> np.ndarray:
    """``exp(-i theta A) |psi>`` in closed form, for commuting-Pauli ``A``.

    The Pauli strings inside one fermionic excitation generator all commute, so
    the exponential factorises, and each factor is exact because ``P^2 = I``:

        exp(-i a P) = cos(a) I - i sin(a) P

    That replaces a ``PauliEvolutionGate`` synthesis and a ``Statevector.evolve``
    per parameter per energy evaluation with two scalar multiplies and one
    sparse matvec.  Verified against the gate path to 3e-17.
    """
    out = vector
    for coefficient, action in terms:
        angle = coefficient * theta
        out = np.cos(angle) * out - 1j * np.sin(angle) * apply_pauli(out, action)
    return out


def operator_gradients(
    hamiltonian: SparsePauliOp,
    state: Statevector,
    pool: list[SparsePauliOp],
    *,
    prepared: Any | None = None,
    hamiltonian_matrix: Any | None = None,
) -> np.ndarray:
    """``|<psi| i[H, A_k] |psi>|`` for every operator in the pool.

    The factor of ``i`` is not cosmetic.  qiskit-nature's UCC generators come
    back **Hermitian** (coefficients +-0.5, real), so ``[H, A]`` is
    *anti*-Hermitian and ``<[H, A]>`` is purely imaginary: taking its real part
    gives exactly zero for every operator in the pool, which reads as "no
    operator has any gradient" and stops ADAPT at zero parameters.  Measured on
    H4: ``max |Re<[H,A]>| = 0.0`` against ``max |Im<[H,A]>| = 2.78e-1``.

    Exact, from the statevector.  The shot-based version of this is what the
    grouped estimator exists for, but the parameter-count question this module
    was written to answer does not need it -- and answering it exactly first is
    much cheaper than answering it approximately.

    **No commutator is ever built.**  For Hermitian ``H`` and ``A``, with
    ``phi = H|psi>``::

        <[H, A]> = <phi|A|psi> - conj(<phi|A|psi>) = 2i Im <phi|A|psi>

    so the whole pool needs *one* shared matvec plus one cheap ``A|psi>`` each.
    The explicit route -- ``H@A - A@H`` then ``simplify`` -- is quadratic in the
    Hamiltonian's term count and was the reason H8 never finished: 2 913 terms
    against a 360-operator pool gives commutators of ~23 000 Pauli terms, rebuilt
    every growth step.  :func:`operator_gradients_reference` keeps that version,
    and the tests hold the two against each other.
    """
    vector = np.asarray(state.data if isinstance(state, Statevector) else state)

    # phi = H|psi>, one matvec shared by the whole pool
    if hamiltonian_matrix is None:
        hamiltonian_matrix = hamiltonian.to_matrix(sparse=True).tocsr()
    phi = hamiltonian_matrix @ vector

    if prepared is None:
        prepared = prepared_pool(list(pool))

    gradients = np.zeros(len(pool))
    for k in range(len(pool)):
        gradients[k] = abs(2.0 * float(np.imag(np.vdot(phi, apply_operator(vector, prepared[k])))))
    return gradients


def operator_gradients_reference(
    hamiltonian: SparsePauliOp, state: Statevector, pool: list[SparsePauliOp]
) -> np.ndarray:
    """The explicit-commutator version, kept as the thing to check against.

    Slow and obviously correct: it builds ``[H, A_k]`` symbolically and takes an
    expectation. That is what :func:`operator_gradients` used to do on every
    growth step, and on H8 -- 2 913 Hamiltonian terms against a 360-operator pool
    -- each commutator carries up to ~23 000 Pauli terms, which is why it never
    finished.

    Retained because a fast path needs something to be verified against, and the
    identity that replaced it is not obvious enough to trust untested.
    """
    gradients = np.zeros(len(pool))
    for k, operator in enumerate(pool):
        commutator = (hamiltonian @ operator - operator @ hamiltonian).simplify(atol=1e-12)
        if len(commutator) == 0:
            continue
        gradients[k] = abs(float(np.imag(complex(state.expectation_value(commutator)))))
    return gradients


def _evolution(operator: SparsePauliOp, theta: float):
    """``exp(-i theta A)`` for a Hermitian pool operator.

    ``PauliEvolutionGate`` applies each Pauli term in turn.  That is *exact*
    rather than a Trotter approximation here, because the Pauli terms within a
    single fermionic excitation generator all commute with each other -- which
    is the same fact that lets UCC Trotterise one excitation without error.
    Verified against ``scipy.linalg.expm`` on the full matrix.
    """
    from qiskit.circuit.library import PauliEvolutionGate

    return PauliEvolutionGate(operator, time=theta)


def adapt_vqe(
    problem,
    *,
    pool_kind: str = "sd",
    max_operators: int = 30,
    gradient_tolerance: float = 1e-3,
    energy_tolerance: float | None = None,
    optimizer_maxiter: int = 2000,
    batch: int = 5,
    lazy: bool = True,
    verbose: bool = False,
) -> AdaptResult:
    """Exact-arithmetic ADAPT-VQE, in the cheapest schedule measured.

    Deliberately noiseless: the question this answers is *how many operators*
    are needed for a given accuracy, which is a property of the ansatz and the
    molecule, not of the estimator.  Mixing in shot noise would only obscure it,
    and Result 25 already established that the noise is a separate problem.

    ``batch`` and ``lazy`` default to the best combination measured
    (RESEARCH_LOG Result 47).  Standard ADAPT is ``batch=1, lazy=False``, and it
    is **4.6x more expensive**: its cost is *(growth steps)* x *(cost of one
    re-optimisation)*, and the defaults shrink both factors.

    ``lazy`` optimises only the newly added parameters at each step, with one
    full pass at the end -- 3.1x on its own.  ``batch`` adds several operators
    per step, trading a worse choice for operators 2..b (they are picked from
    gradients measured before operator 1 was added) against far fewer
    re-optimisations.

    Together on H4 they reach chemical accuracy in 141 evaluations against fixed
    UCCSD's 134 -- cost parity -- using 10 parameters instead of 26, which is
    2.0x fewer two-qubit gates and 18x more surviving signal on a Heron-class
    device.  Set ``batch=2, lazy=False`` for an order of magnitude better
    accuracy (1.08e-4) at 3x the cost.
    """
    from scipy.optimize import minimize

    from .ansatz import hartree_fock_state
    from .problems.chemistry import CHEMICAL_ACCURACY_HA

    energy_tolerance = energy_tolerance or CHEMICAL_ACCURACY_HA
    pool = excitation_pool(problem, pool_kind)
    reference = Statevector(hartree_fock_state(problem.hf_bitstring))

    chosen: list[int] = []
    parameters = np.zeros(0)
    steps: list[AdaptStep] = []
    converged = False

    # Built once for the whole run.  Both were previously rebuilt inside the
    # inner loops -- the commutators once per growth step, the Pauli matrices
    # once per parameter per energy evaluation.  See Result 73.
    evolution_terms = prepared_pool(pool)
    hamiltonian_matrix = problem.hamiltonian.to_matrix(sparse=True).tocsr()
    reference_vector = np.asarray(reference.data)

    def build(params, operators) -> np.ndarray:
        """Apply exp(theta_k A_k) in the order the operators were chosen."""
        vector = reference_vector
        for theta, index in zip(params, operators):
            vector = apply_evolution(vector, evolution_terms[index], theta)
        return vector

    def energy_of(params, operators) -> float:
        vector = build(params, operators)
        return float(np.real(np.vdot(vector, hamiltonian_matrix @ vector)))

    for step in range(max_operators):
        t0 = time.perf_counter()
        state = build(parameters, chosen)
        gradients = operator_gradients(
            problem.hamiltonian, state, pool,
            prepared=evolution_terms, hamiltonian_matrix=hamiltonian_matrix,
        )
        picked = [int(i) for i in np.argsort(-gradients) if gradients[i] >= gradient_tolerance]
        picked = picked[: max(1, batch)]

        if not picked:
            converged = True
            break

        best = picked[0]
        largest = float(gradients[best])
        added = len(picked)
        chosen.extend(picked)
        parameters = np.concatenate([parameters, np.zeros(added)])

        if lazy:
            fixed = parameters[:-added]

            def partial(tail):
                return energy_of(np.concatenate([fixed, tail]), chosen)

            result = minimize(
                partial, parameters[-added:], method="COBYLA",
                options={"maxiter": optimizer_maxiter},
            )
            parameters = np.concatenate([fixed, np.asarray(result.x)])
        else:
            result = minimize(
                energy_of, parameters, args=(chosen,), method="COBYLA",
                options={"maxiter": optimizer_maxiter},
            )
            parameters = np.asarray(result.x)
        energy = float(result.fun)

        steps.append(
            AdaptStep(
                index=best,
                operator_label=str(pool[best].paulis[0]),
                gradient=largest,
                energy=energy,
                parameters=len(chosen),
                seconds=time.perf_counter() - t0,
            )
        )
        if verbose:
            print(
                f"  step {step + 1:2d}: op {best:3d} grad={largest:.3e} "
                f"E={energy:+.8f} err={abs(energy - problem.fci_energy):.3e}",
                flush=True,
            )
        if abs(energy - problem.fci_energy) < energy_tolerance:
            converged = True
            break

    if lazy and chosen and abs(energy_of(parameters, chosen) - problem.fci_energy) >= energy_tolerance:
        # the one full pass the lazy schedule defers to the end
        result = minimize(
            energy_of, parameters, args=(chosen,), method="COBYLA",
            options={"maxiter": optimizer_maxiter * 5},
        )
        parameters = np.asarray(result.x)

    final = energy_of(parameters, chosen) if len(chosen) else float(
        np.real(np.vdot(reference_vector, hamiltonian_matrix @ reference_vector))
    )
    return AdaptResult(
        energy=final,
        parameters=parameters,
        operators=chosen,
        steps=steps,
        converged=converged,
        metadata={"pool_size": len(pool), "pool_kind": pool_kind},
    )


def adapt_circuit(problem, pool, operators, parameters=None):
    """The ADAPT ansatz as a parameterised circuit the estimator can run.

    Hartree-Fock preparation followed by one ``exp(-i theta_k A_k)`` per chosen
    operator, in the order they were chosen.  Left parameterised so the shot
    estimator can transpile once and re-bind, which is what makes a shot-based
    run affordable at all (RESEARCH_LOG Result 28).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    from .ansatz import hartree_fock_state

    circuit = QuantumCircuit(problem.num_qubits, name=f"adapt{len(operators)}")
    circuit.compose(hartree_fock_state(problem.hf_bitstring), inplace=True)
    if not operators:
        return circuit

    theta = ParameterVector("t", len(operators))
    for k, index in enumerate(operators):
        angle = theta[k] if parameters is None else float(parameters[k])
        circuit.append(_evolution(pool[index], angle), range(problem.num_qubits))
    return circuit


def sampled_operator_gradients(
    problem, pool, operators, parameters, environment, shots_per_operator: int, ledger
) -> np.ndarray:
    """Pool gradients estimated from shots rather than from a statevector.

    Each gradient is the expectation of ``i[H, A_k]``, which is an ordinary
    observable -- so it goes through the same grouped estimator as an energy and
    is charged to the same ledger.

    Gradients are measured at ``shots_per_operator``, which is deliberately far
    below what an energy needs: they only have to *rank* operators, and ranking
    survives sigma = 0.01 where chemical accuracy needs 1.6e-3 (Result 44).
    """
    from .estimator import ShotEstimator

    circuit = adapt_circuit(problem, pool, operators)
    gradients = np.zeros(len(pool))
    for k, operator in enumerate(pool):
        commutator = (problem.hamiltonian @ operator - operator @ problem.hamiltonian)
        commutator = (1j * commutator).simplify(atol=1e-12)
        if len(commutator) == 0:
            continue
        estimator = ShotEstimator(
            commutator, circuit, environment, grouping="commuting",
            allocation="adaptive", ledger=ledger,
        )
        bound = np.asarray(parameters, dtype=float) if len(operators) else np.zeros(0)
        gradients[k] = abs(estimator.estimate(bound, shots_per_operator).value)
    return gradients

