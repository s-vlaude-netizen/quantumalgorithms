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


#: Above this state-space dimension the per-operator commutators are kept
#: symbolically instead of as sparse matrices.  Materialising them is faster per
#: gradient but costs O(pool x 2^n) memory, which is fine at 10 qubits (H6: ~1300
#: operators x 1024) and ruinous at 14 (H8: ~4000 x 16384, hundreds of MB each).
#: The expensive part being cached is the symbolic Pauli algebra either way.
MAX_MATERIALISED_DIMENSION = 1 << 13


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


def commutator_cache(
    hamiltonian: SparsePauliOp, pool: list[SparsePauliOp]
) -> list[Any]:
    """Sparse ``i[H, A_k]`` for the whole pool, built once.

    The commutators do not depend on the state, but ``adapt_vqe`` was rebuilding
    and re-``simplify``-ing all of them on **every growth step** -- pool-size
    symbolic Pauli algebra repeated once per step, for an answer that never
    changes.  On H6 that is ~1 300 commutators per step.

    Returning them as sparse matrices also replaces
    ``Statevector.expectation_value(SparsePauliOp)`` with one matvec.
    ``None`` marks an operator that commutes with ``H`` and can never have a
    gradient.
    """
    # Materialising every commutator as a sparse matrix is O(pool x 2^n) in
    # memory, and that is not a small constant: at 16 qubits a commutator with a
    # few hundred Pauli terms is hundreds of MB, and the pool has thousands of
    # them.  A first version did exactly this and drove an H8 run past a gigabyte
    # and climbing before it was killed.
    #
    # The expensive part was never the matrix -- it is the symbolic Pauli algebra
    # and `simplify`.  So the commutators are always cached (cheap: a few hundred
    # terms each), and only *materialised* while the dimension stays small enough
    # for the matvec to be the faster path.
    dimension = 1 << hamiltonian.num_qubits
    materialise = dimension <= MAX_MATERIALISED_DIMENSION

    cache: list[Any] = []
    for operator in pool:
        commutator = (hamiltonian @ operator - operator @ hamiltonian).simplify(atol=1e-12)
        if len(commutator) == 0:
            cache.append(None)
        elif materialise:
            cache.append(commutator.to_matrix(sparse=True).tocsr())
        else:
            cache.append(commutator)
    return cache


def prepared_pool(pool: list[SparsePauliOp]) -> list[list[tuple[float, Any]]]:
    """Each generator as ``[(coefficient, sparse Pauli), ...]``, built once.

    Feeds the closed-form evolution below.  The Pauli matrices are the part worth
    caching: they are fixed for the run and were previously being rebuilt inside
    every energy evaluation via circuit synthesis.
    """
    prepared = []
    for operator in pool:
        terms = [
            (float(np.real(coefficient)), pauli.to_matrix(sparse=True).tocsr())
            for coefficient, pauli in zip(operator.coeffs, operator.paulis)
        ]
        prepared.append(terms)
    return prepared


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
    for coefficient, matrix in terms:
        angle = coefficient * theta
        out = np.cos(angle) * out - 1j * np.sin(angle) * (matrix @ out)
    return out


def operator_gradients(
    hamiltonian: SparsePauliOp,
    state: Statevector,
    pool: list[SparsePauliOp],
    *,
    cache: list[Any] | None = None,
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
    """
    gradients = np.zeros(len(pool))

    if cache is not None:
        vector = np.asarray(state.data if isinstance(state, Statevector) else state)
        wrapped = None  # built lazily, only if the cache holds symbolic entries
        for k, entry in enumerate(cache):
            if entry is None:
                continue
            if isinstance(entry, SparsePauliOp):
                if wrapped is None:
                    wrapped = Statevector(vector)
                value = complex(wrapped.expectation_value(entry))
            else:
                value = np.vdot(vector, entry @ vector)
            gradients[k] = abs(float(np.imag(value)))
        return gradients

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
    gradient_cache = commutator_cache(problem.hamiltonian, pool)
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
            problem.hamiltonian, state, pool, cache=gradient_cache
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

