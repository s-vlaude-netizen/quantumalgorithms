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


def operator_gradients(
    hamiltonian: SparsePauliOp, state: Statevector, pool: list[SparsePauliOp]
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
    verbose: bool = False,
) -> AdaptResult:
    """Exact-arithmetic ADAPT-VQE, to measure how few parameters suffice.

    Deliberately noiseless: the question this answers is *how many operators*
    are needed for a given accuracy, which is a property of the ansatz and the
    molecule, not of the estimator.  Mixing in shot noise would only obscure it,
    and Result 25 already established that the noise is a separate problem.
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

    def build(params, operators) -> Statevector:
        """Apply exp(theta_k A_k) in the order the operators were chosen."""
        state = reference
        for theta, index in zip(params, operators):
            state = state.evolve(_evolution(pool[index], theta))
        return state

    def energy_of(params, operators) -> float:
        return float(np.real(build(params, operators).expectation_value(problem.hamiltonian)))

    for step in range(max_operators):
        t0 = time.perf_counter()
        state = build(parameters, chosen)
        gradients = operator_gradients(problem.hamiltonian, state, pool)
        best = int(np.argmax(gradients))
        largest = float(gradients[best])

        if largest < gradient_tolerance:
            converged = True
            break

        chosen.append(best)
        parameters = np.concatenate([parameters, [0.0]])
        result = minimize(
            energy_of,
            parameters,
            args=(chosen,),
            method="BFGS",
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

    final = energy_of(parameters, chosen) if len(chosen) else float(
        np.real(reference.expectation_value(problem.hamiltonian))
    )
    return AdaptResult(
        energy=final,
        parameters=parameters,
        operators=chosen,
        steps=steps,
        converged=converged,
        metadata={"pool_size": len(pool), "pool_kind": pool_kind},
    )
