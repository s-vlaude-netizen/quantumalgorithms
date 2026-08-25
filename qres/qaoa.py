"""QAOA driver for combinatorial optimisation.

Two things make QAOA a different measurement problem from VQE, and both are
easy to get wrong by porting the VQE machinery unchanged:

1. **Measurement is free.**  The cost Hamiltonian is diagonal, so every term
   commutes with every other and a *single* computational-basis circuit
   measures the whole objective.  All the Pauli-grouping and Neyman-allocation
   machinery that dominates VQE cost is irrelevant here -- there is one group.

2. **The expectation value is not the answer.**  QAOA is used as a *sampler*:
   what a user gets is the best bitstring seen across all shots, not
   ``<H>``.  A parameter setting with a worse expectation value but a heavier
   tail can be the better optimiser.  Scoring on ``<H>`` alone -- which is what
   the optimiser minimises -- systematically misranks configurations, so both
   are reported here and ``best_sampled`` is the headline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from .estimator import ShotEstimator, _counts_to_arrays
from .noise import NoiseEnvironment, describe_environment, make_environment
from .optimizers import OPTIMIZERS, BudgetExceeded, EnergyOracle, OptimizeResult
from .resources import ResourceLedger


def qaoa_circuit(
    hamiltonian: SparsePauliOp,
    reps: int = 1,
    *,
    initial_state: QuantumCircuit | None = None,
    mixer: str = "x",
) -> QuantumCircuit:
    """Standard QAOA ansatz for a diagonal cost Hamiltonian.

    ``exp(-i gamma J Z_i Z_j)`` is ``rzz(2 gamma J)`` and ``exp(-i beta X)`` is
    ``rx(2 beta)``, following Qiskit's ``r*(theta) = exp(-i theta/2 P)``
    convention.

    Parameters are laid out as ``[gamma_0..gamma_{p-1}, beta_0..beta_{p-1}]`` in
    a **single** ParameterVector, and that is load-bearing.  Qiskit orders
    ``circuit.parameters`` by name, so two vectors named ``g`` and ``b`` come
    back as betas-then-gammas and a positional ``assign_parameters`` silently
    swaps the two families.  The optimiser still converges to *something*, which
    is what makes it hard to notice -- but the adiabatic initial ramp is applied
    backwards and parameter transfer between instances is meaningless.  A single
    vector sorts numerically and keeps the intended order.
    """
    n = hamiltonian.num_qubits
    theta = ParameterVector("t", 2 * reps)
    gammas, betas = theta[:reps], theta[reps:]

    qc = QuantumCircuit(n, name=f"qaoa{reps}")
    if initial_state is not None:
        qc.compose(initial_state, inplace=True)
    else:
        qc.h(range(n))

    singles, pairs = _ising_terms(hamiltonian)
    for layer in range(reps):
        for q, h in singles:
            qc.rz(2 * gammas[layer] * h, q)
        for (i, j), J in pairs:
            qc.rzz(2 * gammas[layer] * J, i, j)
        if mixer == "x":
            for q in range(n):
                qc.rx(2 * betas[layer], q)
        else:
            raise ValueError(f"unknown mixer {mixer!r}")
    return qc


def _ising_terms(hamiltonian: SparsePauliOp):
    """Split a diagonal Hamiltonian into (qubit, h) and ((i,j), J) lists."""
    singles, pairs = [], []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            raise ValueError("QAOA cost Hamiltonian must be diagonal")
        qubits = np.flatnonzero(pauli.z).tolist()
        if len(qubits) == 1:
            singles.append((qubits[0], float(coeff.real)))
        elif len(qubits) == 2:
            pairs.append(((qubits[0], qubits[1]), float(coeff.real)))
        elif len(qubits) > 2:
            raise ValueError("higher-order terms not supported yet")
    return singles, pairs


@dataclass
class QAOARun:
    problem_name: str
    config: dict[str, Any]
    #: <H> + offset at the optimiser's final parameters
    expectation: float
    #: cost of the best bitstring seen across every shot of the whole run
    best_sampled: float
    optimal_value: float
    approximation_ratio: float
    best_sampled_ratio: float
    #: fraction of final-parameter shots that landed on an optimal solution
    optimal_probability: float
    ledger: dict[str, Any]
    history: list[float] = field(default_factory=list)
    shot_history: list[int] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    wall_seconds: float = 0.0
    parameters: list[float] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem_name,
            "config": self.config,
            "expectation": self.expectation,
            "best_sampled": self.best_sampled,
            "optimal_value": self.optimal_value,
            "approximation_ratio": self.approximation_ratio,
            "best_sampled_ratio": self.best_sampled_ratio,
            "optimal_probability": self.optimal_probability,
            "ledger": self.ledger,
            "environment": self.environment,
            "wall_seconds": self.wall_seconds,
            "message": self.message,
            "history": self.history,
            "shot_history": self.shot_history,
        }

    def line(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.config.get('label', ''):30s} "
            f"AR={self.approximation_ratio:.4f} best={self.best_sampled_ratio:.4f} "
            f"p_opt={self.optimal_probability:.4f} "
            f"shots={self.ledger['shots']:>9,} wall={self.wall_seconds:5.1f}s"
        )


class SamplingTracker:
    """Watches every shot the estimator takes and keeps the best bitstring.

    QAOA's real output is the best sample, not the expectation value, and the
    samples are already being drawn -- tracking them costs nothing and is the
    only way to score the algorithm the way it is actually used.
    """

    def __init__(self, problem):
        self.problem = problem
        self.best_cost = np.inf
        self.best_bitstring: str | None = None
        self.shots_seen = 0

    def observe(self, counts: dict) -> None:
        for bitstring, count in counts.items():
            clean = bitstring.replace(" ", "")[-self.problem.num_qubits:]
            self.shots_seen += count
            cost = self.problem.cost_of_bitstring(clean)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_bitstring = clean


def run_qaoa(
    problem,
    *,
    reps: int = 1,
    environment: str | NoiseEnvironment = "ideal",
    optimizer: str = "cobyla",
    shots: int = 2048,
    maxiter: int = 200,
    shot_budget: int | None = None,
    seed: int = 0,
    initial_params: Sequence[float] | None = None,
    label: str | None = None,
    optimizer_kwargs: dict | None = None,
) -> QAOARun:
    """Run one QAOA configuration end to end.

    ``initial_params`` lets a caller supply pre-trained angles (parameter
    transfer): pass them with ``maxiter=0`` to skip instance-specific
    optimisation entirely and pay only for the final sampling.
    """
    t_start = time.perf_counter()
    env = make_environment(environment, seed=seed) if isinstance(environment, str) else environment

    circuit = qaoa_circuit(problem.hamiltonian, reps=reps)
    ledger = ResourceLedger()
    estimator = ShotEstimator(
        problem.hamiltonian,
        circuit,
        env,
        grouping="commuting",  # diagonal cost -> exactly one group
        allocation="uniform",
        ledger=ledger,
    )
    tracker = SamplingTracker(problem)
    _attach_tracker(estimator, tracker)

    oracle = EnergyOracle(estimator, default_shots=shots, budget=shot_budget)
    x0 = np.asarray(initial_params, dtype=float) if initial_params is not None else _default_angles(reps)

    kwargs = dict(optimizer_kwargs or {})
    if optimizer in ("cobyla", "powell", "nelder-mead", "spsa"):
        kwargs.setdefault("shots", shots)
    kwargs.setdefault("maxiter", maxiter)
    if optimizer in ("spsa", "icans", "aspsa"):
        kwargs.setdefault("seed", seed)

    if maxiter <= 0:
        result = OptimizeResult(x=x0, fun=float(oracle(x0, shots)), nit=0, message="transferred")
    else:
        try:
            result = OPTIMIZERS[optimizer](oracle, x0, **kwargs)
        except BudgetExceeded as exc:
            x_best = oracle.best_params if oracle.best_params is not None else x0
            result = OptimizeResult(
                x=np.asarray(x_best),
                fun=float(oracle.best_value),
                nit=oracle.calls,
                history=[v for _, v in oracle.trace],
                shot_history=[s for s, _ in oracle.trace],
                message=f"budget exhausted: {exc}",
            )

    final = estimator.estimate(result.x, shots)
    expectation = final.value + problem.offset
    optimum = problem.optimal_value + problem.offset
    p_opt = _optimal_probability(estimator, result.x, problem, shots, env)

    config = {
        "reps": reps,
        "optimizer": optimizer,
        "shots": shots,
        "maxiter": maxiter,
        "seed": seed,
        "shot_budget": shot_budget,
        "transferred": initial_params is not None,
        "label": label or f"qaoa{reps}/{optimizer}",
    }
    return QAOARun(
        problem_name=problem.name,
        config=config,
        expectation=float(expectation),
        best_sampled=float(tracker.best_cost),
        optimal_value=float(optimum),
        approximation_ratio=float(problem.approximation_ratio(expectation)),
        best_sampled_ratio=float(problem.approximation_ratio(tracker.best_cost)),
        optimal_probability=float(p_opt),
        ledger=ledger.to_dict(),
        history=[float(h) for h in result.history],
        shot_history=[int(s) for s in result.shot_history],
        environment=describe_environment(env),
        wall_seconds=time.perf_counter() - t_start,
        parameters=[float(v) for v in result.x],
        message=result.message,
    )


def _attach_tracker(estimator: ShotEstimator, tracker: SamplingTracker) -> None:
    """Route every counts dict the estimator sees through the tracker."""
    original = estimator._group_statistics

    def wrapped(group_index, counts):
        tracker.observe(counts)
        return original(group_index, counts)

    estimator._group_statistics = wrapped


def _default_angles(reps: int) -> np.ndarray:
    """A linear ramp, which is the discretised adiabatic schedule.

    Far better than random angles: gamma should grow and beta shrink across
    layers, and starting from that shape rather than from noise is most of what
    "warm starting" buys at small depth.
    """
    gammas = np.linspace(0.2, 0.8, reps)
    betas = np.linspace(0.8, 0.2, reps)
    return np.concatenate([gammas, betas])


def _optimal_probability(estimator, params, problem, shots, env) -> float:
    """Fraction of shots landing on an optimal assignment at ``params``."""
    circuits = estimator._isa
    if not circuits:
        return 0.0
    bound = circuits[0].assign_parameters(np.asarray(params, dtype=float))
    counts = env.run([bound], shots=shots)
    counts = counts[0] if isinstance(counts, list) else counts
    optimal = set(problem.optimal_bitstrings)
    n = problem.num_qubits
    hits = sum(c for b, c in counts.items() if b.replace(" ", "")[-n:] in optimal)
    total = sum(counts.values())
    return hits / total if total else 0.0
