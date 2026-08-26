"""Classical baselines for the Ising/MaxCut problems.

The chemistry side of this package has a classical anchor (RESEARCH_LOG
Result 42) and the optimisation side did not, which makes every QAOA number here
as unmoored as the VQE ones were.  The comparison is a different shape, though,
and that is why it is worth doing separately: molecular ground states have
CCSD(T), which is polynomial and very accurate, whereas MaxCut is NP-hard.  The
classical competitor is an approximation algorithm with a *proven ratio*, so the
question becomes whether QAOA beats a guarantee rather than whether it beats an
exact answer.

Three baselines, cheapest first:

``greedy``      one pass, place each vertex on the side that cuts more
``local``       greedy plus best-improvement flips to a local optimum
``goemans``     the Goemans-Williamson SDP relaxation with hyperplane rounding,
                which has a proven 0.878 approximation ratio for MaxCut

The SDP is solved in the Burer-Monteiro form -- optimise over unit vectors in
``R^k`` rather than over the full PSD matrix -- which needs only scipy and is
exact for ``k >= sqrt(2n)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OptimizationResult:
    method: str
    cut_value: float
    seconds: float
    approximation_ratio: float
    bitstring: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _weight_matrix(problem) -> np.ndarray:
    """Symmetric adjacency with weights, from the problem's edge list."""
    n = problem.num_variables
    weights = np.zeros((n, n))
    for i, j, w in problem.metadata["edges"]:
        weights[i, j] += w
        weights[j, i] += w
    return weights


def _cut_value(weights: np.ndarray, assignment: np.ndarray) -> float:
    """Total weight of edges crossing the partition."""
    signs = 1 - 2 * assignment  # 0 -> +1, 1 -> -1
    return float((weights * (1 - np.outer(signs, signs))).sum() / 4)


def _greedy_assignment(weights: np.ndarray, n: int) -> np.ndarray:
    """One pass in a fixed order, each vertex to whichever side cuts more."""
    assignment = np.zeros(n, dtype=int)
    for v in range(1, n):
        placed = assignment[:v]
        gain_zero = float(weights[v, :v][placed == 1].sum())
        gain_one = float(weights[v, :v][placed == 0].sum())
        assignment[v] = 1 if gain_one > gain_zero else 0
    return assignment


def greedy_maxcut(problem) -> OptimizationResult:
    t0 = time.perf_counter()
    weights = _weight_matrix(problem)
    assignment = _greedy_assignment(weights, problem.num_variables)
    value = _cut_value(weights, assignment)
    return _record(problem, "greedy", value, time.perf_counter() - t0, assignment)


def _hill_climb(weights: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    """Best-improvement single-vertex flips until no flip helps."""
    while True:
        signs = 1 - 2 * assignment
        # flipping v changes the cut by the weight it currently cuts minus the
        # weight it currently does not
        deltas = signs * (weights @ signs)
        v = int(np.argmax(deltas))
        if deltas[v] <= 1e-12:
            return assignment
        assignment[v] ^= 1


def local_search_maxcut(problem, restarts: int = 10, seed: int = 0) -> OptimizationResult:
    """Hill-climbing from the greedy solution, then from random restarts.

    Restart 0 used to be the all-zeros assignment, whose cut is zero -- the
    worst possible starting point, and one that hill-climbing does not recover
    from: on a 14-vertex Erdos-Renyi instance it settled at 28 where plain
    greedy alone reached 31.  Seeding from greedy costs one extra pass and makes
    the result at least as good as greedy by construction, which is what a
    baseline this project measures QAOA against has to be.
    """
    t0 = time.perf_counter()
    weights = _weight_matrix(problem)
    n = problem.num_variables
    rng = np.random.default_rng(seed)
    best_value, best_assignment = -np.inf, None

    for attempt in range(max(1, restarts)):
        start = _greedy_assignment(weights, n) if attempt == 0 else rng.integers(0, 2, n)
        assignment = _hill_climb(weights, start)
        value = _cut_value(weights, assignment)
        if value > best_value:
            best_value, best_assignment = value, assignment.copy()

    return _record(problem, "local search", best_value, time.perf_counter() - t0, best_assignment)


def goemans_williamson(problem, rounds: int = 100, seed: int = 0) -> OptimizationResult:
    """The 0.878-approximation SDP relaxation, Burer-Monteiro form.

    Maximise ``sum_ij w_ij (1 - v_i . v_j) / 4`` over unit vectors ``v_i`` in
    ``R^k``.  For ``k >= sqrt(2n)`` this is equivalent to the full SDP, so the
    proven ratio carries over.  Rounded by random hyperplanes, best of
    ``rounds``.
    """
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    weights = _weight_matrix(problem)
    n = problem.num_variables
    k = max(2, int(np.ceil(np.sqrt(2 * n))))
    rng = np.random.default_rng(seed)

    def objective(flat):
        vectors = flat.reshape(n, k)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit = vectors / np.maximum(norms, 1e-12)
        # negated because we minimise
        return float(-(weights * (1 - unit @ unit.T)).sum() / 4)

    start = rng.normal(size=n * k)
    result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 2000})
    vectors = result.x.reshape(n, k)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    best_value, best_assignment = -np.inf, None
    for _ in range(rounds):
        plane = rng.normal(size=k)
        assignment = (vectors @ plane < 0).astype(int)
        value = _cut_value(weights, assignment)
        if value > best_value:
            best_value, best_assignment = value, assignment

    return _record(
        problem, "Goemans-Williamson", best_value, time.perf_counter() - t0, best_assignment,
        sdp_bound=-float(result.fun), rank=k,
    )


def _record(problem, method, value, seconds, assignment, **metadata) -> OptimizationResult:
    optimum = problem.metadata.get("max_cut")
    ratio = value / optimum if optimum else float("nan")
    bits = "".join(str(int(b)) for b in reversed(assignment)) if assignment is not None else ""
    return OptimizationResult(
        method=method,
        cut_value=float(value),
        seconds=float(seconds),
        approximation_ratio=float(ratio),
        bitstring=bits,
        metadata=metadata,
    )


def classical_maxcut_baselines(problem) -> list[OptimizationResult]:
    """All three, plus the exact optimum the problem already carries."""
    results = [
        greedy_maxcut(problem),
        local_search_maxcut(problem),
        goemans_williamson(problem),
    ]
    optimum = problem.metadata.get("max_cut")
    if optimum is not None:
        results.append(
            OptimizationResult(
                method="exact (brute force)",
                cut_value=float(optimum),
                seconds=float("nan"),  # already computed when the problem was built
                approximation_ratio=1.0,
            )
        )
    return results
