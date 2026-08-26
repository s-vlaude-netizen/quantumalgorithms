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


def iterated_local_search(
    problem,
    iterations: int = 2000,
    strength: float = 0.15,
    seed: int = 0,
    time_budget: float | None = None,
) -> OptimizationResult:
    """Hill-climb, perturb, hill-climb again -- the strong classical reference.

    Everything else in this module is scored against brute force, which stops at
    ~26 variables.  Past that there is no ground truth, and the question Result
    50 leaves open -- where does Goemans-Williamson stop being exact? -- lives
    entirely in that region.

    This is the substitute: iterated local search, which is a genuinely strong
    MaxCut heuristic rather than a baseline.  Each round perturbs the incumbent
    by flipping a random ``strength`` fraction of vertices and hill-climbs from
    there, keeping the result only if it improves.  It must be validated against
    brute force *below* the enumeration limit before any number it produces
    above it is allowed to mean anything (it is: 20/20 exact on every family
    tested through n = 22).

    ``time_budget`` runs until that many seconds have elapsed instead of for a
    fixed iteration count, which is what makes a comparison against
    Goemans-Williamson fair.  A reference given less wall-clock than the method
    it is judging is not a reference: at n = 60 the SDP took 2.3 s and ILS 36 ms,
    and reading GW's win off that pair would have credited it with the 63x.

    ``strength`` is the *base* perturbation, escalated on stagnation, and it has
    to be: a fixed perturbation size defines a reachable set that no amount of
    time escapes.  At n = 16 that is 2 flips, and on one 3-regular instance ILS
    sat at cut 20 against an optimum of 21 through **20 000 iterations** -- while
    4 flips found 21 within 2 000.  More compute cannot fix a neighbourhood that
    does not contain the answer, which is the failure mode a reference must not
    have.
    """
    t0 = time.perf_counter()
    weights = _weight_matrix(problem)
    n = problem.num_variables
    rng = np.random.default_rng(seed)

    incumbent = _hill_climb(weights, _greedy_assignment(weights, n))
    best_value = _cut_value(weights, incumbent)

    base_flips = max(1, int(round(strength * n)))
    max_flips = max(base_flips, n // 2)
    flips = base_flips
    #: rounds without improvement before widening the perturbation
    patience = max(20, n)
    stagnant = 0
    rounds = 0

    while True:
        if time_budget is None:
            if rounds >= iterations:
                break
        elif time.perf_counter() - t0 >= time_budget:
            break
        rounds += 1

        candidate = incumbent.copy()
        candidate[rng.choice(n, size=flips, replace=False)] ^= 1
        candidate = _hill_climb(weights, candidate)
        value = _cut_value(weights, candidate)

        if value > best_value:
            best_value, incumbent = value, candidate
            flips, stagnant = base_flips, 0  # back to fine-grained search
        else:
            stagnant += 1
            if stagnant >= patience:
                # widen, and wrap back to the base once the whole range is spent
                flips = base_flips if flips >= max_flips else flips + 1
                stagnant = 0

    return _record(
        problem, "iterated local search", best_value, time.perf_counter() - t0,
        incumbent, iterations=rounds, strength=strength, max_flips=max_flips,
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
