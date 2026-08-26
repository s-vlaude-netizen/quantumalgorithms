"""Correctness of the classical MaxCut baselines.

These are the yardstick every QAOA number in this project is measured against
(RESEARCH_LOG Result 50), so a bug here does not produce a wrong baseline -- it
produces a *flattering* one, and the quantum side silently looks better than it
is.  Result 50 was nearly reported with the optimality rates divided by the
wrong denominator for exactly that reason.

The checks that matter are the ones tying every path back to a single
independently-computed cut value:

* the cut of an assignment, computed from the weight matrix, must equal the
  cost the ``IsingProblem`` assigns to the same bitstring
* the graph generator must produce genuinely ``d``-regular simple graphs, and
  must succeed rather than silently returning fewer instances
* no heuristic may ever report a cut above the brute-force optimum
"""

from __future__ import annotations

import numpy as np
import pytest

from qres.classical_optimization import (
    _cut_value,
    _weight_matrix,
    classical_maxcut_baselines,
    goemans_williamson,
    greedy_maxcut,
    local_search_maxcut,
)
from qres.problems.optimization import (
    _random_regular_edges,
    erdos_renyi_maxcut,
    maxcut,
    random_regular_maxcut,
)


def test_cut_value_matches_the_hamiltonian():
    """The two independent routes to a cut value must agree exactly.

    ``_cut_value`` works from the weight matrix; ``cost_of_bitstring`` evaluates
    the Ising Hamiltonian plus its offset.  A sign or offset slip in either
    shows up as a constant shift, which is precisely the bug that once made
    QAOA look worse than random guessing.
    """
    problem = random_regular_maxcut(10, 3, seed=0)
    weights = _weight_matrix(problem)
    rng = np.random.default_rng(0)

    for _ in range(32):
        assignment = rng.integers(0, 2, problem.num_variables)
        bits = "".join(str(int(b)) for b in reversed(assignment))
        assert _cut_value(weights, assignment) == pytest.approx(
            -problem.cost_of_bitstring(bits), abs=1e-9
        )


def test_optimal_bitstring_achieves_the_recorded_optimum():
    for seed in range(4):
        problem = random_regular_maxcut(12, 3, seed=seed)
        weights = _weight_matrix(problem)
        bits = problem.optimal_bitstrings[0]
        assignment = np.array([int(c) for c in reversed(bits)])
        assert _cut_value(weights, assignment) == pytest.approx(
            problem.metadata["max_cut"], abs=1e-9
        )


@pytest.mark.parametrize("n,degree", [(10, 3), (12, 5), (14, 7), (20, 9)])
def test_generator_produces_simple_regular_graphs(n, degree):
    """Rejection sampling used to fail most seeds at degree >= 5.

    It raised, callers skipped the failures, and an optimality study divided by
    the number of *attempted* instances anyway -- reading 3 successes out of 10
    attempts back as "optimal on 30%".  The generator must now succeed, and what
    it returns must actually be regular and simple.
    """
    for seed in range(10):
        rng = np.random.default_rng(seed)
        edges = _random_regular_edges(n, degree, rng)

        assert len(edges) == n * degree // 2
        assert all(a != b for a, b in edges), "self-loop"
        canonical = {(min(a, b), max(a, b)) for a, b in edges}
        assert len(canonical) == len(edges), "parallel edge"

        degrees = np.zeros(n, dtype=int)
        for a, b in edges:
            degrees[a] += 1
            degrees[b] += 1
        assert np.all(degrees == degree)


def test_generator_rejects_impossible_degrees():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        _random_regular_edges(5, 3, rng)  # n*d odd
    with pytest.raises(ValueError):
        _random_regular_edges(4, 4, rng)  # degree >= n


@pytest.mark.parametrize(
    "factory",
    [
        lambda s: random_regular_maxcut(12, 3, seed=s),
        lambda s: random_regular_maxcut(12, 5, seed=s),
        lambda s: erdos_renyi_maxcut(12, 0.5, seed=s),
    ],
)
def test_no_heuristic_ever_beats_the_exact_optimum(factory):
    """A heuristic reporting more than the brute-force maximum is a bug.

    This is the one-sided check that catches an inflated baseline; the other
    side (how close they get) is a measurement, not a correctness property.
    """
    for seed in range(5):
        problem = factory(seed)
        optimum = problem.metadata["max_cut"]
        for result in classical_maxcut_baselines(problem):
            assert result.cut_value <= optimum + 1e-9, result.method


def test_local_search_never_worse_than_greedy_start():
    """Restart 0 is the greedy all-zeros start, so flips can only improve it."""
    for seed in range(5):
        problem = erdos_renyi_maxcut(14, 0.5, seed=seed)
        assert (
            local_search_maxcut(problem, restarts=1, seed=seed).cut_value
            >= greedy_maxcut(problem).cut_value - 1e-9
        )


def test_goemans_williamson_relaxation_upper_bounds_the_optimum():
    """The SDP value is a relaxation, so it cannot fall below the true maximum."""
    for seed in range(4):
        problem = random_regular_maxcut(12, 3, seed=seed)
        result = goemans_williamson(problem, seed=seed)
        assert result.metadata["sdp_bound"] >= problem.metadata["max_cut"] - 1e-6
        assert result.metadata["rank"] >= np.sqrt(2 * problem.num_variables) - 1e-9


def test_weighted_graphs_are_handled():
    """Negative weights are legal and must not be silently dropped."""
    edges = [(0, 1, 2.5), (1, 2, -1.0), (2, 3, 0.5), (3, 0, 1.5)]
    problem = maxcut(edges, num_nodes=4)
    for result in classical_maxcut_baselines(problem):
        assert result.cut_value <= problem.metadata["max_cut"] + 1e-9


def test_reported_bitstring_reproduces_the_reported_cut():
    """Every result must carry the assignment that produced its number."""
    problem = random_regular_maxcut(12, 3, seed=1)
    weights = _weight_matrix(problem)
    for result in [greedy_maxcut(problem), local_search_maxcut(problem), goemans_williamson(problem)]:
        assignment = np.array([int(c) for c in reversed(result.bitstring)])
        assert _cut_value(weights, assignment) == pytest.approx(result.cut_value, abs=1e-9)
