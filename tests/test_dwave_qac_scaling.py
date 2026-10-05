"""Classical annealers on D-Wave's QAC instance class (exp041).

The graph is checked against the paper's description, and every kernel
against brute force on instances small enough to enumerate.
"""

from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("numba")

from experiments.exp041_dwave_qac_scaling import (
    SIDON28,
    UNIT,
    betas_geometric,
    energy,
    instance,
    integer_weights,
    kernels,
    logical_graph,
    neighbour_arrays,
    repetitions,
)


@pytest.mark.parametrize("side", [5, 8, 15])
def test_the_logical_graph_has_six_qubits_per_cell(side):
    n, pairs = logical_graph(side)
    assert n == 6 * side ** 2
    graph = nx.Graph([tuple(p) for p in pairs])
    assert graph.number_of_nodes() == n and nx.is_connected(graph)


def test_the_logical_graph_matches_the_papers_description():
    """Bulk degree 5, native 5-loops, non-planar (arXiv:2401.07184, Fig. 1)."""
    from experiments.exp041_dwave_qac_scaling import full_graph

    nodes, edges = full_graph()
    graph = nx.Graph([(i, j) for i, j, _ in edges])
    bulk = [k for k, (t, y, x, u) in enumerate(nodes) if 2 <= y <= 12 and 2 <= x <= 12]
    assert {graph.degree(k) for k in bulk} == {5}
    small = graph.subgraph([k for k, (t, y, x, u) in enumerate(nodes) if y < 4 and x < 4])
    assert min(len(c) for c in nx.minimum_cycle_basis(small)) == 5
    assert not nx.check_planarity(small)[0]
    assert {m for _, _, m in edges} == {2, 3}


def test_couplings_are_sidon_28():
    _, _, _, couplings = instance(6, 0)
    assert set(np.round(np.abs(couplings), 12)) <= set(np.round(SIDON28, 12))
    assert (couplings > 0).any() and (couplings < 0).any()


def tiny(seed, n=12):
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.35])
    couplings = rng.choice(SIDON28, len(pairs)) * rng.choice([-1.0, 1.0], len(pairs))
    return n, pairs[:, 0], pairs[:, 1], couplings


def brute_force(n, i, j, couplings):
    best = math.inf
    for bits in itertools.product((-1.0, 1.0), repeat=n):
        best = min(best, energy(np.array(bits), i, j, couplings))
    return best


def test_integer_energy_matches_the_float_energy():
    integer_energy = kernels()[0]
    n, i, j, couplings = instance(7, 3)
    pointer, index, weight = neighbour_arrays(n, i, j, couplings)
    spins = np.random.default_rng(1).choice([-1, 1], n).astype(np.int64)
    assert integer_energy(spins, pointer, index, integer_weights(weight)) / UNIT == pytest.approx(
        energy(spins.astype(float), i, j, couplings))


@pytest.mark.parametrize("seed", range(3))
def test_every_solver_finds_the_ground_state_of_a_tiny_glass(seed):
    _, _, anneal, quantum_anneal, tempering = kernels()
    n, i, j, couplings = tiny(seed)
    pointer, index, weight = neighbour_arrays(n, i, j, couplings)
    q = integer_weights(weight)
    exact = round(brute_force(n, i, j, couplings) * UNIT)
    assert anneal(pointer, index, q, betas_geometric(500, 0.5, 8.0), 20, seed).min() == exact
    best, single = quantum_anneal(pointer, index, q, 8, 8.0, np.linspace(3, 1e-3, 300), 10, seed)
    assert best.min() == exact and (best <= single).all()
    hits, best = tempering(pointer, index, q, betas_geometric(8), 2, 300, exact, 5, seed)
    assert (best == exact).all() and (hits > 0).all()


def test_tempering_reports_the_first_hitting_sweep():
    _, _, _, _, tempering = kernels()
    n, i, j, couplings = tiny(7)
    pointer, index, weight = neighbour_arrays(n, i, j, couplings)
    q = integer_weights(weight)
    exact = round(brute_force(n, i, j, couplings) * UNIT)
    unreachable, _ = tempering(pointer, index, q, betas_geometric(8), 2, 50, exact - 1, 3, 0)
    assert (unreachable == -1).all()
    hits, _ = tempering(pointer, index, q, betas_geometric(8), 2, 200, exact, 3, 0)
    assert ((hits >= 1) & (hits <= 200)).all()


def test_repetitions_follow_the_tts_formula():
    assert repetitions(0.99) == 1.0
    assert repetitions(0.0) == math.inf
    assert repetitions(0.5) == pytest.approx(math.log(0.01) / math.log(0.5))
    assert repetitions(0.995) == 1.0
