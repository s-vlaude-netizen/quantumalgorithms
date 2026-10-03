"""Nested-dissection fill of the IonQ + Ansys pipeline against METIS (Result 95).

The comparison is only as good as the fill count, so the symbolic Cholesky is
checked against numeric factorisations, and the separator against its
definition.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("pymetis")

from experiments.exp039_ionq_ansys_partitioning import shell_mesh
from experiments.exp040_nested_dissection_fill import (
    _cut,
    fm_refine,
    metis_nd,
    ordering_from_bisection,
    symbolic_cholesky,
    vertex_separator,
)


def random_graph(n, density, seed):
    g = sp.random(n, n, density=density, random_state=seed)
    g = ((g + g.T) > 0).astype(np.int64)
    g.setdiag(0)
    g.eliminate_zeros()
    return g.tocsr()


def numeric_nnz(graph, order, seed):
    rng = np.random.default_rng(seed)
    n = graph.shape[0]
    a = graph.toarray() * rng.uniform(0.1, 1.0, (n, n))
    a = (a + a.T) / 2
    a += np.diag(np.abs(a).sum(axis=1) + 1)
    factor = np.linalg.cholesky(a[np.ix_(order, order)])
    return int((np.abs(factor) > 1e-14).sum())


@pytest.mark.parametrize("seed", range(4))
def test_symbolic_fill_matches_a_numeric_factorisation(seed):
    graph = random_graph(80, 0.06, seed)
    order = np.random.default_rng(seed).permutation(80)
    assert symbolic_cholesky(graph, order)[0] == numeric_nnz(graph, order, seed)


def test_symbolic_fill_on_a_mesh_with_metis_order():
    graph = shell_mesh(nx=20, ny=14).tocsr().astype(np.int64)
    order = metis_nd(graph)
    assert sorted(order) == list(range(graph.shape[0]))
    assert symbolic_cholesky(graph, order)[0] == numeric_nnz(graph, order, 0)


def test_metis_order_is_new_to_old():
    """pymetis returns (perm, iperm); the elimination order is perm, and it must
    beat both the natural order and its own inverse on a grid."""
    graph = shell_mesh(nx=40, ny=40).tocsr().astype(np.int64)
    good = symbolic_cholesky(graph, metis_nd(graph))[0]
    assert good < symbolic_cholesky(graph, np.arange(graph.shape[0]))[0]
    assert good < symbolic_cholesky(graph, np.argsort(metis_nd(graph)))[0]


@pytest.mark.parametrize("seed", range(6))
def test_separator_disconnects_the_halves_and_is_a_minimum_cover(seed):
    """Koenig: the separator must cover every cut edge, and no smaller set may."""
    import itertools

    n = 14
    graph = random_graph(n, 0.3, seed)
    x = np.random.default_rng(seed).integers(0, 2, n)
    separator = vertex_separator(graph, x)
    keep = np.ones(n, bool)
    keep[separator] = False
    rows = np.repeat(np.arange(n), np.diff(graph.indptr))
    crossing = keep[rows] & keep[graph.indices] & (x[rows] != x[graph.indices])
    assert not crossing.any()

    edges = {(min(a, b), max(a, b)) for a, b in zip(rows, graph.indices) if x[a] != x[b]}
    vertices = sorted({v for e in edges for v in e})
    smallest = next(k for k in range(len(vertices) + 1)
                    for cover in itertools.combinations(vertices, k)
                    if all(a in cover or b in cover for a, b in edges))
    assert len(separator) == smallest


def test_ordering_from_bisection_is_a_permutation_with_separator_last():
    graph = shell_mesh(nx=30, ny=20).tocsr().astype(np.int64)
    x = (np.arange(graph.shape[0]) % 2).astype(np.int64)
    order, size = ordering_from_bisection(graph, x)
    assert sorted(order) == list(range(graph.shape[0]))
    assert set(order[-size:]) == set(vertex_separator(graph, x))


def test_fm_refine_never_worsens_a_balanced_cut():
    graph = shell_mesh(nx=40, ny=30).tocsr().astype(np.int64)
    weights = np.ones(graph.shape[0], dtype=np.int64)
    x = np.random.default_rng(0).permutation(graph.shape[0]) % 2
    refined = fm_refine(graph, weights, x)
    assert _cut(graph, refined) < _cut(graph, x)
    assert abs(refined.mean() - 0.5) <= 0.05
