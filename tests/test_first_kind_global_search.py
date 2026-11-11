"""A_n* is not the best lattice of the first kind for n >= 7 (exp048).

The analytic gradient is checked against finite differences, the class sums of the
complete-bipartite and cone families against the generic exact routine, and the verdicts
against two computations that share no code with the Selling-parameter formula: the
determinant-and-cofactor formulas for the zonotope, and a numerically computed Voronoi
cell of the lattice itself.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from experiments import exp047_an_star_local_optimality as e
from experiments import exp048_first_kind_global_search as g


def test_analytic_gradient_matches_finite_differences_and_the_selling_formula():
    n = 5
    problem = g.FirstKindQuotient(n)
    rng = np.random.default_rng(1)
    x = rng.normal(scale=0.3, size=problem.m)
    value, gradient = problem.value_and_gradient(x)
    h = 1e-6
    eye = np.eye(problem.m)
    numeric = np.array([(problem.value_and_gradient(x + h * eye[i])[0]
                         - problem.value_and_gradient(x - h * eye[i])[0]) / (2 * h)
                        for i in range(problem.m)])
    assert np.abs(gradient - numeric).max() < 1e-8
    rho = np.zeros((n + 1, n + 1))
    rho[problem.pairs[:, 0], problem.pairs[:, 1]] = np.exp(x)
    rho = rho + rho.T
    ones = np.ones((n + 1, n + 1)) - np.eye(n + 1)
    assert value - problem.baseline == pytest.approx(
        np.log(e.selling_value(n, rho) / e.selling_value(n, ones)), abs=1e-12)


def test_random_starts_in_four_dimensions_end_at_a_4_star():
    records = g.search(4, starts=6, seed=3)
    assert all(abs(r["relative"]) < 1e-8 for r in records)


def test_the_search_finds_a_lattice_below_a_7_star():
    # most Selling parameters at their bound: the complete bipartite lattice K(4, 4)
    records = g.search(7, starts=12, seed=7, spreads=(2.0,))
    assert min(r["relative"] for r in records) < -6e-4


def _bipartite_rho(n, a):
    return [[Fraction(1) if (i < a) != (j < a) else Fraction(0) for j in range(n + 1)]
            for i in range(n + 1)]


def test_bipartite_class_sums_agree_with_the_generic_exact_routine():
    generic = g.exact_beats_a_n_star(7, _bipartite_rho(7, 4))
    classes = g.bipartite_certificate(4, 4)
    assert generic["smaller"] and classes["smaller"]
    assert classes["log_ratio"] == pytest.approx(generic["log_ratio"], rel=1e-12)
    assert classes["log_ratio"] == pytest.approx(-6.193403e-4, abs=1e-9)


def test_cone_class_sums_agree_with_the_generic_exact_routine():
    m, w = 3, Fraction(1, 2)
    n, size = 2 * m, 2 * m + 1
    rho = [[Fraction(0)] * size for _ in range(size)]
    for i in range(1, size):
        rho[0][i] = rho[i][0] = w
    for i in range(1, m + 1):
        for j in range(m + 1, size):
            rho[i][j] = rho[j][i] = Fraction(1)
    generic = g.exact_beats_a_n_star(n, rho)
    classes = g.cone_certificate(m, w)
    assert classes["smaller"] == generic["smaller"]
    assert classes["log_ratio"] == pytest.approx(generic["log_ratio"], rel=1e-10)


def test_exact_verdicts_by_dimension():
    # balanced complete bipartite lattices beat A_n* from n = 7 on (odd n), not before
    for n in (3, 5):
        m = (n + 1) // 2
        assert not g.bipartite_certificate(m, m)["smaller"]
    for n in (7, 9, 11, 13, 41, 101):
        m = (n + 1) // 2
        assert g.bipartite_certificate(m, m)["smaller"]
    # even n: the cone with a lighter apex, from n = 8 on
    assert not g.cone_certificate(3, Fraction(1, 2))["smaller"]
    for m, w in ((4, "11/20"), (5, "21/40"), (6, "1/2"), (20, "1/2")):
        assert g.cone_certificate(m, Fraction(w))["smaller"]
    # and the unbalanced bipartite lattice needs n >= 18
    assert not g.bipartite_certificate(8, 9)["smaller"]
    assert g.bipartite_certificate(9, 10)["smaller"]


def _zonotope_iso(n, rho):
    from experiments.exp044_hales_song_certificate import zonotope_measures

    f = e.frame(n)
    lap = -rho.copy()
    np.fill_diagonal(lap, 0)
    np.fill_diagonal(lap, -lap.sum(axis=1))
    p = (n / (n + 1)) ** 2 * f.T @ lap @ f
    w, v = np.linalg.eigh(p)
    a_inv = v @ np.diag(1 / np.sqrt(w)) @ v.T
    gens = [rho[i, j] * (a_inv @ (f[i] - f[j]))
            for i, j in itertools.combinations(range(n + 1), 2) if rho[i, j] > 0]
    return zonotope_measures(gens)[0]


def test_k44_lattice_beats_a_7_star_by_the_zonotope_determinant_formulas():
    n = 7
    rho = np.array(_bipartite_rho(n, 4), dtype=float)
    ratio = _zonotope_iso(n, rho) / float(e.permutohedron_iso(n))
    assert np.log(ratio) == pytest.approx(-6.193403e-4, abs=1e-9)


def _voronoi_numeric(n, rho):
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    f = e.frame(n)
    lap = -rho.copy()
    np.fill_diagonal(lap, 0)
    np.fill_diagonal(lap, -lap.sum(axis=1))
    p = (n / (n + 1)) ** 2 * f.T @ lap @ f
    w, v = np.linalg.eigh(p)
    basis = ((v @ np.diag(np.sqrt(w)) @ v.T) @ f[1:].T).T
    vectors = np.array([np.array(c) @ basis for c in itertools.product((-1, 0, 1), repeat=n) if any(c)])
    halfspaces = np.hstack([vectors, -(vectors ** 2).sum(axis=1, keepdims=True) / 2])
    hull = ConvexHull(HalfspaceIntersection(halfspaces, np.zeros(n)).intersections)
    return hull.area / hull.volume ** ((n - 1) / n), hull.volume, basis


def test_boundary_lattices_match_numerically_computed_voronoi_cells():
    # n = 3: the complete bipartite lattice K(2, 2) is FCC, whose cell is the rhombic
    # dodecahedron, 3 * 2^(5/6); n = 5 and 6: degenerate cells with hundreds of vertices
    q, _, _ = _voronoi_numeric(3, np.array(_bipartite_rho(3, 2), dtype=float))
    assert q == pytest.approx(3 * 2 ** (5 / 6), rel=1e-6)
    for n, a in ((5, 3), (6, 3)):
        rho = np.array(_bipartite_rho(n, a), dtype=float)
        numeric, volume, basis = _voronoi_numeric(n, rho)
        assert volume == pytest.approx(np.sqrt(np.linalg.det(basis @ basis.T)), rel=1e-9)
        assert numeric == pytest.approx(_zonotope_iso(n, rho), rel=1e-10)
