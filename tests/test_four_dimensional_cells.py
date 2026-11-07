"""Four-dimensional cells (exp045): exact values against the numerical
Voronoi machinery, and the affine reduction on known cases."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments.exp045_four_dimensional_cells import (
    affine_minimum,
    lll,
    special_cells,
    voronoi_cell,
)


@pytest.fixture(scope="module")
def exact():
    return {name: float(value) for name, value in special_cells().items()}


def a4_star_basis():
    roots = np.eye(5)[:4] - np.eye(5)[1:]
    q = np.linalg.qr(np.vstack([np.ones(5), np.eye(5)[:4]]).T)[0][:, 1:]
    return np.linalg.inv(roots @ q).T


def test_exact_values(exact):
    assert exact["24-cell (Vor D4)"] == pytest.approx(2 ** 2.75, rel=1e-14)
    assert exact["permutohedron (Vor A4*)"] == pytest.approx(
        20 * (16 + 3 * 6 ** 0.5) / 5 ** (21 / 8), rel=1e-14)
    assert exact["cographic K33 zonotope"] == pytest.approx(
        (32 * 3 ** 0.5 + 4 * 2 ** 0.5) / 9, rel=1e-14)
    assert exact["24-cell (Vor D4)"] < exact["cographic K33 zonotope"] < exact["permutohedron (Vor A4*)"]


@pytest.mark.parametrize("name, basis", [
    ("24-cell (Vor D4)", np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 1, 1]], float)),
    ("permutohedron (Vor A4*)", a4_star_basis()),
    ("tesseract (Z^4)", np.eye(4)),
])
def test_voronoi_cells_match_the_exact_values(exact, name, basis):
    quotient, _, facets = voronoi_cell(basis)
    assert quotient == pytest.approx(exact[name], rel=1e-10)
    assert len(facets) == {"24-cell (Vor D4)": 24, "permutohedron (Vor A4*)": 30,
                           "tesseract (Z^4)": 8}[name]


def test_cographic_zonotope_by_its_cycle_space_projection(exact):
    from experiments.exp044_hales_song_certificate import zonotope_measures

    edges = [(i, j) for i in range(3) for j in range(3)]
    incidence = np.zeros((6, 9))
    for k, (i, j) in enumerate(edges):
        incidence[i, k], incidence[3 + j, k] = 1, -1
    cut = incidence.T @ np.linalg.pinv(incidence @ incidence.T) @ incidence
    cycle = np.eye(9) - cut
    basis = np.linalg.svd(cycle)[0][:, :4]          # orthonormal basis of the cycle space
    generators = [basis.T @ cycle[:, k] for k in range(9)]
    quotient, _ = zonotope_measures(generators)
    assert quotient == pytest.approx(exact["cographic K33 zonotope"], rel=1e-12)
    minors = {round(abs(np.linalg.det(np.array([cycle[:, k] for k in s]) @ basis)), 9)
              for s in itertools.combinations(range(9), 4)}
    assert len(minors - {0.0}) == 1                 # unimodular: every basis has one volume


def test_lll_keeps_the_lattice_and_shortens_the_basis():
    rng = np.random.default_rng(0)
    basis = np.eye(4) + np.triu(rng.integers(-5, 6, size=(4, 4)), 1)
    reduced = lll(basis)
    assert abs(np.linalg.det(reduced)) == pytest.approx(abs(np.linalg.det(basis)))
    change = np.linalg.solve(basis.T, reduced.T)       # integer change of basis
    assert np.allclose(change, np.round(change), atol=1e-9)
    assert max(np.linalg.norm(reduced, axis=1)) <= max(np.linalg.norm(basis, axis=1))


def test_affine_reduction(exact):
    quotient, volume, facets = voronoi_cell(np.diag([2.0, 0.5, 1.0, 1.0]))
    assert quotient > 8.5                           # a stretched box ...
    assert affine_minimum(facets, volume) == pytest.approx(8.0, rel=1e-8)   # ... becomes the cube
    quotient, volume, facets = voronoi_cell(
        np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 1, 1]], float))
    assert affine_minimum(facets, volume) == pytest.approx(exact["24-cell (Vor D4)"], rel=1e-9)
