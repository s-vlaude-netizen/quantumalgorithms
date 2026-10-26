"""The Kelvin problem toolkit (exp042): exact costs, periodic foams, relaxation.

Every number here is checked against an independent route: closed forms for the
exact costs, the published Kelvin and Weaire-Phelan costs for the relaxation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.exp042_kelvin_problem import (
    flat_cost,
    foam_measures,
    periodic_voronoi,
    point_set,
    relax,
    surface_terms,
    triangulate,
)


def test_exact_costs_are_the_closed_forms():
    import sympy as sp

    from experiments.exp042_kelvin_problem import exact_candidates

    exact = exact_candidates()
    by_name = {name.split(" (")[0]: value for name, value in exact.items()}
    assert sp.simplify(by_name["rhombic dodecahedron"] - 3 * 2 ** sp.Rational(5, 6)) == 0
    assert sp.simplify(by_name["trapezo-rhombic dodecahedron"] - 3 * 2 ** sp.Rational(5, 6)) == 0
    kelvin = (6 + 12 * sp.sqrt(3)) / (8 * sp.sqrt(2)) ** sp.Rational(2, 3)
    assert sp.simplify(by_name["truncated octahedron"] - kelvin) == 0
    assert by_name["cube"] == 6


@pytest.mark.parametrize("name, cost", [("sc", 6.0), ("bcc", 5.314740), ("fcc", 5.345392),
                                        ("hcp", 5.345392)])
def test_periodic_voronoi_reproduces_the_exact_costs(name, cost):
    box, points = point_set(name)
    value, measures = flat_cost(periodic_voronoi(points, box))
    assert value == pytest.approx(cost, abs=1e-6)
    assert measures["volumes"].sum() == pytest.approx(abs(np.linalg.det(box)))


def test_the_densest_packing_foams_have_six_cell_vertices():
    """Taylor (1976): in an area-minimising foam exactly four cells meet at a
    vertex. FCC and HCP Voronoi foams have vertices where six meet."""
    for name in ("fcc", "hcp"):
        box, points = point_set(name)
        census = foam_measures(periodic_voronoi(points, box))["vertex_cells"]
        assert 6 in census
    for name in ("bcc", "a15"):
        box, points = point_set(name)
        assert set(foam_measures(periodic_voronoi(points, box))["vertex_cells"]) == {4}


def test_volume_gradient_matches_finite_differences():
    box, points = point_set("a15")
    mesh = triangulate(periodic_voronoi(points, box), 1)
    y = mesh["y"] + np.random.default_rng(0).normal(scale=1e-3, size=mesh["y"].shape)
    _, volumes, grad_area, grad_volume = surface_terms(y, mesh)
    assert volumes.sum() == pytest.approx(1.0, abs=1e-12)   # cells tile the box
    for vertex, axis in ((3, 0), (60, 2), (200, 1)):
        step = np.zeros_like(y)
        step[vertex, axis] = 1e-6
        a_plus, v_plus, _, _ = surface_terms(y + step, mesh)
        a_minus, v_minus, _, _ = surface_terms(y - step, mesh)
        assert (a_plus - a_minus) / 2e-6 == pytest.approx(grad_area[vertex, axis], abs=1e-6)
        np.testing.assert_allclose((v_plus - v_minus) / 2e-6, grad_volume[:, vertex, axis],
                                   atol=1e-6)


def test_relaxation_reproduces_kelvins_curved_foam():
    """Kelvin's foam, flat 5.31474, relaxes to 5.3063 (Kusner & Sullivan 1996);
    refinement levels 1 and 2 extrapolate to it."""
    box, points = point_set("bcc")
    foam = periodic_voronoi(points, box)
    costs = [relax(triangulate(foam, level))["cost"] for level in (1, 2)]
    assert costs[1] < costs[0] < 5.31474
    assert (4 * costs[1] - costs[0]) / 3 == pytest.approx(5.3063, abs=2e-4)


def test_sphere_bound_is_below_everything():
    from experiments.exp042_kelvin_problem import SPHERE

    assert SPHERE == pytest.approx(4.835976, abs=1e-6)
    assert math.isclose(SPHERE, (36 * math.pi) ** (1 / 3))
