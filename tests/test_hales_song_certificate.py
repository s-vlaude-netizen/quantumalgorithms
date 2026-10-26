"""Independent check of the Hales-Song certificate (exp044).

The polynomial is rebuilt from the paper's determinant definitions and must
show every structural fact Appendix A states; the Bernstein machinery is
checked on polynomials with known expansions, and must be able to *fail*.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from experiments.exp044_hales_song_certificate import (
    TRUNCATED_OCTAHEDRON,
    bernstein_on_box,
    certify,
    chart,
    cover,
    covers_unit_cube,
    dense_integer_form,
    phi_psi_d_float,
    shift_to_corner,
    signs,
    stated_facts,
    zone_polynomials,
    zonotope_quotient,
)


@pytest.fixture(scope="module")
def polys():
    return zone_polynomials()


@pytest.fixture(scope="module")
def dense(polys):
    g = polys["f"].eval(polys["symbols"]["34"], 1)
    return dense_integer_form(g, polys["field"])


def test_reconstruction_has_every_stated_property(polys):
    facts = stated_facts(polys)
    assert facts["f_monomials"] == 6083
    assert facts["f_multidegree"] == [6] * 6
    assert facts["f_homogeneous_degree"] == [18]
    assert facts["f_at_one"] == "0"
    assert facts["f_invariant_under_24_relabellings"]
    assert facts["d_at_one"] == 16                      # 4^2 spanning trees of K4
    assert facts["g_monomials"] == 6083
    assert facts["g_multidegree"] == [6] * 5
    assert facts["g_total_degree_range"] == [12, 18]


def test_exact_polynomial_matches_float_definitions(polys):
    rng = np.random.default_rng(5)
    kappa = (1 + 2 * math.sqrt(3)) ** 3 / 128
    for _ in range(3):
        beta = rng.exponential(size=6)
        phi, psi, _ = phi_psi_d_float(beta)
        exact = sum(float(c) * math.prod(x ** k for x, k in zip(beta, m))
                    for m, c in polys["f"].terms())
        assert exact == pytest.approx(phi ** 3 - kappa ** 2 * psi, rel=1e-9, abs=1e-9 * phi ** 3)


def test_dense_integer_form_evaluates_like_sympy(polys, dense):
    import sympy as sp

    a, b, denominator = dense
    point = [Fraction(1, 3), Fraction(2, 5), Fraction(1, 7), Fraction(3, 4), Fraction(1, 2)]
    value_a = value_b = Fraction(0)
    for monom in np.ndindex(a.shape):
        weight = math.prod(p ** k for p, k in zip(point, monom))
        value_a += int(a[monom]) * weight
        value_b += int(b[monom]) * weight
    g = polys["f"].eval(polys["symbols"]["34"], 1)
    reference = g.as_expr().subs(dict(zip(list(polys["symbols"].values())[:5],
                                          [sp.Rational(p.numerator, p.denominator) for p in point])))
    assert sp.simplify((sp.Rational(value_a.numerator, value_a.denominator)
                        + sp.sqrt(3) * sp.Rational(value_b.numerator, value_b.denominator))
                       / denominator - reference) == 0


def test_signs_in_z_sqrt3_are_exact():
    a = np.array([2, 1, -7, 7, 0, 0, -1, 26], dtype=object)
    b = np.array([-1, -1, 4, -4, 0, 1, 1, -15], dtype=object)
    # 2-sqrt3 > 0, 1-sqrt3 < 0, -7+4sqrt3 < 0, 7-4sqrt3 > 0, 0, sqrt3, -1+sqrt3 > 0,
    # 26-15sqrt3 = 0.0192... > 0
    assert signs(a, b).tolist() == [1, -1, -1, 1, 0, 1, 1, 1]


def test_bernstein_coefficients_of_a_known_polynomial():
    # (x - 1/2)^2 = 1/4 - x + x^2 on [0, 1]: Bernstein coefficients 1/4, -1/4, 1/4;
    # on [1/2, 1] it is y^2/4 with y in [0, 1]: coefficients 0, 0, 1/4.
    a = np.array([Fraction(1, 4), Fraction(-1), Fraction(1)], dtype=object) * 4
    a = np.array([int(x) for x in a], dtype=object)
    b = np.zeros(3, dtype=object)
    b[...] = 0
    whole = bernstein_on_box(a, b, [(0, 1)])[0]
    half = bernstein_on_box(a, b, [(Fraction(1, 2), 1)])[0]
    assert [Fraction(int(x), int(whole[0])) for x in whole] == [1, -1, 1]
    assert [int(x) for x in half[:2]] == [0, 0] and int(half[2]) > 0


def test_cover_is_the_appendix_table_and_covers_the_cube():
    rectangles = cover()
    kinds = [kind.split()[0] for _, kind, _ in rectangles]
    assert len(rectangles) == 54
    assert kinds.count("box") == 44              # 28 ordinary + 16 opposite-pair subboxes
    assert kinds.count("small-chart") == 5 and kinds.count("large-chart") == 5
    assert covers_unit_cube(samples=3000) == 0


def test_chart_degrees_match_the_appendix(dense):
    a, b, _ = dense
    small = chart(a, b, 0, 12)
    za, zb = shift_to_corner(a, b, (1,) * 5)
    large = chart(za, zb, 0, 2)
    assert small[0].shape == (7, 7, 7, 7, 7)
    assert large[0].shape == (17, 7, 7, 7, 7)
    with pytest.raises(ValueError):          # g has no term below degree 12
        chart(a, b, 0, 13)


def test_certificate_passes_and_a_perturbed_one_fails(polys, dense):
    a, b, _ = dense
    records = certify(a, b)
    assert sum(r["coefficients"] for r in records) == 1_027_628
    assert sum(r["negative"] for r in records) == 0
    assert sum(r["zero"] for r in records) == 29_204

    import sympy as sp

    field = polys["field"]
    b = polys["symbols"]
    # the inequality is sharp at beta = 1: a 0.1% larger kappa breaks the
    # chart factorisation that condition (b) of Proposition A.2 requires
    tighter = (polys["phi"] ** 3 - polys["psi"] * sp.Poly(
        ((1 + 2 * sp.sqrt(3)) ** 3 / 128 * sp.Rational(1001, 1000)) ** 2,
        *b.values(), domain=field))
    ta, tb, _ = dense_integer_form(tighter.eval(b["34"], 1), field)
    with pytest.raises(ValueError):
        certify(ta, tb)
    # and a dent that makes g negative inside the ordinary box LLSLL must give
    # negative Bernstein coefficients there
    point = {b["12"]: sp.Rational(1, 2), b["13"]: sp.Rational(1, 2), b["14"]: sp.Rational(1, 8),
             b["23"]: sp.Rational(1, 2), b["24"]: sp.Rational(1, 2), b["34"]: 1}
    value = polys["f"].as_expr().subs(point)
    bump = b["12"] ** 6 * b["13"] ** 6 * b["14"] ** 6
    dented = polys["f"] - sp.Poly(sp.Rational(101, 100) * value / bump.subs(point) * bump,
                                  *b.values(), domain=field)
    assert sp.N(dented.as_expr().subs(point)) < 0
    da, db, _ = dense_integer_form(dented.eval(b["34"], 1), field)
    box = next(box for label, _, box in cover() if label == "macro LLSLL")
    ba, bb = bernstein_on_box(da, db, box)
    assert (signs(ba, bb) < 0).any()


def test_zonotope_geometry_at_the_truncated_octahedron():
    quotient, volume = zonotope_quotient(np.ones(6), np.eye(3))
    assert quotient == pytest.approx(TRUNCATED_OCTAHEDRON, rel=1e-12)
    assert volume == pytest.approx(16.0, rel=1e-12)
    rng = np.random.default_rng(3)
    for _ in range(5):
        beta = rng.exponential(size=6)
        phi, psi, d = phi_psi_d_float(beta)
        q, v = zonotope_quotient(beta, np.eye(3))
        assert v == pytest.approx(d, rel=1e-10)
        assert q >= 6 * (phi ** 3 / psi) ** (1 / 6) - 1e-12 >= TRUNCATED_OCTAHEDRON - 1e-9


def test_permutohedron_closed_forms_match_the_cofactor_formula():
    import sympy as sp

    from experiments.exp044_hales_song_certificate import (
        graphical_generators,
        permutohedron_quotient,
        zonotope_measures,
    )

    assert sp.simplify(permutohedron_quotient(3) - 3 * (1 + 2 * sp.sqrt(3)) / 4 ** sp.Rational(2, 3)) == 0
    assert sp.simplify(permutohedron_quotient(2) - 2 * sp.sqrt(2) * 3 ** sp.Rational(1, 4)) == 0
    assert sp.simplify(permutohedron_quotient(4)
                       - 20 * (16 + 3 * sp.sqrt(6)) / 5 ** sp.Rational(21, 8)) == 0
    for n in (2, 3, 4):
        quotient, volume = zonotope_measures(graphical_generators(n, np.ones(n * (n + 1) // 2)))
        assert quotient == pytest.approx(float(permutohedron_quotient(n)), rel=1e-12)
        assert volume == pytest.approx((n + 1) ** (n - 1), rel=1e-12)      # Cayley


def test_the_24_cell_beats_the_four_dimensional_permutohedron():
    import itertools

    import sympy as sp
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    from experiments.exp044_hales_song_certificate import (
        permutohedron_quotient,
        twenty_four_cell_quotient,
    )

    assert sp.simplify(twenty_four_cell_quotient() - 2 ** sp.Rational(11, 4)) == 0
    roots = [v for v in itertools.product((-1, 0, 1), repeat=4) if sum(map(abs, v)) == 2]
    halfspaces = np.array([list(v) + [-1.0] for v in roots], float)    # x.v <= |v|^2 / 2 = 1
    cell = ConvexHull(HalfspaceIntersection(halfspaces, np.zeros(4)).intersections)
    assert cell.area / cell.volume ** 0.75 == pytest.approx(2 ** 2.75, rel=1e-12)
    assert 2 ** 2.75 < float(permutohedron_quotient(4))


def test_general_phi_psi_reduces_to_the_three_dimensional_one():
    from experiments.exp044_hales_song_certificate import phi_psi_general

    beta = np.random.default_rng(0).exponential(size=6)
    phi, psi = phi_psi_general(3, beta)
    phi3, psi3, _ = phi_psi_d_float(beta)
    assert phi == pytest.approx(phi3, rel=1e-12) and psi == pytest.approx(psi3, rel=1e-12)


def test_three_dimensional_bound_is_lowest_at_the_truncated_octahedron_on_every_stratum():
    from experiments.exp044_hales_song_certificate import (
        chain_bound_strata,
        connected_graphs,
    )

    assert len(connected_graphs(4)) == 6 and len(connected_graphs(5)) == 21
    strata = chain_bound_strata(3, starts=2)
    full = next(s for s in strata if s["edges"] == 6)
    assert full["minimum"] == pytest.approx(TRUNCATED_OCTAHEDRON, abs=1e-9)
    assert all(s["minimum"] >= TRUNCATED_OCTAHEDRON - 1e-9 for s in strata)
    trees = [s for s in strata if s["edges"] == 3]
    assert all(s["minimum"] == pytest.approx(6.0, abs=1e-9) for s in trees)    # parallelepipeds


def test_equal_weights_are_a_strict_local_minimum_of_the_bound():
    from experiments.exp044_hales_song_certificate import bound_hessian

    for n, blocks in ((3, (2, 3)), (4, (5, 4))):
        eigenvalues = np.sort(bound_hessian(n))
        assert abs(eigenvalues[0]) < 1e-6                     # the scale direction
        rest = eigenvalues[1:]
        assert rest.min() > 1e-3
        # two blocks, sized by the irreducible pieces of S_{n+1} on the edges
        assert np.allclose(rest[:blocks[0]], rest[0], rtol=1e-4)
        assert np.allclose(rest[blocks[0]:], rest[-1], rtol=1e-4)
