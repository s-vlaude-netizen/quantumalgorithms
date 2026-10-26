"""Experiment 044 -- an independent check of the Hales-Song certificate.

Hales & Song, *The Truncated Octahedral Conjecture* (arXiv:2609.00997, v2,
September 2026), prove Bezdek's conjecture: among parallelohedra of fixed
volume the Archimedean truncated octahedron has the least surface area. It is
the flat, one-cell-type relative of the Kelvin problem (exp042). The proof is
analytic except for one inequality in six variables, proved by an exact
Bernstein certificate in their Appendix A::

    f(beta) = phi(beta)^3 - kappa^2 psi(beta) >= 0   for beta in R^6_{>=0},
    kappa = (1 + 2 sqrt 3)^3 / 128.

Their package (``verify_appendix.py``) was written by an AI model, as the
paper's acknowledgements say. This module re-derives the certificate from the
paper's *definitions* with separately written code and does not read or run
their package:

1. **Reconstruction.** ``phi`` and ``psi`` are built from the determinant
   definitions (paper eq. ``phi-psi``, Section 5) of the six-zone model
   (Section 6.1): the regular tetrahedral frame, the seven cuts, the tree
   weights ``gamma_b``, the cut weights ``r_b`` and ``d = tau(beta)``. Their
   verifier goes through the Cauchy-Binet sums over the 29 normal bases
   (eqs. 51-52) instead, so the two constructions meet only in the
   polynomial itself. SymPy, exact arithmetic in ``Q(sqrt 3)``.
2. **The stated facts.** 6083 monomials, multidegree (6,...,6), homogeneous
   of degree 18, invariance under the 24 relabellings of the frame,
   ``f(1) = 0``, minimum total degree 12 after setting ``beta_34 = 1``, and
   no term of degree below 2 in ``g(1 - z)``.
3. **The cover.** All 54 boxes of Appendix A.3: 28 ordinary macro-boxes, 16
   opposite-pair subboxes, and the 5 + 5 charts at the all-small and
   all-large corners. On each box the tensor-product Bernstein coefficients
   are computed in exact integers (a common positive scale is dropped) and
   every sign is decided exactly in ``Z[sqrt 3]``.
4. **The geometric reduction, numerically.** For random zone weights and
   random volume-preserving maps the actual zonotope is built (convex hull)
   and its isoperimetric quotient compared with the chain
   ``iota >= 6 (phi^3/psi)^(1/6) >= 6 kappa^(1/3)`` -- a check that the
   polynomial certified is the one the geometry needs.
5. **The theorem itself, numerically.** ``iota(Z(beta, M))`` minimised over
   all eleven parameters from random starts.
6. **Four dimensions.** Sections 2-5 hold in every dimension; only the
   polynomial inequality ``f_n >= 0`` is proved, for n = 3. Its bound is
   minimised numerically on all 21 strata of the n = 4 weight orthant
   (connected support graphs on five vertices), and the 24-cell -- a
   four-dimensional parallelohedron that is not a zonotope -- is compared
   with the regular permutohedron, in closed form.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qres.bench import RESULTS_DIR

EDGES = ("12", "13", "14", "23", "24", "34")
FRAME = {1: (1, 1, 1), 2: (1, -1, -1), 3: (-1, 1, -1), 4: (-1, -1, 1)}   # n_i = 2^(-2/3) FRAME[i]
TRUNCATED_OCTAHEDRON = 3 * (1 + 2 * math.sqrt(3)) / 4 ** (2 / 3)


# --------------------------------------------------------- reconstruction

def symbols():
    import sympy as sp

    return dict(zip(EDGES, sp.symbols("b12 b13 b14 b23 b24 b34", nonnegative=True)))


def _beta(b, i, j):
    return b[f"{min(i, j)}{max(i, j)}"]


def cuts(b):
    """The seven facet pairs as (integer normal m_b, gamma_b, r_b, |m_b|).

    ``n_b = 2^(-2/3) m_b``. Singleton cuts {i}: ``m_b = m_i``, ``gamma`` is the
    spanning-tree polynomial of the triangle on the other three vertices.
    Two-two cuts {ij|kl}: ``m_b = m_i + m_j``, ``gamma = beta_ij beta_kl``.
    ``r_b`` sums the weights of the edges crossing the cut.
    """
    import sympy as sp

    out = []
    for i in range(1, 5):
        j, k, l = (x for x in range(1, 5) if x != i)
        gamma = (_beta(b, j, k) * _beta(b, j, l) + _beta(b, j, k) * _beta(b, k, l)
                 + _beta(b, j, l) * _beta(b, k, l))
        r = sum(_beta(b, i, x) for x in (j, k, l))
        out.append((FRAME[i], gamma, r, sp.sqrt(3)))
    for (i, j), (k, l) in (((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))):
        normal = tuple(p + q for p, q in zip(FRAME[i], FRAME[j]))
        r = _beta(b, i, k) + _beta(b, i, l) + _beta(b, j, k) + _beta(b, j, l)
        out.append((normal, _beta(b, i, j) * _beta(b, k, l), r, sp.Integer(2)))
    return out


def zone_polynomials():
    """``phi``, ``psi``, ``d`` and ``f`` as SymPy Polys over ``Q(sqrt 3)``.

    With ``n_b = 2^(-2/3) m_b`` the definitions
    ``phi = det(sum gamma_b/|n_b| n_b n_b^T)`` and
    ``psi = d^3 det(sum gamma_b r_b/|n_b|^3 n_b n_b^T)`` become
    ``phi = det(sum gamma_b/|m_b| m_b m_b^T) / 4`` and
    ``psi = 4 d^3 det(sum gamma_b r_b/|m_b|^3 m_b m_b^T)``.
    """
    import sympy as sp

    b = symbols()
    gens = list(b.values())
    field = sp.QQ.algebraic_field(sp.sqrt(3))

    def poly(expr):
        return sp.Poly(expr, *gens, domain=field)

    laplacian = sp.zeros(4, 4)
    for i, j in itertools.permutations(range(1, 5), 2):
        laplacian[i - 1, j - 1] = -_beta(b, i, j)
        laplacian[i - 1, i - 1] += _beta(b, i, j)
    d = poly(sp.expand(laplacian[1:, 1:].det()))

    def matrix(weight):
        entries = [[poly(0)] * 3 for _ in range(3)]
        for normal, gamma, r, length in cuts(b):
            w = weight(poly(gamma), poly(r), length)
            for i, j in itertools.product(range(3), repeat=2):
                if normal[i] * normal[j]:
                    entries[i][j] = entries[i][j] + w * poly(normal[i] * normal[j])
        return entries

    def det3(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))

    phi = det3(matrix(lambda g, r, length: g * poly(1 / length))) * poly(sp.Rational(1, 4))
    psi = d ** 3 * det3(matrix(lambda g, r, length: g * r * poly(1 / length ** 3))) * poly(4)
    kappa = poly((1 + 2 * sp.sqrt(3)) ** 3 / 128)
    f = phi ** 3 - psi * kappa ** 2
    return {"phi": phi, "psi": psi, "d": d, "f": f, "symbols": b, "field": field}


def vertex_relabellings():
    """The 24 permutations of the six edge variables induced by permuting the
    four frame vectors, as index maps ``edge k -> edge perm[k]``."""
    for perm in itertools.permutations(range(1, 5)):
        image = dict(zip(range(1, 5), perm))
        yield [EDGES.index("".join(sorted(str(image[int(v)]) for v in e))) for e in EDGES]


def invariant_under(poly, perm):
    """Whether relabelling the variables by ``perm`` leaves ``poly`` unchanged:
    a variable substitution permutes exponent positions, so compare terms."""
    terms = dict(poly.rep.terms())
    moved = {}
    for monom, coeff in terms.items():
        image = [0] * len(monom)
        for k, power in enumerate(monom):
            image[perm[k]] = power
        moved[tuple(image)] = coeff
    return moved == terms


def stated_facts(polys):
    """Appendix A.2's degree data and the symmetry used to normalise."""
    f, b = polys["f"], polys["symbols"]
    ones = dict.fromkeys(b.values(), 1)
    degrees = {sum(m) for m in f.monoms()}
    invariant = all(invariant_under(f, perm) for perm in vertex_relabellings())
    g = f.eval(b["34"], 1)
    g_degrees = [sum(m) for m in g.monoms()]
    return {
        "f_monomials": len(f.terms()),
        "f_multidegree": list(f.degree_list()),
        "f_homogeneous_degree": sorted(degrees),
        "f_at_one": str(f.as_expr().subs(ones)),
        "f_invariant_under_24_relabellings": invariant,
        "phi_monomials": len(polys["phi"].terms()),
        "d_at_one": int(polys["d"].as_expr().subs(ones)),
        "g_monomials": len(g.terms()),
        "g_multidegree": list(g.degree_list()),
        "g_total_degree_range": [min(g_degrees), max(g_degrees)],
    }


# ------------------------------------------------- exact dense integer form

def dense_integer_form(g, field):
    """``g = (A + B sqrt 3) / D`` with integer tensors A, B indexed by exponents.

    D is a positive integer, so the sign of every coefficient is the sign of
    the integer pair, which is all a nonnegativity certificate needs.
    """
    import sympy as sp

    if field.ext.as_expr() != sp.sqrt(3):
        raise ValueError("expected the field Q(sqrt 3) generated by sqrt 3")
    shape = tuple(k + 1 for k in g.degree_list())
    pairs = {}
    denominator = 1
    for monom, coeff in g.rep.terms():     # coeff = q sqrt3 + p, stored as [q, p]
        digits = [Fraction(int(c.numerator), int(c.denominator)) for c in coeff.to_list()]
        p = digits[-1] if digits else Fraction(0)
        q = digits[-2] if len(digits) > 1 else Fraction(0)
        pairs[monom] = (p, q)
        denominator = math.lcm(denominator, p.denominator, q.denominator)
    a = np.zeros(shape, dtype=object)
    bb = np.zeros(shape, dtype=object)
    a[...] = 0
    bb[...] = 0
    for monom, (p, q) in pairs.items():
        a[monom] = int(p * denominator)
        bb[monom] = int(q * denominator)
    return a, bb, denominator


def taylor_shift(coeffs, axis, origin, scale):
    """Coefficients of ``p(origin + scale * x)`` along ``axis`` (exact)."""
    m = coeffs.shape[axis] - 1
    origin, scale = Fraction(origin), Fraction(scale)
    matrix = [[Fraction(0)] * (m + 1) for _ in range(m + 1)]
    for k in range(m + 1):          # x^k <- (origin + scale x)^k
        for j in range(k + 1):
            matrix[j][k] = comb(k, j) * origin ** (k - j) * scale ** j
    return _apply_rational(coeffs, axis, matrix)


def to_bernstein(coeffs, axis):
    """Power-basis coefficients on [0, 1] to Bernstein coefficients along
    ``axis``: ``b_i = sum_{j <= i} C(i, j) / C(m, j) a_j``."""
    m = coeffs.shape[axis] - 1
    matrix = [[Fraction(comb(i, j), comb(m, j)) if j <= i else Fraction(0)
               for j in range(m + 1)] for i in range(m + 1)]
    return _apply_rational(coeffs, axis, matrix)


def _apply_rational(coeffs, axis, matrix):
    """Multiply ``coeffs`` along ``axis`` by a rational matrix, scaled by the
    positive lcm of its denominators so everything stays an integer."""
    scale = 1
    for row in matrix:
        for x in row:
            scale = math.lcm(scale, x.denominator)
    integer = np.array([[int(x * scale) for x in row] for row in matrix], dtype=object)
    moved = np.moveaxis(coeffs, axis, 0)
    out = np.tensordot(integer, moved, axes=(1, 0))
    return np.moveaxis(out, 0, axis)


def signs(a, b):
    """Exact signs of ``a + b sqrt 3`` for integer arrays: -1, 0 or 1."""
    flat_a, flat_b = a.ravel(), b.ravel()
    out = np.empty(flat_a.shape, dtype=np.int8)
    for k, (x, y) in enumerate(zip(flat_a, flat_b)):
        if x >= 0 and y >= 0:
            out[k] = 1 if (x or y) else 0
        elif x <= 0 and y <= 0:
            out[k] = -1
        elif x > 0:                     # y < 0
            out[k] = 1 if x * x > 3 * y * y else -1
        else:                           # x < 0 < y
            out[k] = 1 if 3 * y * y > x * x else -1
    return out.reshape(a.shape)


def bernstein_on_box(a, b, box):
    """Bernstein coefficients (A, B pair) of the polynomial on ``box``,
    a list of (lo, hi) per axis."""
    for axis, (lo, hi) in enumerate(box):
        a = to_bernstein(taylor_shift(a, axis, lo, Fraction(hi) - Fraction(lo)), axis)
        b = to_bernstein(taylor_shift(b, axis, lo, Fraction(hi) - Fraction(lo)), axis)
    return a, b


# ------------------------------------------------------------- the charts

def chart(a, b, j, power):
    """The chart polynomial of A.3 at a corner, from dense coefficients in
    corner-centred variables x: ``x_j = t``, ``x_i = t u_i``; checks that the
    lowest ``power`` total degrees vanish, divides by ``t^power`` and returns
    the tensor in (t, u without j)."""
    m = a.shape
    top = sum(k - 1 for k in m)
    shape = (top - power + 1,) + tuple(m[i] for i in range(len(m)) if i != j)
    out_a = np.zeros(shape, dtype=object)
    out_b = np.zeros(shape, dtype=object)
    out_a[...] = 0
    out_b[...] = 0
    for monom in itertools.product(*(range(k) for k in m)):
        x, y = a[monom], b[monom]
        if not (x or y):
            continue
        degree = sum(monom)
        if degree < power:
            raise ValueError(f"term of degree {degree} below the chart factor t^{power}")
        index = (degree - power,) + tuple(monom[i] for i in range(len(m)) if i != j)
        out_a[index] += x
        out_b[index] += y
    # trim t to its true degree: certifying in an elevated degree would also
    # be valid, but A.3 states the degrees, and these are what get compared
    used = [k for k in range(shape[0]) if np.any(out_a[k] != 0) or np.any(out_b[k] != 0)]
    return out_a[:max(used) + 1], out_b[:max(used) + 1]


def shift_to_corner(a, b, corner):
    """Coefficients in ``z`` of ``g(corner - z)`` (``corner`` per axis 0 or 1):
    ``x_i = c_i - z_i`` maps the all-large corner to the origin."""
    for axis, c in enumerate(corner):
        a = taylor_shift(a, axis, c, -1 if c else 1)
        b = taylor_shift(b, axis, c, -1 if c else 1)
    return a, b


# ---------------------------------------------------------------- the cover

SMALL, LARGE = (Fraction(0), Fraction(1, 4)), (Fraction(1, 4), Fraction(1))
COORDS = ("12", "13", "14", "23", "24")                    # beta_34 = 1


def cover():
    """The 54 certificate rectangles of Appendix A.3, as (label, kind, box).

    ``kind`` is "box" (a box in beta), "small-chart j" or "large-chart j".
    """
    special = {
        (True,) * 5: "all large", (False,) * 5: "all small",
        tuple(c not in ("13", "24") for c in COORDS): "13,24 small",
        tuple(c not in ("14", "23") for c in COORDS): "14,23 small",
    }
    out = []
    for large in itertools.product((False, True), repeat=5):
        box = [LARGE if x else SMALL for x in large]
        if large not in special:
            out.append(("macro " + "".join("L" if x else "S" for x in large), "box", box))
            continue
        kind = special[large]
        if kind.endswith("small") and kind != "all small":
            for halves in itertools.product((0, 1), repeat=3):
                sub, h = [], iter(halves)
                for x in large:
                    if x:
                        half = next(h)
                        sub.append((Fraction(1, 4), Fraction(1, 2)) if half == 0
                                   else (Fraction(1, 2), Fraction(1)))
                    else:
                        sub.append(SMALL)
                out.append((f"{kind} sub{''.join(map(str, halves))}", "box", sub))
    for j in range(5):
        out.append((f"all-small chart {COORDS[j]}", f"small-chart {j}",
                    [(Fraction(0), Fraction(1, 4))] + [(Fraction(0), Fraction(1))] * 4))
    for j in range(5):
        out.append((f"all-large chart {COORDS[j]}", f"large-chart {j}",
                    [(Fraction(0), Fraction(3, 4))] + [(Fraction(0), Fraction(1))] * 4))
    return out


def certify(a, b, verbose=False):
    """Bernstein-certify ``g = (A + B sqrt3)/D >= 0`` on ``[0,1]^5`` over the
    cover. Returns one record per rectangle."""
    small_charts = [chart(a, b, j, 12) for j in range(5)]
    za, zb = shift_to_corner(a, b, (1,) * 5)
    large_charts = [chart(za, zb, j, 2) for j in range(5)]
    records = []
    for label, kind, box in cover():
        start = time.time()
        if kind == "box":
            pa, pb = a, b
        elif kind.startswith("small-chart"):
            pa, pb = small_charts[int(kind.split()[-1])]
        else:
            pa, pb = large_charts[int(kind.split()[-1])]
        ba, bb = bernstein_on_box(pa, pb, box)
        s = signs(ba, bb)
        largest = max(max(abs(int(x)) for x in ba.ravel()), max(abs(int(x)) for x in bb.ravel()))
        shift = max(largest.bit_length() - 900, 0)          # keep floats in range
        value = np.array([float(int(x) >> shift) + math.sqrt(3) * float(int(y) >> shift)
                          for x, y in zip(ba.ravel(), bb.ravel())]).reshape(ba.shape)
        positive = value[s > 0]
        records.append({
            "rectangle": label, "multidegree": [k - 1 for k in pa.shape],
            "coefficients": int(s.size), "negative": int((s < 0).sum()),
            "zero": int((s == 0).sum()),
            "smallest_positive_relative": float(positive.min() / np.abs(value).max())
            if positive.size else None,
            "seconds": round(time.time() - start, 2),
        })
        if verbose:
            print(f"  {label:28s} {records[-1]['coefficients']:6d} coefficients, "
                  f"{records[-1]['negative']} negative, {records[-1]['zero']} zero "
                  f"({records[-1]['seconds']} s)", flush=True)
    return records


def covers_unit_cube(samples=20000, seed=0):
    """Monte Carlo check that every sampled point of [0,1]^5 lies in some
    rectangle of the cover (charts mapped back to beta)."""
    rng = np.random.default_rng(seed)
    rects = cover()
    missed = 0
    for x in rng.random((samples, 5)):
        hit = False
        for _, kind, box in rects:
            if kind == "box":
                hit = all(float(lo) <= v <= float(hi) for v, (lo, hi) in zip(x, box))
            elif kind.startswith("small-chart"):
                j = int(kind.split()[-1])
                t = x[j]
                hit = t <= 0.25 and x.max() <= t
            else:
                j = int(kind.split()[-1])
                z = 1 - x
                hit = z[j] <= 0.75 and z.max() <= z[j]
            if hit:
                break
        missed += not hit
    return missed


# ------------------------------------------------- geometric reduction check

def zonotope_quotient(beta, matrix):
    """Isoperimetric quotient of Z(beta, M) from its convex hull."""
    from scipy.spatial import ConvexHull

    frame = {i: np.array(v, float) * 2 ** (-2 / 3) for i, v in FRAME.items()}
    generators = []
    for e, weight in zip(EDGES, beta):
        i, j = int(e[0]), int(e[1])
        w = (frame[i] - frame[j]) / 4 ** (1 / 3)     # w_ij = (n+1)^(-1/n) (n_i - n_j)
        generators.append(weight * (matrix @ w))
    points = np.array([sum(s * g / 2 for s, g in zip(signs_, generators))
                       for signs_ in itertools.product((-1, 1), repeat=len(generators))])
    hull = ConvexHull(points)
    return hull.area / hull.volume ** (2 / 3), hull.volume


def phi_psi_d_float(beta):
    """``phi``, ``psi`` and ``d`` in floating point straight from the paper's
    definitions with the actual frame ``n_i`` (no integer rescaling), as an
    independent route to the exact construction."""
    weight = dict(zip(EDGES, beta))

    def w(i, j):
        return weight[f"{min(i, j)}{max(i, j)}"]

    frame = {i: np.array(v, float) * 2 ** (-2 / 3) for i, v in FRAME.items()}
    laplacian = np.zeros((4, 4))
    for i, j in itertools.permutations(range(1, 5), 2):
        laplacian[i - 1, j - 1] = -w(i, j)
        laplacian[i - 1, i - 1] += w(i, j)
    d = np.linalg.det(laplacian[1:, 1:])
    first, second = np.zeros((3, 3)), np.zeros((3, 3))
    shores = [(i,) for i in range(1, 5)] + [(1, 2), (1, 3), (1, 4)]
    for shore in shores:
        rest = [x for x in range(1, 5) if x not in shore]
        normal = sum(frame[i] for i in shore)
        if len(shore) == 1:
            j, k, l = rest
            gamma = w(j, k) * w(j, l) + w(j, k) * w(k, l) + w(j, l) * w(k, l)
        else:
            gamma = w(*shore) * w(*rest)
        r = sum(w(i, j) for i in shore for j in rest)
        length = np.linalg.norm(normal)
        first += gamma / length * np.outer(normal, normal)
        second += gamma * r / length ** 3 * np.outer(normal, normal)
    return np.linalg.det(first), d ** 3 * np.linalg.det(second), d


def evaluate(poly, point):
    """Floating-point value of a SymPy Poly over Q(sqrt 3) at ``point``."""
    total = 0.0
    for monom, coeff in poly.terms():
        total += float(coeff) * math.prod(x ** k for x, k in zip(point, monom))
    return total


def reduction_check(polys, samples=300, seed=1):
    """Sampled ``iota(Z) >= 6 (phi^3/psi)^(1/6) >= 6 kappa^(1/3)``,
    ``vol = d(beta)``, and the exact ``f`` against the float definitions;
    returns the worst slacks."""
    rng = np.random.default_rng(seed)
    kappa = (1 + 2 * math.sqrt(3)) ** 3 / 128
    worst_geometry, worst_polynomial, worst_volume = math.inf, math.inf, 0.0
    worst_exact = 0.0
    for k in range(samples):
        beta = rng.exponential(size=6) * (rng.random(6) < 0.9) + 1e-3
        x = rng.normal(size=(3, 3))
        matrix = x / abs(np.linalg.det(x)) ** (1 / 3) * np.sign(np.linalg.det(x))
        if rng.random() < 0.3:
            matrix = np.linalg.qr(rng.normal(size=(3, 3)))[0]
            matrix *= np.sign(np.linalg.det(matrix))
        quotient, volume = zonotope_quotient(beta, matrix)
        phi, psi, d = phi_psi_d_float(beta)
        bound = 6 * (phi ** 3 / psi) ** (1 / 6)
        worst_geometry = min(worst_geometry, quotient - bound)
        worst_polynomial = min(worst_polynomial, bound - TRUNCATED_OCTAHEDRON)
        worst_volume = max(worst_volume, abs(volume / d - 1))
        if k < 20:      # the exact polynomial agrees with the float definitions
            f_float = phi ** 3 - kappa ** 2 * psi
            f_exact = evaluate(polys["f"], beta)
            worst_exact = max(worst_exact, abs(f_exact - f_float) / phi ** 3)
    return {"samples": samples, "min iota - 6(phi^3/psi)^(1/6)": worst_geometry,
            "min 6(phi^3/psi)^(1/6) - iota(truncated octahedron)": worst_polynomial,
            "max |hull volume / d(beta) - 1|": worst_volume,
            "max |f exact - f float| / phi^3 (20 samples)": worst_exact}


# --------------------------------------------------------- other dimensions

def regular_frame(n):
    """Rows n_1..n_{n+1} of the regular balanced frame in R^n, normalised by
    ``|det(n_1, ..., n_n)| = 1`` (paper eq. ``normal-gram``)."""
    centred = np.eye(n + 1) - 1.0 / (n + 1)
    basis = np.linalg.qr(np.vstack([np.ones(n + 1), np.eye(n + 1)[:-1]]).T)[0][:, 1:]
    frame = centred @ basis
    return frame / abs(np.linalg.det(frame[:n])) ** (1 / n)


def graphical_generators(n, beta, matrix=None):
    """Generators ``beta_ij M w_ij`` of the K_{n+1}-graphical zonotope."""
    frame = regular_frame(n)
    matrix = np.eye(n) if matrix is None else matrix
    pairs = list(itertools.combinations(range(n + 1), 2))
    return [b * (matrix @ ((frame[i] - frame[j]) / (n + 1) ** (1 / n)))
            for b, (i, j) in zip(beta, pairs)]


def zonotope_measures(generators):
    """Isoperimetric quotient and volume of the zonotope sum [-v/2, v/2] from
    the determinant and cofactor formulas (paper eqs. 6-7), any dimension."""
    n = len(generators[0])
    volume = sum(abs(np.linalg.det(np.array(s))) for s in itertools.combinations(generators, n))
    area = 0.0
    for s in itertools.combinations(generators, n - 1):
        rows = np.array(s)
        cofactor = [np.linalg.det(np.delete(rows, k, axis=1)) for k in range(n)]
        area += 2 * np.linalg.norm(cofactor)
    return area / volume ** ((n - 1) / n), volume


def permutohedron_quotient(n):
    """Exact isoperimetric quotient of the regular permutohedron in R^n.

    Volume = number of spanning trees of K_{n+1} = (n+1)^(n-1). Half the
    boundary = sum over bipartitions (S, S^c) of the two-tree forests with
    those components, tau(K_s) tau(K_{n+1-s}) = s^(s-2) (n+1-s)^(n-1-s), times
    ``|n_b| = (n+1)^(-(n-1)/(2n)) sqrt(s (n+1-s))``.
    """
    import sympy as sp

    half_area = 0
    for s in range(1, n + 1):
        pairs = sp.binomial(n + 1, s) / 2                     # unordered bipartitions
        forests = sp.Integer(s) ** (s - 2) * sp.Integer(n + 1 - s) ** (n - 1 - s)
        length = (n + 1) ** sp.Rational(-(n - 1), 2 * n) * sp.sqrt(s * (n + 1 - s))
        half_area += pairs * forests * length
    volume = sp.Integer(n + 1) ** (n - 1)
    return sp.nsimplify(sp.simplify(2 * half_area / volume ** sp.Rational(n - 1, n)))


def twenty_four_cell_quotient():
    """The 24-cell, Voronoi cell of D4 and a four-dimensional parallelohedron:
    hypervolume 2 a^4 and 24 octahedral facets of volume (sqrt 2/3) a^3."""
    import sympy as sp

    return sp.simplify(24 * sp.sqrt(2) / 3 / sp.Integer(2) ** sp.Rational(3, 4))


def phi_psi_general(n, beta):
    """``phi`` and ``psi`` of Section 5 in R^n, floating point."""
    frame = regular_frame(n)
    pairs = list(itertools.combinations(range(n + 1), 2))
    weight = dict(zip(pairs, beta))

    def tau(nodes):
        if len(nodes) == 1:
            return 1.0
        lap = np.zeros((len(nodes), len(nodes)))
        for a, i in enumerate(nodes):
            for c, j in enumerate(nodes):
                if i != j:
                    x = weight[(min(i, j), max(i, j))]
                    lap[a, c] = -x
                    lap[a, a] += x
        return np.linalg.det(lap[1:, 1:])

    first, second, seen = np.zeros((n, n)), np.zeros((n, n)), set()
    for size in range(1, n + 1):
        for shore in itertools.combinations(range(n + 1), size):
            rest = tuple(v for v in range(n + 1) if v not in shore)
            if min(shore, rest) in seen:
                continue
            seen.add(min(shore, rest))
            normal = frame[list(shore)].sum(axis=0)
            gamma = tau(shore) * tau(rest)
            r = sum(weight[(min(i, j), max(i, j))] for i in shore for j in rest)
            length = np.linalg.norm(normal)
            first += gamma / length * np.outer(normal, normal)
            second += gamma * r / length ** 3 * np.outer(normal, normal)
    d = tau(tuple(range(n + 1)))
    return np.linalg.det(first), d ** (2 * n - 3) * np.linalg.det(second)


def connected_graphs(vertices):
    """One edge set per isomorphism class of connected spanning graphs on
    ``vertices`` vertices -- the strata of the closed weight orthant where the
    graphical zonotope stays full-dimensional (zero weight = deleted zone)."""
    pairs = list(itertools.combinations(range(vertices), 2))
    classes = {}
    for mask in range(1, 2 ** len(pairs)):
        edges = [pairs[k] for k in range(len(pairs)) if mask >> k & 1]
        reach, frontier = {0}, [0]
        while frontier:
            v = frontier.pop()
            for a, b in edges:
                for x, y in ((a, b), (b, a)):
                    if x == v and y not in reach:
                        reach.add(y)
                        frontier.append(y)
        if len(reach) < vertices:
            continue
        key = min(tuple(sorted(tuple(sorted((p[a], p[b]))) for a, b in edges))
                  for p in itertools.permutations(range(vertices)))
        classes.setdefault(key, [pairs.index(e) for e in key])
    return list(classes.values())


def chain_bound_strata(n, starts=3, seed=4):
    """Minimise the Hales-Song bound ``2n (phi^3/psi)^(1/(2n))`` on every
    stratum (connected support graph) of the weight orthant in R^n."""
    from scipy.optimize import minimize

    m = n * (n + 1) // 2
    rng = np.random.default_rng(seed)
    out = []
    for support in connected_graphs(n + 1):

        def bound(x, support=support):
            beta = np.zeros(m)
            beta[support] = np.exp(np.clip(x - x.max(), -30, 0))
            phi, psi = phi_psi_general(n, beta)
            value = 2 * n * (phi ** 3 / psi) ** (1 / (2 * n)) if psi > 0 else np.inf
            return value if np.isfinite(value) else 1e9

        best = math.inf
        for _ in range(starts):
            x0 = rng.normal(scale=1.0, size=len(support))
            if len(support) == 1:
                best = min(best, bound(x0))
                continue
            r = minimize(bound, x0, method="Nelder-Mead",
                         options={"maxfev": 8000, "xatol": 1e-10, "fatol": 1e-13, "adaptive": True})
            best = min(best, float(minimize(bound, r.x, method="BFGS").fun))
        out.append({"edges": len(support), "support": support, "minimum": best})
    return out


def bound_hessian(n, step=1e-4):
    """Eigenvalues of the Hessian of ``2n (phi^3/psi)^(1/(2n))`` in log-weights
    at equal weights: one zero (the scale), the rest positive if the equality
    point is a strict local minimum. By symmetry they come in two blocks, one
    per non-trivial irreducible piece of the edge representation of S_{n+1}."""
    m = n * (n + 1) // 2

    def bound(x):
        phi, psi = phi_psi_general(n, np.exp(x))
        return 2 * n * (phi ** 3 / psi) ** (1 / (2 * n))

    hessian = np.zeros((m, m))
    for i, j in itertools.combinations_with_replacement(range(m), 2):
        a, b = np.eye(m)[i] * step, np.eye(m)[j] * step
        hessian[i, j] = hessian[j, i] = (bound(a + b) - bound(a - b) - bound(b - a)
                                         + bound(-a - b)) / (4 * step * step)
    return np.linalg.eigvalsh(hessian).tolist()


def dimension_four(quotient_starts=3, seed=4):
    """Does the Hales-Song chain extend to n = 4? The bound on all 21 strata,
    and the quotient itself minimised over weights and M in SL(4); compared
    with the regular permutohedron and with the 24-cell."""
    from scipy.linalg import expm
    from scipy.optimize import minimize

    n, m = 4, 10

    def quotient(x):
        a = x[m:].reshape(n, n)
        beta = np.exp(np.clip(x[:m] - x[:m].max(), -30, 0))
        return zonotope_measures(graphical_generators(n, beta,
                                                      expm(a - np.trace(a) / n * np.eye(n))))[0]

    rng = np.random.default_rng(seed)
    quotients = []
    for _ in range(quotient_starts):
        y0 = np.concatenate([rng.normal(scale=1.5, size=m), rng.normal(scale=0.4, size=n * n)])
        q = minimize(quotient, y0, method="Nelder-Mead",
                     options={"maxfev": 40000, "xatol": 1e-9, "fatol": 1e-12, "adaptive": True})
        quotients.append(float(minimize(quotient, q.x, method="BFGS").fun))
    return {"strata": chain_bound_strata(4, seed=seed), "quotient_minima": quotients,
            "hessian_at_equal_weights": bound_hessian(4),
            "permutohedron": float(permutohedron_quotient(4)),
            "twenty_four_cell": float(twenty_four_cell_quotient())}


def minimise_quotient(starts=20, seed=2):
    """Direct numerical test of the theorem on the truncated-octahedral type:
    minimise ``iota(Z(beta, M))`` over the six zone weights and ``M`` in
    SL(3) (as exp of a traceless matrix) from random starts."""
    from scipy.linalg import expm
    from scipy.optimize import minimize

    def unpack(x):
        beta = np.exp(x[:6])
        traceless = x[6:].reshape(3, 3) - np.trace(x[6:].reshape(3, 3)) / 3 * np.eye(3)
        return beta, expm(traceless)

    def objective(x):
        beta, matrix = unpack(x)
        return zonotope_quotient(beta, matrix)[0]

    rng = np.random.default_rng(seed)
    best = []
    for _ in range(starts):
        x0 = np.concatenate([rng.normal(scale=1.0, size=6), rng.normal(scale=0.5, size=9)])
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-10,
                                   "fatol": 1e-13, "adaptive": True})
        result = minimize(objective, result.x, method="BFGS", options={"gtol": 1e-10})
        beta, _ = unpack(result.x)
        best.append({"quotient": float(result.fun),
                     "weight_spread": float(beta.max() / beta.min())})
    values = [b["quotient"] for b in best]
    return {"starts": starts, "lowest": min(values), "highest": max(values),
            "lowest_minus_truncated_octahedron": min(values) - TRUNCATED_OCTAHEDRON,
            "runs": best}


def main() -> int:
    start = time.time()
    polys = zone_polynomials()
    facts = stated_facts(polys)
    print("reconstruction:", json.dumps(facts, indent=1), flush=True)
    g = polys["f"].eval(polys["symbols"]["34"], 1)
    a, b, denominator = dense_integer_form(g, polys["field"])
    print(f"g = (A + B sqrt3) / {denominator}, tensor shape {a.shape}", flush=True)
    reduction = reduction_check(polys)
    print("geometric reduction:", reduction, flush=True)
    direct = minimise_quotient()
    print(f"direct minimisation: lowest {direct['lowest']:.12f} "
          f"(truncated octahedron {TRUNCATED_OCTAHEDRON:.12f}), highest {direct['highest']:.6f}",
          flush=True)
    strata3 = chain_bound_strata(3)
    four = dimension_four()
    lowest4 = min(four["strata"], key=lambda s: s["minimum"])
    print(f"n = 3 bound over {len(strata3)} strata: lowest {min(s['minimum'] for s in strata3):.12f}",
          flush=True)
    print(f"n = 4 bound over {len(four['strata'])} strata: lowest {lowest4['minimum']:.12f} "
          f"on {lowest4['edges']} edges; quotient minima {four['quotient_minima']}; "
          f"permutohedron {four['permutohedron']:.12f}, 24-cell {four['twenty_four_cell']:.12f}",
          flush=True)
    missed = covers_unit_cube()
    print(f"cover: {missed} of 20000 sampled points outside every rectangle", flush=True)
    records = certify(a, b, verbose=True)
    total = sum(r["coefficients"] for r in records)
    negative = sum(r["negative"] for r in records)
    print(f"{len(records)} rectangles, {total} coefficients, {negative} negative "
          f"({time.time() - start:.0f} s)")
    out = {"facts": facts, "denominator": str(denominator), "reduction_check": reduction,
           "direct_minimisation": direct,
           "bound_strata_n3": strata3, "dimension_four": four,
           "cover_samples_missed": missed, "rectangles": records,
           "coefficients": total, "negative": negative,
           "certified": negative == 0 and missed == 0}
    path = RESULTS_DIR / "exp044_hales_song_certificate.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
