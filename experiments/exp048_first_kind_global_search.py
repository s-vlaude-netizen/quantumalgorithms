"""Experiment 048 -- A_n* is not the best lattice of the first kind once n >= 7.

Result 100 shows that ``A_n*`` is a strict local minimum of the isoperimetric quotient
among lattices for ``3 <= n <= 14`` and not for ``n >= 15``. Local is not global:
Cesaroni & Novaga's second open problem asks about optimality among lattices of
Voronoi's first kind. This module searches that class globally, and the answer turns
out to be no already at n = 7.

A first-kind lattice is a point ``rho`` of positive Selling parameters (one per pair of
the ``n + 1`` superbase vectors; zeros are allowed and give degenerate cells), and its
quotient is, up to a constant of n alone,

    f(rho) = log U(rho) - (1 - 1/(2n)) log d(rho),    U = sum_S tau(S) tau(S^c) sqrt(r(S)),

with ``tau`` the weighted spanning-tree count of the complete graph on a subset,
``r(S)`` the weight of the cut and ``d`` the count for all vertices.

**Search.** The gradient is analytic: ``d log tau / d rho_ij`` is the effective
resistance between ``i`` and ``j`` in the graph on the subset, and ``d r / d rho_ij`` is 1
across the cut. Random-start L-BFGS over the ``n(n+1)/2`` log-parameters reaches
``A_n*`` from most starts for n <= 6 and never goes below it; at n = 7, 8, 9 it ends, from
14% to 48% of the starts, at a lattice **below A_n***, with most Selling parameters at their
lower bound.

**What the lattices are.** Selling parameters 0 or 1 supported on the edges of a graph G
give the *cut lattice* of G (the lattice spanned by the vertex vectors of the incidence
matrix), whose Voronoi cell is the graphical zonotope of G; ``A_n*`` is G = K_{n+1}. The
optimiser finds the complete bipartite graph K_{m,m} (n = 2m - 1) and, for even n, a cone
over K_{m,m} with a lighter apex (n = 2m).

**Certificates.** Both families have closed-form class sums in integers or rationals
(``tau(K_{x,y}) = x^(y-1) y^(x-1)``, and the quotient-matrix eigenvalues for the cone), so
the comparison with ``A_n*`` is exact arithmetic plus interval arithmetic for the square
roots, for every n to 200.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp047_an_star_local_optimality import Cell


class FirstKindQuotient:
    """``f`` and its gradient in the log-Selling parameters of an n-dimensional lattice."""

    def __init__(self, n: int):
        self.n = n
        self.cell = Cell(n)
        self.pairs = self.cell.pair_index
        self.m = self.cell.m
        self.baseline = self.value_and_gradient(np.zeros(self.m))[0]

    def _tau_and_resistances(self, rho, subsets):
        """``tau`` of each subset and its matrix of effective resistances."""
        batch, k = subsets.shape
        if k == 1:
            return np.ones(batch), np.zeros((batch, 1, 1))
        sub = rho[subsets[:, :, None], subsets[:, None, :]]
        lap = -sub
        idx = np.arange(k)
        lap[:, idx, idx] = sub.sum(axis=2)
        reduced = lap[:, 1:, 1:]
        tau = np.linalg.det(reduced)
        inverse = np.linalg.inv(reduced)
        diagonal = np.einsum("bii->bi", inverse)
        resistance = np.zeros((batch, k, k))
        resistance[:, 1:, 1:] = diagonal[:, :, None] + diagonal[:, None, :] - 2 * inverse
        resistance[:, 0, 1:] = diagonal
        resistance[:, 1:, 0] = diagonal
        return tau, resistance

    def value_and_gradient(self, x):
        """``f`` and ``df/dx`` for ``rho = exp(x)`` (entries in the pair order of Cell)."""
        n = self.n
        rho = np.zeros((n + 1, n + 1))
        rho[self.pairs[:, 0], self.pairs[:, 1]] = np.exp(x)
        rho = rho + rho.T
        grad = np.zeros((n + 1, n + 1))                  # upper triangle used
        total = 0.0
        for inside, outside, _ in self.cell.batches:
            tau_in, res_in = self._tau_and_resistances(rho, inside)
            tau_out, _ = self._tau_and_resistances(rho, outside)
            cut = np.array([rho[np.ix_(a, b)].sum() for a, b in zip(inside, outside)])
            term = tau_in * tau_out * np.sqrt(cut)
            total += term.sum()
            k = inside.shape[1]
            for a in range(k):
                for b in range(a + 1, k):
                    i, j = inside[:, a], inside[:, b]
                    np.add.at(grad, (np.minimum(i, j), np.maximum(i, j)), 2 * term * res_in[:, a, b])
            across = term / (2 * cut)
            for a in range(k):
                for c in range(outside.shape[1]):
                    i, j = inside[:, a], outside[:, c]
                    np.add.at(grad, (np.minimum(i, j), np.maximum(i, j)), across)
        everything = np.arange(n + 1)[None, :]
        d, resistance = self._tau_and_resistances(rho, everything)
        d_grad = np.zeros((n + 1, n + 1))
        for a in range(n + 1):
            for b in range(a + 1, n + 1):
                d_grad[a, b] = resistance[0, a, b]
        exponent = 1 - 1 / (2 * n)
        value = np.log(total) - exponent * np.log(d[0])
        gradient_rho = grad / total - exponent * d_grad
        gradient_x = gradient_rho[self.pairs[:, 0], self.pairs[:, 1]] * np.exp(x)
        return float(value), gradient_x

    def safe(self, x):
        """``value_and_gradient`` that survives a numerically singular Laplacian (very
        unequal weights): a large value with no gradient makes the line search back off."""
        try:
            value, gradient = self.value_and_gradient(x)
        except np.linalg.LinAlgError:
            return 1e6, np.zeros_like(x)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            return 1e6, np.zeros_like(x)
        return value, gradient

    def relative(self, x):
        """``f(x) - f(A_n*)``: negative means a lattice better than A_n*."""
        return self.value_and_gradient(x)[0] - self.baseline


def _tau_exact_or_zero(rho, vertices):
    """Exact weighted spanning-tree count; 0 if the weighted graph is disconnected."""
    from fractions import Fraction

    k = len(vertices)
    if k == 1:
        return Fraction(1)
    lap = [[Fraction(0)] * k for _ in range(k)]
    for a, i in enumerate(vertices):
        for b, j in enumerate(vertices):
            if a != b:
                lap[a][b] = -rho[i][j]
                lap[a][a] += rho[i][j]
    mat = [row[1:] for row in lap[1:]]
    size, det = k - 1, Fraction(1)
    for col in range(size):
        pivot = next((r for r in range(col, size) if mat[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
            det = -det
        det *= mat[col][col]
        for r in range(col + 1, size):
            factor = mat[r][col] / mat[col][col]
            if factor:
                for c in range(col, size):
                    mat[r][c] -= factor * mat[col][c]
    return det


def exact_terms_all(n: int, rho):
    """``d`` and the terms ``(coefficient, cut weight)`` of ``U``, summed over every proper
    nonempty subset, in exact rational arithmetic (``rho`` a matrix of Fractions)."""
    import itertools
    from fractions import Fraction

    vertices = list(range(n + 1))
    terms = []
    for size in range(1, n + 1):
        for inside in itertools.combinations(vertices, size):
            outside = [v for v in vertices if v not in inside]
            coefficient = _tau_exact_or_zero(rho, list(inside)) * _tau_exact_or_zero(rho, outside)
            if coefficient:
                cut = sum((rho[i][j] for i in inside for j in outside), Fraction(0))
                terms.append((coefficient, cut))
    return _tau_exact_or_zero(rho, vertices), terms


def exact_beats_a_n_star(n: int, rho, dps: int = 60):
    """Rigorous comparison of the lattice with Selling parameters ``rho`` (Fractions, >= 0)
    against ``A_n*`` (all parameters 1): is ``U/d^(1-1/(2n))`` strictly smaller? Exact
    rationals for the tree counts and cut weights; interval arithmetic for the square
    roots and the comparison ``(U/U_0)^(2n) < (d/d_0)^(2n-1)``."""
    from fractions import Fraction

    import mpmath as mp

    mp.iv.dps = dps
    ones = [[Fraction(0) if i == j else Fraction(1) for j in range(n + 1)] for i in range(n + 1)]
    d0, terms0 = exact_terms_all(n, ones)
    d1, terms1 = exact_terms_all(n, rho)

    def interval_sum(terms):
        total = mp.iv.mpf(0)
        for coefficient, cut in terms:
            total += (mp.iv.mpf(coefficient.numerator) / coefficient.denominator
                      * mp.iv.sqrt(mp.iv.mpf(cut.numerator) / cut.denominator))
        return total

    ratio = interval_sum(terms1) / interval_sum(terms0)
    left = ratio ** (2 * n)
    right_exact = (d1 / d0) ** (2 * n - 1)
    right = mp.iv.mpf(right_exact.numerator) / right_exact.denominator
    left_low, right_high = mp.mpf(left.a.a), mp.mpf(right.b.a)
    return {"n": n, "left_upper": mp.nstr(left.b.a, 15), "right_lower": mp.nstr(right.a.a, 15),
            "smaller": bool(left.b < right.a),
            "log_ratio": float(mp.log(left_low / right_high) / (2 * n))}


def complete_bipartite(n: int, sizes):
    """Selling parameters 1 across the bipartition ``sizes = (a, b)`` of the ``n + 1``
    vertices and 0 inside the parts, as Fractions."""
    from fractions import Fraction

    a, b = sizes
    assert a + b == n + 1
    return [[Fraction(1) if (i < a) != (j < a) else Fraction(0) for j in range(n + 1)]
            for i in range(n + 1)]


def _bipartite_tau(x: int, y: int) -> int:
    """Spanning-tree count of the complete bipartite graph K_{x,y} (unit weights); a single
    vertex counts 1 and a disconnected graph 0."""
    if x + y == 1:
        return 1
    if x == 0 or y == 0:
        return 0
    return x ** (y - 1) * y ** (x - 1)


def bipartite_terms(a: int, b: int):
    """``d`` and the terms ``(coefficient, cut weight)`` of ``U`` for the lattice whose
    Selling parameters are 1 across a bipartition ``(a, b)`` of the ``a + b`` superbase
    vectors and 0 inside the parts: integers only, summed over the ``(s_a, s_b)`` classes
    of subsets."""
    from math import comb

    terms = []
    for sa in range(a + 1):
        for sb in range(b + 1):
            if (sa == 0 and sb == 0) or (sa == a and sb == b):
                continue
            coefficient = (comb(a, sa) * comb(b, sb) * _bipartite_tau(sa, sb)
                           * _bipartite_tau(a - sa, b - sb))
            if coefficient:
                terms.append((coefficient, sa * (b - sb) + sb * (a - sa)))
    return _bipartite_tau(a, b), terms


def a_star_terms(n: int):
    """The same for ``A_n*`` (every parameter 1): ``tau(K_s) = s^(s-2)`` (1 for ``s = 1``)."""
    from math import comb

    size = n + 1

    def cayley(count: int) -> int:
        return 1 if count == 1 else count ** (count - 2)

    terms = [(comb(size, s) * cayley(s) * cayley(size - s), s * (size - s)) for s in range(1, n + 1)]
    return size ** (n - 1), terms


def bipartite_certificate(a: int, b: int, dps: int = 60):
    """Rigorous comparison of the complete-bipartite lattice ``K_{a,b}`` (dimension
    ``n = a + b - 1``) with ``A_n*``: exact integers, interval arithmetic for the square
    roots, and the test ``(U/U_0)^(2n) < (d/d_0)^(2n-1)``. Also the log of the quotient
    ratio, from the interval midpoints."""
    from fractions import Fraction

    import mpmath as mp

    mp.iv.dps = dps
    n = a + b - 1
    d_k, terms_k = bipartite_terms(a, b)
    d_a, terms_a = a_star_terms(n)

    def interval_sum(terms):
        total = mp.iv.mpf(0)
        for coefficient, cut in terms:
            total += mp.iv.mpf(coefficient) * mp.iv.sqrt(mp.iv.mpf(cut))
        return total

    ratio = interval_sum(terms_k) / interval_sum(terms_a)
    left = ratio ** (2 * n)
    right_exact = Fraction(d_k, d_a) ** (2 * n - 1)
    right = mp.iv.mpf(right_exact.numerator) / right_exact.denominator
    smaller = bool(left.b < right.a)
    log_ratio = (mp.log(mp.mpf(left.a.a) / mp.mpf(right.b.a)) / (2 * n))
    return {"n": n, "a": a, "b": b, "smaller": smaller, "log_ratio": float(log_ratio)}


def _cone_tau(x: int, y: int, w):
    """Weighted spanning-tree count of the graph on one apex and ``x + y`` other vertices
    with weight ``w`` from the apex to everything, 1 across the two groups of sizes ``x``
    and ``y``, and 0 inside the groups (exact Fractions).

    The Laplacian has eigenvalue ``w + y`` on the zero-sum vectors of the first group
    (``x - 1`` of them), ``w + x`` on those of the second (``y - 1``), and on the
    class-constant vectors the quotient matrix, whose two nonzero eigenvalues multiply to
    the sum of its principal 2 x 2 minors; ``tau`` is the product of all nonzero
    eigenvalues over the number of vertices."""
    from fractions import Fraction

    w = Fraction(w)
    count = 1 + x + y
    if x == 0 and y == 0:
        return Fraction(1)
    q = [[w * (x + y), -w * x, -w * y], [-w, w + y, -y], [-w, -x, w + x]]
    if x == 0:
        q = [[w * y, -w * y], [-w, w]]
        return Fraction(1, count) * (w + 0) ** (y - 1) * (w * y + w)
    if y == 0:
        return Fraction(1, count) * (w + 0) ** (x - 1) * (w * x + w)
    minors = sum(q[i][i] * q[j][j] - q[i][j] * q[j][i] for i, j in ((0, 1), (0, 2), (1, 2)))
    return Fraction(1, count) * (w + y) ** (x - 1) * (w + x) ** (y - 1) * minors


def cone_terms(m: int, w):
    """``d`` and the terms of ``U`` for the lattice whose Selling parameters are 1 across a
    bipartition ``(m, m)``, ``w`` from one extra apex vertex to all of them, and 0 inside the
    parts: ``K_{1,m,m}`` with apex weight ``w``, dimension ``2m``. Exact Fractions, summed
    over classes ``(apex in S?, s_a, s_b)``."""
    from fractions import Fraction
    from math import comb

    w = Fraction(w)
    terms = []

    def tau(has_apex, x, y):
        if has_apex:
            return _cone_tau(x, y, w)
        return Fraction(_bipartite_tau(x, y))

    for apex in (0, 1):
        for sa in range(m + 1):
            for sb in range(m + 1):
                if apex == 0 and sa == 0 and sb == 0:
                    continue
                if apex == 1 and sa == m and sb == m:
                    continue
                coefficient = comb(m, sa) * comb(m, sb) * tau(apex, sa, sb) * tau(1 - apex, m - sa, m - sb)
                if coefficient:
                    inside_cut = sa * (m - sb) + sb * (m - sa)
                    apex_cut = w * ((m - sa) + (m - sb)) if apex else w * (sa + sb)
                    terms.append((coefficient, inside_cut + apex_cut))
    return tau(1, m, m), terms


def cone_certificate(m: int, w, dps: int = 60):
    """Rigorous comparison of the cone lattice ``K_{1,m,m}`` with apex weight ``w``
    (dimension ``n = 2m``) against ``A_n*``, as :func:`bipartite_certificate`."""
    from fractions import Fraction

    import mpmath as mp

    mp.iv.dps = dps
    n = 2 * m
    d_k, terms_k = cone_terms(m, Fraction(w))
    d_a, terms_a = a_star_terms(n)

    def interval_sum(terms):
        total = mp.iv.mpf(0)
        for coefficient, cut in terms:
            coefficient, cut = Fraction(coefficient), Fraction(cut)
            total += (mp.iv.mpf(coefficient.numerator) / coefficient.denominator
                      * mp.iv.sqrt(mp.iv.mpf(cut.numerator) / cut.denominator))
        return total

    ratio = interval_sum(terms_k) / interval_sum(terms_a)
    left = ratio ** (2 * n)
    right_exact = (Fraction(d_k) / d_a) ** (2 * n - 1)
    right = mp.iv.mpf(right_exact.numerator) / right_exact.denominator
    log_ratio = mp.log(mp.mpf(left.a.a) / mp.mpf(right.b.a)) / (2 * n)
    return {"n": n, "m": m, "w": str(Fraction(w)), "smaller": bool(left.b < right.a),
            "log_ratio": float(log_ratio)}


def search(n: int, starts: int, seed: int = 0, spreads=(0.15, 0.5, 1.0, 2.0), maxiter=400):
    """Random-start L-BFGS from log-parameters ``N(0, spread^2)``. Returns one record per
    start: final ``f - f(A_n*)``, the spread of the final log-weights, and iterations."""
    from scipy.optimize import minimize

    problem = FirstKindQuotient(n)
    rng = np.random.default_rng(seed)
    records = []
    for index in range(starts):
        spread = spreads[index % len(spreads)]
        x0 = rng.normal(scale=spread, size=problem.m)
        result = minimize(problem.safe, x0, jac=True, method="L-BFGS-B",
                          bounds=[(-8.0, 8.0)] * problem.m,
                          options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10})
        x = result.x - result.x.mean()
        records.append({"start_spread": spread, "relative": float(result.fun - problem.baseline),
                        "weight_spread": float(x.std()), "iterations": int(result.nit),
                        "converged": bool(result.success)})
    return records


def certificate_rows(top: int = 200):
    """One exact verdict per dimension: odd ``n = 2m - 1`` uses ``K_{m,m}``; even ``n = 2m``
    uses the cone ``K_{1,m,m}`` with apex weight 1/2, 21/40 or 11/20 (the best of the three)
    and also ``K_{m,m+1}``. Only the best family per dimension is recorded."""
    from fractions import Fraction

    rows = []
    for n in range(3, top + 1):
        if n % 2 == 1:
            m = (n + 1) // 2
            result = bipartite_certificate(m, m)
            rows.append({"n": n, "family": f"K({m},{m})", "w": None,
                         "smaller": result["smaller"], "log_ratio": result["log_ratio"]})
            continue
        m = n // 2
        candidates = [("cone", w, cone_certificate(m, Fraction(w)))
                      for w in ("1/2", "21/40", "11/20")]
        bip = bipartite_certificate(m, m + 1)
        best = min(candidates, key=lambda item: item[2]["log_ratio"])
        if bip["log_ratio"] < best[2]["log_ratio"]:
            rows.append({"n": n, "family": f"K({m},{m + 1})", "w": None,
                         "smaller": bip["smaller"], "log_ratio": bip["log_ratio"]})
        else:
            rows.append({"n": n, "family": f"K(1,{m},{m}), apex weight {best[1]}", "w": best[1],
                         "smaller": best[2]["smaller"], "log_ratio": best[2]["log_ratio"]})
    return rows


def figure(rows, path):
    """Two panels on their own axes: the first dimensions, and all certified dimensions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted, "axes.edgecolor": grid})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), facecolor=surface)
    odd = [r for r in rows if r["n"] % 2 == 1]
    even = [r for r in rows if r["n"] % 2 == 0]
    for ax, top, title in ((axes[0], 40, "The first dimensions"), (axes[1], 200, "All certified dimensions")):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.7)
        ax.axhline(0, color=muted, linewidth=0.9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", color=ink, fontsize=10, fontweight="bold")
        ax.set_xlabel("dimension n")
        ax.set_ylabel("log(Iso of the lattice / Iso of A\u2099*), \u00d710\u207b\u00b3")
        for group, colour, label in ((odd, blue, "odd n: K(m, m)"), (even, orange, "even n: best of cone K(1, m, m) and K(m, m+1)")):
            xs = [r["n"] for r in group if r["n"] <= top]
            ys = [1e3 * r["log_ratio"] for r in group if r["n"] <= top]
            ax.plot(xs, ys, color=colour, linewidth=2, marker="o" if top == 40 else None,
                    markersize=4.5, markeredgecolor=surface, markeredgewidth=0.8, label=label)
        ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    axes[0].axvline(6.5, color=muted, linewidth=0.9, linestyle=(0, (2, 3)))
    axes[0].text(7.4, 3.0, "n = 7: first lattice\nbelow A\u2087*", color=ink, fontsize=8.5, va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    import json
    import time

    from qres.bench import RESULTS_DIR

    figure_path = RESULTS_DIR / "figures" / "exp048_first_kind_beats_a_n_star.png"
    output = RESULTS_DIR / "exp048_first_kind_global_search.json"
    if argv and argv[0] == "--figure-only":
        print(figure(json.loads(output.read_text())["certificates"], figure_path))
        return 0
    starts = {4: 200, 5: 200, 6: 200, 7: 80, 8: 40, 9: 40}
    searches = {}
    for n, count in starts.items():
        begin = time.time()
        records = search(n, starts=count, seed=n, spreads=(0.1, 0.3, 0.7, 1.5, 2.5))
        rel = np.array([r["relative"] for r in records])
        values = sorted({round(float(x), 6) for x in rel})
        print(f"n={n}: {count} starts; at A_n*: {int((np.abs(rel) < 1e-9).sum())}; lowest {rel.min():+.3e}; "
              f"distinct final values {values[:5]} ({time.time() - begin:.0f} s)", flush=True)
        searches[n] = {"starts": count, "relative": [float(x) for x in rel]}
    begin = time.time()
    rows = certificate_rows()
    beaten = [r["n"] for r in rows if r["smaller"]]
    print(f"exact certificates: A_n* beaten at {len(beaten)} of {len(rows)} dimensions "
          f"(n = {beaten[0]}..{beaten[-1]}); not beaten at {[r['n'] for r in rows if not r['smaller']]} "
          f"({time.time() - begin:.0f} s)")
    output.write_text(json.dumps({"searches": searches, "certificates": rows}, indent=1))
    print("saved", output, "and", figure(rows, figure_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
