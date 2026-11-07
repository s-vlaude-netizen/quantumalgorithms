"""Experiment 045 -- the four-dimensional case of the parallelohedron problem.

Cesaroni & Novaga (arXiv:2609.02384, the second proof of Bezdek's conjecture)
close with open problems. Two of them are four-dimensional:

* **Open problem 1.** Determine ``gamma_4 = inf_L Iso_4(Vor(L))`` over lattices
  in R^4. Is the regular 24-cell, the Voronoi cell of D4 (``2^(11/4)``), the
  unique minimiser?
* **Open problem 2.** Among lattices of Voronoi's first kind -- whose Voronoi
  cells are the K_{n+1}-graphical zonotopes -- for which n is A_n* optimal?
  (exp044 bears on n = 4: the Hales-Song chain bound is lowest at A_4* on all
  21 strata, with a positive-definite Hessian there.)

This module measures, without proving:

1. **Exact values** of the special cells (SymPy):
   * the 24-cell;
   * the A_4* permutohedron;
   * the zonotope of the cographic matroid of K_{3,3} in its symmetric
     embedding -- the one zonotopal four-dimensional parallelohedron outside
     the graphical family, and so outside open problem 2.
2. **A random-start search over all four-dimensional lattices** (nine shape
   parameters, LLL-reduced, Voronoi cell by half-space intersection).
3. **The same over all four-dimensional parallelohedra** -- affine images
   ``A Vor(L)``, which by Voronoi's conjecture (proved through dimension five,
   Garber 2025) is every parallelohedron. For a fixed polytope, area under
   ``A in SL(4)`` is ``sum_k a_k |A^-T u_k|``, minimised over ``A`` from the
   facet data alone.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qres.bench import RESULTS_DIR

# ---------------------------------------------------------------- exact values

def special_cells():
    """Exact isoperimetric quotients of the three special four-dimensional
    cells, from Gram determinants of their generators (zonotopes) or their
    facets (24-cell)."""
    import sympy as sp

    def zonotope_quotient(vectors, n=4):
        def gram_det(vs):
            a = sp.Matrix.hstack(*vs)
            return sp.simplify((a.T * a).det())

        volume = sum(sp.sqrt(gram_det(list(b))) for b in itertools.combinations(vectors, n))
        area = 2 * sum(sp.sqrt(gram_det(list(t))) for t in itertools.combinations(vectors, n - 1))
        return sp.nsimplify(sp.simplify(volume)), sp.simplify(area)

    def projection(rows, columns):
        incidence = sp.Matrix(rows)
        cut = incidence.T * (incidence * incidence.T).pinv() * incidence
        return cut, sp.eye(columns) - cut

    # K5: the cut-space projection of the edge basis gives Vor(A_4*)
    k5 = list(itertools.combinations(range(5), 2))
    incidence5 = [[(1 if v == i else -1 if v == j else 0) for i, j in k5] for v in range(5)]
    cut5, _ = projection(incidence5, 10)
    volume5, area5 = zonotope_quotient([cut5[:, k] for k in range(10)])
    # K_{3,3}: the cycle-space projection gives the cographic zonotope
    k33 = [(i, j) for i in range(3) for j in range(3)]
    incidence33 = ([[1 if i == a else 0 for a, _ in k33] for i in range(3)]
                   + [[-1 if j == b else 0 for _, b in k33] for j in range(3)])
    _, cycle33 = projection(incidence33, 9)
    volume33, area33 = zonotope_quotient([cycle33[:, k] for k in range(9)])
    out = {
        "24-cell (Vor D4)": sp.Integer(2) ** sp.Rational(11, 4),
        "cographic K33 zonotope": sp.simplify(area33 / volume33 ** sp.Rational(3, 4)),
        "permutohedron (Vor A4*)": sp.simplify(area5 / volume5 ** sp.Rational(3, 4)),
        "tesseract (Z^4)": sp.Integer(8),
    }
    return out


# ----------------------------------------------------------------- lattices

def lll(basis, delta=0.75):
    """LLL-reduce the rows of ``basis`` (small dimension, floating point)."""
    b = np.array(basis, float)
    n = len(b)

    def gram_schmidt(b):
        q = np.zeros_like(b)
        mu = np.zeros((n, n))
        for i in range(n):
            q[i] = b[i]
            for j in range(i):
                mu[i, j] = b[i] @ q[j] / (q[j] @ q[j])
                q[i] = q[i] - mu[i, j] * q[j]
        return q, mu

    q, mu = gram_schmidt(b)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            r = round(mu[k, j])
            if r:
                b[k] = b[k] - r * b[j]
                q, mu = gram_schmidt(b)
        if q[k] @ q[k] >= (delta - mu[k, k - 1] ** 2) * (q[k - 1] @ q[k - 1]):
            k += 1
        else:
            b[[k, k - 1]] = b[[k - 1, k]]
            q, mu = gram_schmidt(b)
            k = max(k - 1, 1)
    return b


def voronoi_cell(basis, reach=2):
    """Voronoi cell of the lattice spanned by the rows of ``basis``: its
    isoperimetric quotient, volume, and facets as (area, unit normal)."""
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    b = lll(basis)
    n = len(b)
    vectors = [np.array(c) @ b for c in itertools.product(range(-reach, reach + 1), repeat=n)
               if any(c)]
    halfspaces = np.array([np.append(v, -(v @ v) / 2) for v in vectors])
    vertices = HalfspaceIntersection(halfspaces, np.zeros(n)).intersections
    hull = ConvexHull(vertices)
    facets = {}
    for simplex, equation in zip(hull.simplices, hull.equations):
        key = tuple(np.round(equation[:n], 7))
        points = vertices[simplex]
        edges = points[1:] - points[0]
        measure = math.sqrt(abs(np.linalg.det(edges @ edges.T))) / math.factorial(n - 1)
        facets.setdefault(key, [0.0, equation[:n]])[0] += measure
    data = [(area, normal / np.linalg.norm(normal)) for area, normal in facets.values()]
    return hull.area / hull.volume ** ((n - 1) / n), hull.volume, data


def affine_minimum(facets, volume):
    """Least isoperimetric quotient over the SL(n) images of a polytope with
    the given facets: ``min_S sum_k a_k sqrt(u_k^T S u_k)`` over symmetric
    positive definite ``S`` of determinant one (``S = A^-T A^-1``)."""
    from scipy.linalg import expm
    from scipy.optimize import minimize

    areas = np.array([a for a, _ in facets])
    normals = np.array([u for _, u in facets])
    n = normals.shape[1]
    upper = np.triu_indices(n)

    def area(x):
        sym = np.zeros((n, n))
        sym[upper] = x
        sym = sym + sym.T - np.diag(np.diag(sym))
        s = expm(sym - np.trace(sym) / n * np.eye(n))
        return float(areas @ np.sqrt(np.einsum("ki,ij,kj->k", normals, s, normals)))

    result = minimize(area, np.zeros(len(upper[0])), method="BFGS", options={"gtol": 1e-10})
    return result.fun / volume ** ((n - 1) / n)


def random_basis(rng, n=4):
    lower = np.tril(rng.normal(scale=0.6, size=(n, n)), -1) + np.diag(np.exp(rng.normal(scale=0.3, size=n)))
    return lower / abs(np.linalg.det(lower)) ** (1 / n)


def search(starts=20, seed=3, affine=False, max_evaluations=3000):
    """Minimise the quotient over four-dimensional lattices (or, with
    ``affine``, over their SL(4) images) from random starts."""
    from scipy.optimize import minimize

    def unpack(x):
        lower = np.zeros((4, 4))
        lower[np.tril_indices(4)] = x
        lower[np.diag_indices(4)] = np.exp(np.diag(lower))
        return lower

    def objective(x):
        try:
            quotient, volume, facets = voronoi_cell(unpack(x))
        except (ValueError, np.linalg.LinAlgError, RuntimeError):   # degenerate input
            return 1e3
        return affine_minimum(facets, volume) if affine else quotient

    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(starts):
        start = time.time()
        basis = random_basis(rng)
        x0 = basis[np.tril_indices(4)].copy()
        diagonal = np.cumsum([0, 1, 2, 3]) + np.arange(4)       # positions of the diagonal
        x0[diagonal] = np.log(np.abs(basis[np.diag_indices(4)]))
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxfev": max_evaluations, "xatol": 1e-8, "fatol": 1e-11,
                                   "adaptive": True})
        facets = voronoi_cell(unpack(result.x))[2]
        largest = max(a for a, _ in facets)
        facets = [f for f in facets if f[0] > 1e-6 * largest]   # drop facets shrunk to nothing
        runs.append({"quotient": float(result.fun), "facets": len(facets),
                     "seconds": round(time.time() - start, 1)})
        print(f"  {'affine' if affine else 'lattice'} start: {result.fun:.9f} "
              f"({len(facets)} facets, {time.time() - start:.0f} s)", flush=True)
    return runs


def root_lattice(n):
    """Basis of A_n, in orthonormal coordinates of the sum-zero hyperplane."""
    roots = np.eye(n + 1)[:n] - np.eye(n + 1)[1:]
    plane = np.linalg.qr(np.vstack([np.ones(n + 1), np.eye(n + 1)[:n]]).T)[0][:, 1:]
    return roots @ plane


def named_lattices():
    """Iso_4 of the Voronoi cells of other natural four-dimensional lattices,
    and of their best affine images (numerical), for context."""
    def blocks(*parts):
        size = sum(len(b) for b in parts)
        out, k = np.zeros((size, size)), 0
        for b in parts:
            out[k:k + len(b), k:k + len(b)] = b
            k += len(b)
        return out

    a2, a3 = root_lattice(2), root_lattice(3)
    cases = {"A4 (root lattice)": root_lattice(4),
             "A3* + Z (truncated-octahedron prism)": blocks(np.linalg.inv(a3).T, np.eye(1)),
             "A3 + Z (rhombic-dodecahedron prism)": blocks(a3, np.eye(1)),
             "A2 + A2 (hexagon x hexagon)": blocks(a2, a2),
             "A2 + Z + Z": blocks(a2, np.eye(2))}
    out = {}
    for name, basis in cases.items():
        quotient, volume, facets = voronoi_cell(basis)
        out[name] = {"voronoi": quotient, "best_affine_image": affine_minimum(facets, volume)}
    return out


def cached_search(name, **kwargs):
    """``search`` with its runs kept in results/raw, so an interrupted run
    resumes instead of starting over."""
    raw = RESULTS_DIR / "raw" / "exp045"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    runs = search(**kwargs)
    path.write_text(json.dumps(runs, indent=1))
    return runs


def main() -> int:
    exact = special_cells()
    for name, value in exact.items():
        print(f"{name:28s} {float(value):.12f}   {value}")
    named = named_lattices()
    for name, value in named.items():
        print(f"{name:40s} {value['voronoi']:.9f} (best affine image {value['best_affine_image']:.9f})")
    lattice_runs = cached_search("lattices", starts=16, affine=False)
    affine_runs = cached_search("parallelohedra", starts=8, affine=True, seed=5)
    out = {"exact": {k: [str(v), float(v)] for k, v in exact.items()}, "named_lattices": named,
           "lattice_search": lattice_runs, "affine_search": affine_runs,
           "lowest_lattice": min(r["quotient"] for r in lattice_runs),
           "lowest_affine": min(r["quotient"] for r in affine_runs)}
    print(f"lowest over lattices {out['lowest_lattice']:.9f}, over parallelohedra "
          f"{out['lowest_affine']:.9f}; 24-cell {float(exact['24-cell (Vor D4)']):.9f}")
    path = RESULTS_DIR / "exp045_four_dimensional_cells.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
