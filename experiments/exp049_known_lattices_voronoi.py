"""Experiment 049 -- isoperimetric quotients of the Voronoi cells of the classical lattices.

Result 100 found first-kind lattices below A_n* from n = 7. How do they compare with the
classical lattices of the second kind, in the dimensions where those are the standard
candidates (E7, E7*, E8, D_n, D_n*)? Each Voronoi cell is computed numerically from the
lattice itself: half-space intersection over all lattice vectors up to a norm bound, then
checked by its volume, which must equal the covolume.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def e8_simple_roots():
    """Bourbaki simple roots of E8 (rows), a basis of the E8 lattice in R^8."""
    half = 0.5
    return np.array([
        [half, -half, -half, -half, -half, -half, -half, half],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, -1, 1, 0],
    ], dtype=float)


def dual_basis(basis):
    """Basis of the dual lattice, inside the span of the given rows."""
    gram = basis @ basis.T
    return np.linalg.inv(gram) @ basis


def lattices():
    e8 = e8_simple_roots()
    e7 = e8[:7]
    out = {"E8": e8, "E7": e7, "E7*": dual_basis(e7)}
    for n in (3, 4, 7, 8):
        out[f"D{n}"] = d_n(n)
        out[f"D{n}*"] = dual_basis(d_n(n))
        out[f"A{n}*"] = dual_basis(a_n(n))
        out[f"Z{n}"] = np.eye(n)
    return out


def d_n(n):
    rows = [[1.0 if j == i else (-1.0 if j == i + 1 else 0.0) for j in range(n)] for i in range(n - 1)]
    rows.append([0.0] * (n - 2) + [1.0, 1.0])
    return np.array(rows)


def a_n(n):
    """A_n in R^n: vectors with integer coordinates summing to 0 (basis e_i - e_{i+1}, embedded
    in the first n coordinates of R^(n+1) and then rotated into R^n by a QR factor)."""
    rows = np.array([[1.0 if j == i else (-1.0 if j == i + 1 else 0.0) for j in range(n + 1)]
                     for i in range(n)])
    q, _ = np.linalg.qr(np.vstack([np.ones(n + 1) / np.sqrt(n + 1), np.eye(n + 1)[:n]]).T)
    return rows @ q[:, 1:]


def voronoi_quotient(basis, norm_factor=4.0, reach=2):
    """Isoperimetric quotient of the Voronoi cell, and the check volume / covolume.

    Candidate vectors are all lattice vectors with coefficients in ``[-reach, reach]``
    (in the given basis) and norm at most ``norm_factor`` times the squared minimum."""
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    from experiments.exp045_four_dimensional_cells import lll

    basis = lll(basis)
    rank = basis.shape[0]
    coefficients = np.array(list(itertools.product(range(-reach, reach + 1), repeat=rank)), dtype=float)
    coefficients = coefficients[np.any(coefficients != 0, axis=1)]
    vectors = coefficients @ basis
    norms = (vectors ** 2).sum(axis=1)
    keep = norms <= norm_factor * norms.min() + 1e-9
    vectors = vectors[keep]
    # express the cell in orthonormal coordinates of the span
    _, _, vt = np.linalg.svd(basis, full_matrices=False)
    coords = vectors @ vt.T
    halfspaces = np.hstack([coords, -(coords ** 2).sum(axis=1, keepdims=True) / 2])
    hull = ConvexHull(HalfspaceIntersection(halfspaces, np.zeros(rank), qhull_options="Qx Q12").intersections)
    covolume = np.sqrt(np.linalg.det(basis @ basis.T))
    return hull.area / hull.volume ** ((rank - 1) / rank), hull.volume / covolume, len(vectors)


def main(argv=None) -> int:
    import time

    names = argv or ["E7", "E7*", "D7", "D7*", "A7*", "Z7"]           # E8 and the 8-D cells are slow in qhull
    table = lattices()
    for name in names:
        start = time.time()
        quotient, volume_ratio, count = voronoi_quotient(table[name], norm_factor=2.0 if name == "E8" else 4.0)
        print(f"{name:4s} dim {table[name].shape[0]}: Iso = {quotient:.6f} (volume / covolume {volume_ratio:.6f}, "
              f"{count} candidate vectors, {time.time() - start:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
