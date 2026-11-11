"""Experiment 047 -- is A_n* a local minimum of the isoperimetric quotient, for every n?

Cesaroni & Novaga (arXiv:2609.02384), Open problem 2: among lattices of Voronoi's
first kind, for which dimensions n is ``A_n*`` optimal? Result 98 found, for
n = 4, that the Hales-Song bound is lowest at ``A_n*``. This module asks the local
question in every dimension, with the quotient itself.

**Set-up.** ``f_0 .. f_n`` are the unit vectors of the regular simplex
(``f_i . f_j = -1/n``); ``A_n*`` is the lattice they span. Every lattice is a linear
image ``A A_n*`` with ``A`` symmetric positive definite (rotations change nothing).
Near ``A_n*`` all the Selling parameters ``rho_ij = -(A f_i) . (A f_j)`` are
positive, so every nearby lattice is of the first kind, its Voronoi-relevant vectors
are the ``2^(n+1) - 2`` sums ``sum_{i in S} A f_i``, and its Voronoi cell is the
graphical zonotope with edge weights ``beta_ij = rho_ij`` in the frame ``A``:

    area   =  sum over bipartitions (S, S^c) of  tau(S) tau(S^c) |A n_S|,   n_S = sum_{i in S} f_i
    volume =  tau({0..n}),

with ``tau`` the weighted spanning-tree count (Kirchhoff). The quotient is
``area / volume^((n-1)/n)`` up to a constant that is fixed by ``A_n*`` itself.
Everything is a smooth function of ``(log beta, A)`` near equal weights, so the
second-order test is a finite-difference Hessian.

Two classes are compared:

* **lattices**: ``beta`` is tied to ``A`` (``beta_ij = -n f_i^T A^2 f_j``), so the
  parameter is the symmetric traceless ``T`` with ``A = exp(T)``;
* **K_{n+1}-graphical zonotopes and all their affine images**: ``beta`` and ``A``
  free. The Hales-Song chain bounds this class from below.

By symmetry (``S_{n+1}`` acts on both) the Hessian splits into the two non-trivial
irreducible pieces of the edge representation: the standard one ``V`` (dimension n)
and ``(n-1, 2)`` (dimension ``(n+1)(n-2)/2``). Each is one direction in the lattice
class and a 2 x 2 block in the affine class.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def frame(n: int) -> np.ndarray:
    """The ``n + 1`` unit vectors of the regular simplex in R^n (rows)."""
    centred = np.eye(n + 1) - 1.0 / (n + 1)
    basis = np.linalg.qr(np.vstack([np.ones(n + 1), np.eye(n + 1)[:-1]]).T)[0][:, 1:]
    f = centred @ basis
    return f / np.linalg.norm(f, axis=1, keepdims=True)


class Cell:
    """Quotient of the graphical zonotope with edge weights ``exp(x)`` in the frame
    ``exp(T)`` for symmetric traceless ``T``; precomputes the subset structure."""

    def __init__(self, n: int):
        self.n = n
        self.f = frame(n)
        self.pairs = list(itertools.combinations(range(n + 1), 2))
        self.m = len(self.pairs)
        self.pair_index = np.array(self.pairs)
        vertices = range(n + 1)
        self.batches = []
        for k in range(1, n + 1):
            inside = np.array(list(itertools.combinations(vertices, k)))
            outside = np.array([[v for v in vertices if v not in set(s)] for s in inside])
            normals = self.f[inside].sum(axis=1)                       # (batch, n)
            self.batches.append((inside, outside, normals))
        self.base = None
        self.base = self.quotient(np.zeros(self.m), np.zeros((n, n)))

    def _tau(self, w, subsets):
        k = subsets.shape[1]
        if k == 1:
            return np.ones(len(subsets))
        sub = w[subsets[:, :, None], subsets[:, None, :]]              # (batch, k, k)
        lap = -sub
        idx = np.arange(k)
        lap[:, idx, idx] = sub.sum(axis=2)
        return np.linalg.det(lap[:, 1:, 1:])

    def quotient(self, x, t, calibrated: bool = True):
        """Isoperimetric quotient (calibrated to ``A_n*`` when ``calibrated``)."""
        n = self.n
        beta = np.exp(x)
        w = np.zeros((n + 1, n + 1))
        w[self.pair_index[:, 0], self.pair_index[:, 1]] = beta
        w = w + w.T
        a = expm(t)
        total = 0.0
        for inside, outside, normals in self.batches:
            gamma = self._tau(w, inside) * self._tau(w, outside)
            total += float((gamma * np.linalg.norm(normals @ a.T, axis=1)).sum())
        volume = self._tau(w, np.arange(n + 1)[None, :])[0]
        value = total / volume ** ((n - 1) / n)
        if self.base is None or not calibrated:
            return value
        return value * self.exact_base / self.base

    @property
    def exact_base(self):
        from experiments.exp044_hales_song_certificate import permutohedron_quotient

        return float(permutohedron_quotient(self.n))

    # ---------------------------------------------------------- parametrisations
    def lattice_weights(self, t):
        """Log edge weights of the Voronoi cell of ``exp(T) A_n*``: ``beta_ij =
        -n f_i^T exp(2T) f_j``. NaN if a Selling parameter is not positive."""
        p = expm(2 * t)
        gram = self.f @ p @ self.f.T
        beta = -self.n * gram[self.pair_index[:, 0], self.pair_index[:, 1]]
        return np.log(beta) if (beta > 0).all() else np.full(self.m, np.nan)

    def lattice_quotient(self, t):
        return self.quotient(self.lattice_weights(t), t)

    # ------------------------------------------------------------------ directions
    def edge_vectors(self):
        """One vector of each non-trivial irreducible piece of the edge space."""
        n, pairs = self.n, self.pairs
        index = {p: k for k, p in enumerate(pairs)}
        standard = np.zeros(self.m)           # x_ij = a_i + a_j with a = e_0 - e_1
        for (i, j), k in index.items():
            standard[k] = ((i == 0) - (i == 1)) + ((j == 0) - (j == 1))
        rectangle = np.zeros(self.m)          # +e01 - e02 - e13 + e23: a 4-cycle
        for pair, sign in (((0, 1), 1), ((0, 2), -1), ((1, 3), -1), ((2, 3), 1)):
            rectangle[index[pair]] = sign
        assert n >= 3
        return {"standard": standard, "(n-1,2)": rectangle}

    def matrix_of(self, w):
        """The symmetric traceless matrix ``sum_ij w_ij (f_i f_j^T + f_j f_i^T)``."""
        t = np.zeros((self.n, self.n))
        for value, (i, j) in zip(w, self.pairs):
            if value:
                t += value * (np.outer(self.f[i], self.f[j]) + np.outer(self.f[j], self.f[i]))
        return t - np.trace(t) / self.n * np.eye(self.n)


def central_second_derivative(function, direction, h):
    """``d^2/ds^2 function(s * direction)`` at ``s = 0`` by central differences."""
    return (function(h * direction) - 2 * function(0 * direction) + function(-h * direction)) / h ** 2


def block_curvatures(n: int, h: float = 1e-3):
    """Curvature of the quotient at ``A_n*`` along each irreducible piece of the
    edge representation. In the lattice class that is one number per piece; in the
    affine class (weights and frame free) a 2 x 2 block, reported with its Schur
    complement, which is the curvature once the frame has relaxed. All directions
    are unit vectors, so the numbers are eigenvalues up to one common factor."""
    cell = Cell(n)
    out = {}
    for name, w in cell.edge_vectors().items():
        t_dir = cell.matrix_of(w)
        t_dir /= np.linalg.norm(t_dir)
        w = w / np.linalg.norm(w)

        def mixed(sx, st, w=w, t_dir=t_dir):
            return cell.quotient(sx * w, st * t_dir)

        lattice = central_second_derivative(cell.lattice_quotient, t_dir, h)
        h_xx = (mixed(h, 0) - 2 * mixed(0, 0) + mixed(-h, 0)) / h ** 2
        h_tt = (mixed(0, h) - 2 * mixed(0, 0) + mixed(0, -h)) / h ** 2
        h_xt = (mixed(h, h) - mixed(h, -h) - mixed(-h, h) + mixed(-h, -h)) / (4 * h * h)
        out[name] = {"lattice": float(lattice), "xx": float(h_xx), "TT": float(h_tt),
                     "xT": float(h_xt), "schur": float(h_xx - h_xt ** 2 / h_tt)}
    return out


# ------------------------------------------------- symmetry-reduced curvature, any n

def special_directions(n: int):
    """Edge-weight matrices ``W`` (symmetric, zero diagonal) of one vector from each
    non-trivial irreducible piece, on the first few vertices only."""
    standard = np.zeros((n + 1, n + 1))
    a = np.zeros(n + 1)
    a[0], a[1] = 1.0, -1.0
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            standard[i, j] = standard[j, i] = a[i] + a[j]
    rectangle = np.zeros((n + 1, n + 1))
    for (i, j), sign in (((0, 1), 1), ((0, 2), -1), ((1, 3), -1), ((2, 3), 1)):
        rectangle[i, j] = rectangle[j, i] = sign
    return {"standard": (standard, 2), "(n-1,2)": (rectangle, 4)}


def permutohedron_iso(n: int, mpf=float):
    """Isoperimetric quotient of the regular permutohedron (Voronoi cell of A_n*), the
    closed form of ``exp044.permutohedron_quotient`` in the given number type."""
    from math import comb

    half_area = 0
    for s in range(1, n + 1):
        half_area += (mpf(comb(n + 1, s)) / 2 * mpf(s) ** (s - 2) * mpf(n + 1 - s) ** (n - 1 - s)
                      * mpf(n + 1) ** (-mpf(n - 1) / (2 * n)) * (mpf(s) * (n + 1 - s)) ** 0.5
                      if mpf is float else
                      mpf(comb(n + 1, s)) / 2 * mpf(s) ** (s - 2) * mpf(n + 1 - s) ** (n - 1 - s)
                      * mpf(n + 1) ** (-mpf(n - 1) / (2 * n)) * mpf(s * (n + 1 - s)) ** (mpf(1) / 2))
    return 2 * half_area / mpf(n + 1) ** (mpf((n - 1) ** 2) / n)


class ReducedCell:
    """The quotient near ``A_n*`` along one irreducible piece, by sums over classes of
    vertex subsets (a subset matters only through its pattern on the few special
    vertices of the direction and the number ``k`` of other vertices).

    ``quotient(eps_x, eps_t, lattice)``: the weights are ``exp(eps_x * x_ij)`` with
    ``x`` the unit edge vector, and the frame is ``exp(eps_t * T)`` with ``T`` its unit
    image in the symmetric traceless matrices. With ``lattice`` the weights are instead
    those of the Voronoi cell of ``exp(eps_t T) A_n*``. Exact matrix exponentials; with
    ``dps`` everything runs in mpmath at that many digits."""

    def __init__(self, n: int, kind: str, dps: int | None = None):
        self.n, self.dps = n, dps
        w_raw, self.n_special = special_directions(n)[kind]
        size = n + 1
        self.size, self.generic = size, size - self.n_special
        if dps:
            import mpmath as mp

            mp.mp.dps = dps
            self.mp = mp
            self.mpf, self.sqrt = mp.mpf, mp.sqrt
            gf = mp.matrix([[mp.mpf(1) if i == j else -mp.mpf(1) / n for j in range(size)]
                            for i in range(size)])
            w = mp.matrix([[mp.mpf(int(w_raw[i, j])) for j in range(size)] for i in range(size)])
            square = w * gf * w * gf
            self.t_edges = w / sum(square[i, i] for i in range(size)) ** 0.5     # unit T
            self.x_edges = w / mp.sqrt(sum(w[i, j] ** 2 for i in range(size) for j in range(i + 1, size)))
        else:
            self.mpf, self.sqrt = float, math.sqrt
            gf = np.full((size, size), -1.0 / n) + (1.0 + 1.0 / n) * np.eye(size)
            self.t_edges = w_raw / math.sqrt(np.trace(w_raw @ gf @ w_raw @ gf))
            self.x_edges = w_raw / math.sqrt((w_raw ** 2).sum() / 2)
        self.gf = gf

    def frame_gram(self, eps_t):
        """``F exp(2 eps T) F^T = exp(2 eps Gf W) Gf`` for the frame matrix ``F``."""
        if self.dps:
            return self.mp.expm(2 * eps_t * (self.gf * self.t_edges)) * self.gf
        return expm(2 * eps_t * (self.gf @ self.t_edges)) @ self.gf

    def quotient(self, eps_x, eps_t, lattice: bool = False):
        from math import comb

        n, size, dps = self.n, self.size, self.dps
        m_eps = self.frame_gram(eps_t)
        if lattice:
            weights = -n * m_eps
        elif dps:
            weights = self.mp.matrix(size, size)
            for i in range(size):
                for j in range(size):
                    weights[i, j] = self.mp.exp(eps_x * self.x_edges[i, j]) if i != j else 0
        else:
            weights = np.exp(eps_x * self.x_edges)
            np.fill_diagonal(weights, 0)
        total = 0
        top = 2 ** self.n_special - 1
        for pattern in range(top + 1):
            inside_special = [v for v in range(self.n_special) if pattern >> v & 1]
            outside_special = [v for v in range(self.n_special) if not pattern >> v & 1]
            for k in range(self.generic + 1):
                if (pattern == 0 and k == 0) or (pattern == top and k == self.generic):
                    continue
                inside = inside_special + list(range(self.n_special, self.n_special + k))
                outside = outside_special + list(range(self.n_special + k, size))
                chosen = set(inside)
                indicator = [1 if v in chosen else 0 for v in range(size)]
                if dps:
                    vec = self.mp.matrix(indicator)
                    norm2 = (vec.T * m_eps * vec)[0]
                else:
                    vec = np.array(indicator, dtype=float)
                    norm2 = float(vec @ m_eps @ vec)
                total += (comb(self.generic, k) * _tau_matrix(weights, inside, dps)
                          * _tau_matrix(weights, outside, dps) * self.sqrt(norm2))
        return total / _tau_matrix(weights, list(range(size)), dps) ** (self.mpf(n - 1) / n)

    def calibration(self):
        return permutohedron_iso(self.n, self.mpf) / self.quotient(0, 0)

    def curvatures(self, h):
        """Second derivatives at ``A_n*``: the lattice class, and the 2 x 2 block of the
        affine class (``xx``, ``TT``, ``xT``) with its Schur complement."""
        mpf = self.mpf
        h = mpf(h)
        zero = mpf(0)
        q0 = self.quotient(zero, zero)
        scale = permutohedron_iso(self.n, mpf) / q0
        lattice = (self.quotient(zero, h, True) - 2 * q0 + self.quotient(zero, -h, True)) / h ** 2
        xx = (self.quotient(h, zero) - 2 * q0 + self.quotient(-h, zero)) / h ** 2
        tt = (self.quotient(zero, h) - 2 * q0 + self.quotient(zero, -h)) / h ** 2
        xt = (self.quotient(h, h) - self.quotient(h, -h) - self.quotient(-h, h)
              + self.quotient(-h, -h)) / (4 * h ** 2)
        xx, tt, xt, lattice = xx * scale, tt * scale, xt * scale, lattice * scale
        return {"lattice": lattice, "xx": xx, "TT": tt, "xT": xt, "schur": xx - xt ** 2 / tt}


def reduced_lattice_curvature(n: int, kind: str, h: float = 1e-3, dps: int | None = None):
    """Curvature of the lattice quotient at ``A_n*`` along a unit vector of the given
    irreducible piece (see :class:`ReducedCell`)."""
    cell = ReducedCell(n, kind, dps)
    mpf = cell.mpf
    h = mpf(h)
    q0 = cell.quotient(mpf(0), mpf(0), True)
    value = (cell.quotient(mpf(0), h, True) - 2 * q0 + cell.quotient(mpf(0), -h, True)) / h ** 2
    return value * permutohedron_iso(n, mpf) / q0


def _tau_matrix(weights, vertices, dps):
    """Weighted spanning-tree count of the complete graph on ``vertices``."""
    k = len(vertices)
    if k == 1:
        return 1
    if dps:
        import mpmath as mp

        sub = mp.matrix(k, k)
        for a, i in enumerate(vertices):
            for b, j in enumerate(vertices):
                if a != b:
                    sub[a, b] = -weights[i, j]
        for a in range(k):
            sub[a, a] = -sum(sub[a, b] for b in range(k) if b != a)
        return mp.det(sub[1:, 1:])
    idx = np.array(vertices)
    sub = -weights[np.ix_(idx, idx)]
    np.fill_diagonal(sub, 0)
    np.fill_diagonal(sub, -sub.sum(axis=1))
    return float(np.linalg.det(sub[1:, 1:]))


# --------------------------------------- Selling-parameter formula and exact certificate

def selling_value(n: int, rho):
    """Isoperimetric quotient of the lattice with Selling parameters ``rho`` (a
    symmetric ``(n+1) x (n+1)`` array, positive off the diagonal), up to a constant
    that depends on n only:

        U(rho) / d(rho)^(1 - 1/(2n)),     U = sum_S tau(S) tau(S^c) sqrt(r(S)),

    summed over all proper nonempty subsets S of the ``n + 1`` superbase vectors, where
    ``tau`` is the weighted spanning-tree count of the complete graph on a subset,
    ``r(S) = sum_{i in S, j not in S} rho_ij`` is the cut weight (which is ``|v_S|^2``
    for the Voronoi-relevant vector ``v_S``), and ``d`` the count for all vertices
    (``= det`` of the Gram matrix, the squared covolume). Full sum over subsets."""
    cell = Cell(n)
    rho = np.array(rho, dtype=float)
    total = 0.0
    for inside, outside, _ in cell.batches:
        gamma = cell._tau(rho, inside) * cell._tau(rho, outside)
        cut = np.array([rho[np.ix_(a, b)].sum() for a, b in zip(inside, outside)])
        total += float((gamma * np.sqrt(cut)).sum())
    return total / cell._tau(rho, np.arange(n + 1)[None, :])[0] ** (1 - 1 / (2 * n))


def _tau_exact(rho, vertices):
    """Weighted spanning-tree count on ``vertices`` in exact rational arithmetic."""
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
        pivot = next(r for r in range(col, size) if mat[r][col] != 0)
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


def rectangle_selling(n: int, delta):
    """Selling parameters ``1 + delta`` on the pairs 01 and 23, ``1 - delta`` on 02 and
    13, and 1 elsewhere: the rectangle direction of the (n-1, 2) piece, exactly linear."""
    from fractions import Fraction

    delta = Fraction(delta)
    rho = [[Fraction(1)] * (n + 1) for _ in range(n + 1)]
    for pair, value in (((0, 1), 1 + delta), ((2, 3), 1 + delta), ((0, 2), 1 - delta),
                        ((1, 3), 1 - delta)):
        rho[pair[0]][pair[1]] = rho[pair[1]][pair[0]] = value
    return rho


def exact_terms(n: int, rho, n_special: int = 4):
    """Exact ``d(rho)`` and the list ``[(coefficient, cut weight)]`` whose sum
    ``sum coefficient * sqrt(cut weight)`` is ``U(rho)``, grouping subsets into classes
    (pattern on the first ``n_special`` vertices, number of the others). Requires ``rho``
    to be invariant under permutations of the vertices from ``n_special`` on."""
    from fractions import Fraction
    from math import comb

    size = n + 1
    generic = size - n_special
    terms = []
    top = 2 ** n_special - 1
    for pattern in range(top + 1):
        inside_special = [v for v in range(n_special) if pattern >> v & 1]
        outside_special = [v for v in range(n_special) if not pattern >> v & 1]
        for k in range(generic + 1):
            if (pattern == 0 and k == 0) or (pattern == top and k == generic):
                continue
            inside = inside_special + list(range(n_special, n_special + k))
            outside = outside_special + list(range(n_special + k, size))
            cut = sum((rho[i][j] for i in inside for j in outside), Fraction(0))
            coefficient = comb(generic, k) * _tau_exact(rho, inside) * _tau_exact(rho, outside)
            terms.append((coefficient, cut))
    return _tau_exact(rho, list(range(size))), terms


def _det_small(matrix):
    """Exact determinant of a small matrix of Fractions."""
    from fractions import Fraction

    size = len(matrix)
    mat = [row[:] for row in matrix]
    det = Fraction(1)
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
            for c in range(col, size):
                mat[r][c] -= factor * mat[col][c]
    return det


RECTANGLE_EXCESS = {(0, 1): 1, (2, 3): 1, (0, 2): -1, (1, 3): -1}      # rho - 1, in units of delta


def exact_terms_fast(n: int, delta):
    """Same as :func:`exact_terms` for the rectangle family, in closed form.

    All weights are 1 except among the four special vertices, so on a vertex set of
    size ``N`` containing ``s`` special ones the Laplacian is ``N I - J + Delta`` with
    ``Delta`` supported on the special block, and ``tau = det(N I + Delta) / N^2 =
    N^(N-s-2) det(N I_s + Delta_s)``. No ``n x n`` determinant is needed, so this
    reaches n in the thousands."""
    from fractions import Fraction
    from math import comb

    delta = Fraction(delta)

    def excess(a, b):
        return RECTANGLE_EXCESS.get((min(a, b), max(a, b)), 0) * delta

    def tau(specials, generic_count):
        count = len(specials) + generic_count
        block = [[Fraction(0)] * len(specials) for _ in specials]
        for x, a in enumerate(specials):
            block[x][x] += count
            for y, b in enumerate(specials):
                if x != y:
                    block[x][y] -= excess(a, b)
                    block[x][x] += excess(a, b)
        return Fraction(count) ** (count - len(specials) - 2) * _det_small(block)

    generic = n - 3
    terms = []
    for pattern in range(1, 16):
        inside = [v for v in range(4) if pattern >> v & 1]
        outside = [v for v in range(4) if not pattern >> v & 1]
        for k in range(generic + 1):
            if pattern == 15 and k == generic:
                continue
            crossing = sum((excess(a, b) for a in inside for b in outside), Fraction(0))
            cut = (len(inside) + k) * (len(outside) + generic - k) + crossing
            terms.append((comb(generic, k) * tau(inside, k) * tau(outside, generic - k), cut))
    for k in range(1, generic + 1):                      # pattern 0: only generic vertices inside
        cut = k * (4 + generic - k)
        terms.append((comb(generic, k) * tau([], k) * tau(list(range(4)), generic - k), cut))
    return tau(list(range(4)), generic), terms


def certificate(n: int, delta, dps: int = 60, fast: bool = True):
    """Rigorous check that the lattice with the rectangle Selling parameters of size
    ``delta`` has a smaller quotient than ``A_n*``.

    Exact rational arithmetic for every spanning-tree count and cut weight; the square
    roots and the final comparison in interval arithmetic. The quotient is
    ``U / d^(1 - 1/(2n))`` up to a constant, so it is smaller exactly when
    ``(U / U_0)^(2n) < (d / d_0)^(2n - 1)``. Returns the interval bounds of both sides
    and whether they are separated."""
    import mpmath as mp

    mp.iv.dps = dps
    if fast:
        d_base, base_terms = exact_terms_fast(n, 0)
        d_pert, pert_terms = exact_terms_fast(n, delta)
    else:
        d_base, base_terms = exact_terms(n, rectangle_selling(n, 0))
        d_pert, pert_terms = exact_terms(n, rectangle_selling(n, delta))

    def interval_sum(terms):
        total = mp.iv.mpf(0)
        for coefficient, cut in terms:
            total += (mp.iv.mpf(coefficient.numerator) / coefficient.denominator
                      * mp.iv.sqrt(mp.iv.mpf(cut.numerator) / cut.denominator))
        return total

    ratio = interval_sum(pert_terms) / interval_sum(base_terms)
    left = ratio ** (2 * n)
    right_exact = (d_pert / d_base) ** (2 * n - 1)
    right = mp.iv.mpf(right_exact.numerator) / right_exact.denominator
    return {"n": n, "delta": str(delta), "left_low": mp.nstr(left.a, 25),
            "left_high": mp.nstr(left.b, 25), "right_low": mp.nstr(right.a, 25),
            "right_high": mp.nstr(right.b, 25), "smaller": bool(left.b < right.a)}


# ------------------------------------------------------------------ the survey

PIECES = ("standard", "(n-1,2)")
SCAN = list(range(3, 41)) + [45, 50, 60, 70, 80, 90, 100]
HIGH_PRECISION = {13: PIECES[1:], 14: PIECES[1:], 15: PIECES, 16: PIECES[1:]}


def curvature_table(ns=SCAN, h=1e-3):
    """Double-precision curvatures at A_n* for every n: per irreducible piece, the
    lattice class and the affine class (Schur complement of the 2 x 2 block)."""
    rows = []
    for n in ns:
        row = {"n": n}
        for kind in PIECES:
            values = ReducedCell(n, kind).curvatures(h)
            row[kind] = {key: float(value) for key, value in values.items()}
        rows.append(row)
    return rows


def high_precision_table(plan=HIGH_PRECISION, dps=40, h="1e-10"):
    """The same near the sign change, in mpmath at ``dps`` digits."""
    import mpmath as mp

    rows = []
    for n, kinds in plan.items():
        row = {"n": n}
        for kind in kinds:
            values = ReducedCell(n, kind, dps).curvatures(mp.mpf(h))
            row[kind] = {key: mp.nstr(value, 18) for key, value in values.items()}
        rows.append(row)
    return rows


def witnesses(plan=((15, (0.02, 0.05)), (16, (0.05,)), (17, (0.05,)))):
    """Explicit lattices ``exp(eps T) A_n*`` whose Voronoi cells have a smaller
    quotient than ``A_n*``: ``eps`` is the Frobenius size of the perturbation ``T``
    along the (n-1,2) direction. The smallest Selling parameter, relative to the mean,
    shows the lattice is of the first kind."""
    out = []
    for n, epsilons in plan:
        cell = ReducedCell(n, PIECES[1])
        scale = cell.calibration()
        base = cell.quotient(0.0, 0.0, True) * scale
        for eps in epsilons:
            iso = cell.quotient(0.0, eps, True) * scale
            weights = -n * cell.frame_gram(eps)
            off_diagonal = weights[~np.eye(n + 1, dtype=bool)]
            out.append({"n": n, "eps": eps, "iso_A_n_star": float(base), "iso": float(iso),
                        "change": float(iso - base),
                        "min_selling_over_mean": float(off_diagonal.min() / off_diagonal.mean())})
    return out


CERTIFIED = list(range(15, 61)) + [70, 80, 100, 150, 200, 300, 500, 1000]


def certificate_table(ns=CERTIFIED, delta="1/2000"):
    """One exact certificate per dimension: the lattice with rectangle Selling
    parameters of size ``delta`` beats ``A_n*``. Compact record per n."""
    from fractions import Fraction

    import mpmath as mp

    rows = []
    for n in ns:
        record = certificate(n, Fraction(delta))
        left = mp.mpf(record["left_high"].strip("[]").split(",")[0])
        right = mp.mpf(record["right_low"].strip("[]").split(",")[0])
        rows.append({"n": n, "delta": delta, "smaller": record["smaller"],
                     "left_upper": mp.nstr(left, 15), "right_lower": mp.nstr(right, 15),
                     "gap": mp.nstr(right - left, 4)})
    return rows


def figure(rows, path):
    """Three panels, each on its own axis: the lattice-class curvature of both
    pieces; the (n-1,2) piece zoomed at the sign change; the affine-class curvature
    of the (n-1,2) piece."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted, "axes.edgecolor": grid})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), facecolor=surface)
    by_n = {row["n"]: row for row in rows}

    def style(ax, title, xlabel="dimension n", ylabel=None):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.7)
        ax.axhline(0, color=muted, linewidth=0.9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", color=ink, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    wide = [n for n in by_n if n <= 60]
    ax = axes[0]
    style(ax, "Lattices of the first kind", ylabel="second variation of Iso at A\u2099*")
    ax.plot(wide, [by_n[n][PIECES[0]]["lattice"] for n in wide], color=blue, linewidth=2,
            label="standard piece")
    ax.plot(wide, [by_n[n][PIECES[1]]["lattice"] for n in wide], color=orange, linewidth=2,
            label="(n\u22121, 2) piece")
    ax.axvline(14.5, color=muted, linewidth=0.9, linestyle=(0, (2, 3)))
    ax.text(16, 0.9, "n = 15:\nfirst negative", color=ink, fontsize=8.5, va="center")
    ax.text(wide[-1], by_n[wide[-1]][PIECES[0]]["lattice"] - 0.35, "standard", color=ink,
            fontsize=8.5, ha="right", va="top")
    ax.text(wide[-1], by_n[wide[-1]][PIECES[1]]["lattice"] + 0.3, "(n\u22121, 2)", color=ink,
            fontsize=8.5, ha="right", va="bottom")
    ax.legend(frameon=False, loc="center right", fontsize=8.5, bbox_to_anchor=(1.0, 0.62))

    ax = axes[1]
    style(ax, "The (n\u22121, 2) piece at the sign change", ylabel="second variation of Iso at A\u2099*")
    zoom = [n for n in range(10, 21) if n in by_n]
    ax.plot(zoom, [by_n[n][PIECES[1]]["lattice"] for n in zoom], color=orange, linewidth=2,
            marker="o", markersize=6, markeredgecolor=surface, markeredgewidth=1.2)
    for n, offset in ((14, (10, 16)), (15, (-92, -22))):
        value = by_n[n][PIECES[1]]["lattice"]
        ax.annotate(f"n = {n}: {value:+.4f}".replace("-", "\u2212"), (n, value),
                    textcoords="offset points", xytext=offset, color=ink, fontsize=8.5,
                    arrowprops={"arrowstyle": "-", "color": muted, "linewidth": 0.8,
                                "shrinkA": 2, "shrinkB": 4})

    ax = axes[2]
    style(ax, "All weights and frames free", ylabel="Schur complement, \u00d710\u207b\u00b3")
    affine = [n for n in by_n if 9 <= n <= 40]
    ax.plot(affine, [1e3 * by_n[n][PIECES[1]]["schur"] for n in affine], color=orange, linewidth=2,
            marker="o", markersize=5, markeredgecolor=surface, markeredgewidth=1)
    ax.axvline(13.5, color=muted, linewidth=0.9, linestyle=(0, (2, 3)))
    ax.text(14.5, 1.3, "n = 14:\nfirst negative", color=ink, fontsize=8.5, va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    import json
    import time

    from qres.bench import RESULTS_DIR

    if argv and argv[0] == "--figure-only":
        saved = json.loads((RESULTS_DIR / "exp047_an_star_local_optimality.json").read_text())
        print(figure(saved["double_precision"], RESULTS_DIR / "figures" / "exp047_an_star_curvature.png"))
        return 0
    start = time.time()
    rows = curvature_table()
    print(f"{'n':>4s} {'lattice, standard':>18s} {'lattice, (n-1,2)':>17s} "
          f"{'affine, standard':>17s} {'affine, (n-1,2)':>16s}")
    for row in rows:
        print(f"{row['n']:4d} {row[PIECES[0]]['lattice']:18.6f} {row[PIECES[1]]['lattice']:17.6f} "
              f"{row[PIECES[0]]['schur']:17.6f} {row[PIECES[1]]['schur']:16.7f}")
    precise = high_precision_table()
    for row in precise:
        print("high precision:", row)
    proofs = witnesses()
    for item in proofs:
        print("witness:", item)
    certified = certificate_table()
    print("exact certificates:", sum(c["smaller"] for c in certified), "of", len(certified),
          "dimensions, n =", certified[0]["n"], "to", certified[-1]["n"])
    path = RESULTS_DIR / "exp047_an_star_local_optimality.json"
    path.write_text(json.dumps({"double_precision": rows, "high_precision": precise,
                                "witnesses": proofs, "exact_certificates": certified}, indent=1))
    print("saved", path, "and", figure(rows, RESULTS_DIR / "figures" / "exp047_an_star_curvature.png"),
          f"({time.time() - start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
