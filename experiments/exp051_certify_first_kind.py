"""Experiment 051 -- exact certificate for the isoperimetric quotient of a first-kind lattice with rational Selling parameters.

Given rational rho_ij (n + 1 superbase vectors), computes  D = log Iso(rho) - log Iso(A_n*)  where
    Iso ~ U / d^(1 - 1/(2n)),   U = sum over cuts {S, S^c} of tau(S) tau(S^c) sqrt(r(S)),
with tau (weighted spanning-tree counts) and r (cut weights) computed in EXACT integer arithmetic (Bareiss fraction-free
determinants after clearing denominators) and only the square roots and logarithms in 60-digit decimal arithmetic.
No numpy floating point is involved.  The result is rigorous up to the decimal rounding (about 1e-50), far below any difference quoted.

Validated on the lattices certified in Result 100 (K(4,4) at n = 7 and K(5,5) at n = 9: -0.619e-3 and -1.488e-3, and the cone
K(1,5,5) with apex weight 21/40 at n = 10: -1.255e-3), see ``selftest``.

Usage as a module:  from experiments.exp051_certify_first_kind import certify; certify(n, rho_dict_or_matrix) -> Decimal
"""
from __future__ import annotations

import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from math import gcd, lcm

getcontext().prec = 60


def bareiss_det(M):
    """Exact determinant of an integer matrix (list of lists), fraction-free."""
    n = len(M)
    if n == 0:
        return 1
    A = [row[:] for row in M]
    sign = 1; prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = None
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    sw = i; break
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]; sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def to_integers(n, rho):
    """rho: dict {(i,j): Fraction} for i<j (missing = 0).  Returns integer matrix W and the common denominator."""
    N1 = n + 1
    den = 1
    for v in rho.values():
        den = lcm(den, Fraction(v).denominator)
    W = [[0] * N1 for _ in range(N1)]
    for (i, j), v in rho.items():
        w = int(Fraction(v) * den)
        W[i][j] = W[j][i] = w
    return W, den


def tau_cut_data(n, W):
    N1 = n + 1
    full = (1 << N1) - 1
    tau = {}
    for m in range(1, full):
        mem = [v for v in range(N1) if (m >> v) & 1]
        s = len(mem)
        if s == 1:
            tau[m] = 1; continue
        # reduced Laplacian (ground = mem[0])
        L = [[0] * (s - 1) for _ in range(s - 1)]
        for a in range(1, s):
            va = mem[a]
            deg = sum(W[va][vb] for vb in mem if vb != va)
            for b in range(1, s):
                vb = mem[b]
                L[a - 1][b - 1] = deg if a == b else -W[va][vb]
        tau[m] = bareiss_det(L)
    # cut weights
    r = {}
    for m in range(1, full):
        mem = [v for v in range(N1) if (m >> v) & 1]; out = [v for v in range(N1) if not (m >> v) & 1]
        r[m] = sum(W[i][j] for i in mem for j in out)
    # full determinant
    Lf = [[0] * n for _ in range(n)]
    for a in range(1, N1):
        deg = sum(W[a][b] for b in range(N1) if b != a)
        for b in range(1, N1):
            Lf[a - 1][b - 1] = deg if a == b else -W[a][b]
    d = bareiss_det(Lf)
    return tau, r, d, full


def log_iso(n, rho):
    """log( U / d^(1-1/(2n)) ) in decimal, for rho a dict of Fractions.  The scale of rho does not matter."""
    W, den = to_integers(n, rho)
    tau, r, d, full = tau_cut_data(n, W)
    U = Decimal(0)
    for m in range(1, full):
        if tau[m] == 0 or tau[full - m] == 0:
            continue
        U += Decimal(tau[m]) * Decimal(tau[full - m]) * Decimal(r[m]).sqrt()
    U /= 2
    c = Decimal(1) - Decimal(1) / (2 * n)
    return U.ln() - c * Decimal(d).ln()


def certify(n, rho):
    """log Iso(rho) - log Iso(A_n*); negative means better than A_n*."""
    N1 = n + 1
    an = {(i, j): Fraction(1) for i in range(N1) for j in range(i + 1, N1)}
    return log_iso(n, rho) - log_iso(n, an)


def multipartite(n, parts, within=Fraction(0), apex_weight=None):
    """rho = 1 across the parts of a complete multipartite graph (vertex 0.. in the order of the parts), `within` inside parts.
    If apex_weight is given, the first part must have size 1 (the apex) and its edges get that weight instead."""
    lab = []
    for k, sz in enumerate(parts):
        lab += [k] * sz
    assert len(lab) == n + 1
    rho = {}
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if lab[i] != lab[j]:
                w = Fraction(1)
                if apex_weight is not None and (lab[i] == 0 or lab[j] == 0):
                    w = Fraction(apex_weight)
                rho[(i, j)] = w
            elif within != 0:
                rho[(i, j)] = Fraction(within)
    return rho


def selftest():
    cases = [("K(3,3), n=5", 5, multipartite(5, (3, 3)), "+1.285e-3"),
             ("K(4,4), n=7", 7, multipartite(7, (4, 4)), "-0.619e-3"),
             ("K(5,5), n=9", 9, multipartite(9, (5, 5)), "-1.488e-3"),
             ("cone K(1,5,5) apex 21/40, n=10", 10, multipartite(10, (1, 5, 5), apex_weight=Fraction(21, 40)), "-1.255e-3")]
    for name, n, rho, expected in cases:
        v = certify(n, rho)
        print(f"{name}: log ratio = {float(v)*1e3:+.4f}e-3   (Result 100: {expected})   [{str(v)[:14]}...]")


if __name__ == "__main__":
    selftest()
