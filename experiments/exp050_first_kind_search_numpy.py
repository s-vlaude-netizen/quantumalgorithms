"""Experiment 050 -- global search over lattices of Voronoi's first kind, numpy only.

Same objective as exp048 (Result 100), re-implemented without scipy or mpmath so that it runs where only numpy is installed,
and used to take up the follow-up in NEXT_STEPS item 4: what are the minimisers among first-kind lattices for n = 10 .. 14?

A first-kind lattice of dimension n is a point rho (Selling parameters, one per pair of the n+1 superbase vectors).  Up to a
constant of n alone its isoperimetric quotient is

    Iso ~ U(rho) / d(rho)^(1 - 1/(2n)),     U = sum over cuts {S, S^c} of tau(S) tau(S^c) sqrt(r(S)),

tau(T) = weighted spanning-tree count of the complete graph on T (determinant of a reduced Laplacian), r(S) = weight of the cut,
d = tau(all n+1 vertices).  We report  f = log Iso - log Iso(A_n*)  (A_n* is rho = 1), so f < 0 means better than A_n*.

Gradient: d log tau(T) / d rho_ij = effective resistance between i and j in T (Kirchhoff), d r / d rho_ij = 1 across the cut.
The optimiser is a plain L-BFGS on log rho written here.  Checked against the values certified in Result 100:
K(4,4) at n = 7 and K(5,5) at n = 9 (see ``selftest``).

usage:  python experiments/exp050_first_kind_search_numpy.py selftest
        python experiments/exp050_first_kind_search_numpy.py search --n 10 --starts 20 --seed 1 [--sigma 1.0] [--maxit 400]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


class Problem:
    """Precomputed subset structure for dimension n (n + 1 superbase vectors)."""

    def __init__(self, n: int):
        self.n = n
        self.N1 = n + 1
        self.pairs = [(i, j) for i in range(self.N1) for j in range(i + 1, self.N1)]
        self.pidx = {p: k for k, p in enumerate(self.pairs)}
        self.full = (1 << self.N1) - 1
        masks = np.arange(1, self.full, dtype=np.int64)               # all nonempty proper subsets
        self.masks = masks
        self.Z = ((masks[:, None] >> np.arange(self.N1)[None, :]) & 1).astype(np.float64)
        self.size = self.Z.sum(1).astype(int)
        self.partner = (self.full - masks) - 1                        # row index of the complementary subset
        # group subsets by size; store the member vertex lists
        self.groups = {}
        for s in range(1, self.N1):
            rows = np.nonzero(self.size == s)[0]
            members = np.array([[v for v in range(self.N1) if (int(masks[r]) >> v) & 1] for r in rows], dtype=np.int64)
            self.groups[s] = (rows, members)
        self.c = 1.0 - 1.0 / (2.0 * n)
        iu = np.triu_indices(self.N1, 1)
        self.iu = iu

    def rho_matrix(self, rho_vec):
        R = np.zeros((self.N1, self.N1))
        R[self.iu] = rho_vec
        return R + R.T

    def tau_and_resistances(self, R, need_res=True):
        """tau[m] for every proper nonempty subset (row order of self.masks); optionally accumulates sum_m coef[m] * Res_m(i,j)."""
        tau = np.ones(len(self.masks))
        stores = {}
        for s, (rows, mem) in self.groups.items():
            if s == 1:
                tau[rows] = 1.0
                continue
            A = R[mem[:, :, None], mem[:, None, :]]                   # (k, s, s)
            L = -A
            idx = np.arange(s)
            L[:, idx, idx] = A.sum(2)
            Lr = L[:, 1:, 1:]
            sign, logdet = np.linalg.slogdet(Lr)
            tau[rows] = np.exp(logdet)
            if need_res:
                try:
                    G = np.linalg.inv(Lr)                              # (k, s-1, s-1)
                except np.linalg.LinAlgError:
                    G = np.linalg.pinv(Lr)                             # numerically singular: very small weights
                stores[s] = G
        return tau, stores

    def value_grad(self, x):
        """f(x) = log U - c log d  (absolute, not relative to A_n*) and its gradient with respect to x = log rho."""
        rho_vec = np.exp(x)
        R = self.rho_matrix(rho_vec)
        tau, stores = self.tau_and_resistances(R)
        Zr = self.Z @ R
        rcut = (Zr * (1.0 - self.Z)).sum(1)                           # weight of the cut (S, S^c)
        coef = tau * tau[self.partner] * np.sqrt(rcut)
        U = 0.5 * coef.sum()
        # d = tau(all): reduced Laplacian of the whole graph
        Lfull = -R.copy(); np.fill_diagonal(Lfull, R.sum(1))
        sign, logd = np.linalg.slogdet(Lfull[1:, 1:])
        try:
            Gfull = np.linalg.inv(Lfull[1:, 1:])
        except np.linalg.LinAlgError:
            Gfull = np.linalg.pinv(Lfull[1:, 1:])
        Rfull = np.zeros((self.N1, self.N1))
        dg = np.diag(Gfull)
        Rfull[1:, 1:] = dg[:, None] + dg[None, :] - 2 * Gfull
        Rfull[0, 1:] = dg; Rfull[1:, 0] = dg
        # gradient of U with respect to rho_ij
        gU = np.zeros((self.N1, self.N1))
        for s, (rows, mem) in self.groups.items():
            if s == 1:
                continue
            G = stores[s]
            dgl = np.diagonal(G, axis1=1, axis2=2)                    # (k, s-1)
            Res = np.zeros((len(rows), s, s))
            Res[:, 1:, 1:] = dgl[:, :, None] + dgl[:, None, :] - 2 * G
            Res[:, 0, 1:] = dgl; Res[:, 1:, 0] = dgl
            w = coef[rows]
            contrib = Res * w[:, None, None]
            np.add.at(gU, (mem[:, :, None], mem[:, None, :]), contrib)
        w_cut = coef / (2.0 * np.maximum(rcut, 1e-300))
        v = (w_cut[:, None] * self.Z).sum(0)
        W = (self.Z * w_cut[:, None]).T @ self.Z
        gcut = 0.5 * (v[:, None] + v[None, :] - 2 * W)
        gU = gU + gcut
        gU_vec = gU[self.iu]
        f = math.log(U) - self.c * logd
        grad_rho = gU_vec / U - self.c * Rfull[self.iu]
        return f, grad_rho * rho_vec

    def value(self, x):
        """Same f(x) as value_grad(x)[0], computed without the inverses and the gradient: far less memory and time."""
        rho_vec = np.exp(x)
        R = self.rho_matrix(rho_vec)
        tau, _ = self.tau_and_resistances(R, need_res=False)
        Zr = self.Z @ R
        rcut = (Zr * (1.0 - self.Z)).sum(1)
        coef = tau * tau[self.partner] * np.sqrt(rcut)
        U = 0.5 * coef.sum()
        Lfull = -R.copy(); np.fill_diagonal(Lfull, R.sum(1))
        sign, logd = np.linalg.slogdet(Lfull[1:, 1:])
        return math.log(U) - self.c * logd

    def f_an(self):
        return self.value(np.zeros(len(self.pairs)))


def lbfgs(fun, x0, maxit=400, m=10, gtol=1e-10, ftol=1e-15):
    """Plain L-BFGS with backtracking (Armijo) line search; fun(x) -> (f, g)."""
    FLOOR = -27.6      # weights below exp(-27.6) ~ 1e-12 of the largest are held at that floor: the floating-point determinants
                       # and inverses are not reliable below it (they gave spurious values at weights ~1e-40), and the
                       # contribution of such weights to the quotient is far below the digits quoted anyway.
    def safe(z):
        try:
            z0 = z - np.max(z)
            mask = z0 < FLOOR
            zc = np.where(mask, FLOOR, z0)
            fz, gz = fun(zc)
            if not (np.isfinite(fz) and np.all(np.isfinite(gz))):
                return np.inf, np.zeros_like(z)
            gz = np.array(gz, dtype=float)
            gz[mask] = 0.0                      # clipped coordinates do not move the function
            return fz, gz
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(z)
    x = x0.copy(); f, g = safe(x)
    S, Y = [], []
    for it in range(maxit):
        gn = np.max(np.abs(g))
        if gn < gtol:
            break
        q = g.copy(); alphas = []
        for s_, y_ in reversed(list(zip(S, Y))):
            rho = 1.0 / np.dot(y_, s_); a = rho * np.dot(s_, q); alphas.append((a, rho, s_, y_)); q -= a * y_
        if S:
            gamma = np.dot(S[-1], Y[-1]) / np.dot(Y[-1], Y[-1]); q *= gamma
        else:
            q *= 1.0 / max(1.0, np.linalg.norm(g))
        for a, rho, s_, y_ in reversed(alphas):
            b = rho * np.dot(y_, q); q += s_ * (a - b)
        d = -q
        if np.dot(d, g) >= 0:
            d = -g; S, Y = [], []
        t = 1.0; gd = np.dot(g, d); ok = False
        for _ in range(40):
            xn = x + t * d; fn, gnew = safe(xn)
            if np.isfinite(fn) and fn <= f + 1e-4 * t * gd:
                ok = True; break
            t *= 0.5
        if not ok:
            break
        s_ = xn - x; y_ = gnew - g
        if np.dot(s_, y_) > 1e-14:
            S.append(s_); Y.append(y_)
            if len(S) > m:
                S.pop(0); Y.pop(0)
        dec = f - fn
        x, f, g = xn, fn, gnew
        if dec < ftol * max(1.0, abs(f)) and gn < 1e-6:
            break
    return x, f, g, it


def bipartite_rho(prob: Problem, parts, eps=1e-9):
    """rho = 1 across the parts of a complete multipartite graph, eps inside parts (vertex order: part 0 first)."""
    lab = []
    for k, sz in enumerate(parts):
        lab += [k] * sz
    assert len(lab) == prob.N1
    return np.array([1.0 if lab[i] != lab[j] else eps for (i, j) in prob.pairs])


def selftest():
    # 1. gradient against finite differences
    for n in (5, 6, 7):
        prob = Problem(n); rng = np.random.default_rng(n)
        x = rng.normal(0, 0.7, len(prob.pairs)); f, g = prob.value_grad(x)
        num = np.zeros_like(g)
        for k in range(len(x)):
            h = 1e-6; e = np.zeros_like(x); e[k] = h
            num[k] = (prob.value_grad(x + e)[0] - prob.value_grad(x - e)[0]) / (2 * h)
        print(f"n={n}: gradient check, max |analytic - numeric| = {np.max(np.abs(g - num)):.2e}   (max |g| = {np.max(np.abs(g)):.2e})")
    # 2. A_n* is a critical point (symmetry)
    for n in (5, 7):
        prob = Problem(n); f, g = prob.value_grad(np.zeros(len(prob.pairs)))
        print(f"n={n}: |grad| at A_n* = {np.max(np.abs(g)):.2e}")
    # 3. certified values of Result 100: K(4,4) at n=7: -0.619e-3, K(5,5) at n=9: -1.488e-3, K(3,3) at n=5: +1.285e-3
    for n, parts, expected in ((5, (3, 3), 1.285e-3), (7, (4, 4), -0.619e-3), (9, (5, 5), -1.488e-3)):
        prob = Problem(n)
        for eps in (1e-6, 1e-9):
            f = prob.value_grad(np.log(bipartite_rho(prob, parts, eps)))[0] - prob.f_an()
            print(f"n={n} K{parts} eps={eps:g}: log ratio = {f*1e3:+.4f}e-3   (Result 100: {expected*1e3:+.3f}e-3)")


def search(n, starts, seed, sigma, maxit, outdir):
    prob = Problem(n); fa = prob.f_an(); rng = np.random.default_rng(seed)
    results = []
    t0 = time.time()
    for s in range(starts):
        x0 = rng.normal(0, sigma, len(prob.pairs))
        x, f, g, it = lbfgs(prob.value_grad, x0, maxit=maxit)
        rho = np.exp(x); rho = rho / np.max(rho)                     # scale is irrelevant (f is scale invariant); normalise
        small = int(np.sum(rho < 1e-3))
        results.append({"start": s, "dlog_vs_An": f - fa, "iterations": it, "n_small": small, "gnorm": float(np.max(np.abs(g))),
                        "rho_sorted_head": sorted(float(r) for r in rho)[:5]})
        print(f"n={n} start {s:3d}: log ratio vs A_n* = {(f-fa)*1e3:+9.4f}e-3   iterations {it:4d}   params < 1e-3: {small:3d}/{len(rho)}   |g|={np.max(np.abs(g)):.1e}   [{time.time()-t0:.0f}s]", flush=True)
        if f - fa < -1e-4:
            results[-1]["rho"] = [float(r) for r in rho]
    best = min(r["dlog_vs_An"] for r in results)
    print(f"n={n}: best log ratio over {starts} starts = {best*1e3:+.4f}e-3")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"exp050_n{n}_seed{seed}.json").write_text(json.dumps({"n": n, "seed": seed, "sigma": sigma, "maxit": maxit, "results": results}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "search"])
    ap.add_argument("--n", type=int, default=7); ap.add_argument("--starts", type=int, default=10); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=1.0); ap.add_argument("--maxit", type=int, default=400)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        search(a.n, a.starts, a.seed, a.sigma, a.maxit, Path(__file__).resolve().parents[1] / "results")
