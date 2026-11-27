"""Experiment 052 -- first-kind lattices whose Selling parameters are circulant on Z_{n+1}.

The unconstrained global search of exp050 ends, for n = 10, 11 (and apparently 12), at lattices whose parameters live on a
4-regular circulant graph with two weight classes (n = 10: weight 1 on differences +-1 and ~0.81 on +-4 of Z_11; n = 11: 1 and
~0.94 on two difference classes of Z_12).  Here rho_ij = c_d with d = the circular distance of i and j in Z_{n+1}, so the
lattice has one parameter per distance class (floor((n+1)/2) of them) and the cyclic symmetry is imposed.  The search is cheap
(a handful of parameters) and finds the best circulant for each n; it is then certified in exact arithmetic with exp051.

usage:  python experiments/exp052_circulant_scan.py --n 10 --starts 40 [--sigma 2.0] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp050_first_kind_search_numpy import Problem, lbfgs  # noqa: E402


def class_structure(prob: Problem):
    N1 = prob.N1
    cls = np.array([min((j - i) % N1, (i - j) % N1) - 1 for (i, j) in prob.pairs])
    return cls, int(cls.max()) + 1


def run(n, starts, sigma, seed):
    prob = Problem(n); fa = prob.f_an()
    cls, ncl = class_structure(prob)
    rng = np.random.default_rng(seed)

    def fun(xc):
        f, g = prob.value_grad(xc[cls])
        return f, np.bincount(cls, weights=g, minlength=ncl)

    out = []
    for s in range(starts):
        x0 = rng.normal(0, sigma, ncl)
        x, f, g, it = lbfgs(fun, x0, maxit=300)
        c = np.exp(x - x.max())
        out.append({"dlog": f - fa, "c": [float(v) for v in c], "iterations": it, "gnorm": float(np.max(np.abs(g)))})
    out.sort(key=lambda r: r["dlog"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True); ap.add_argument("--starts", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=2.0); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    res = run(a.n, a.starts, a.sigma, a.seed)
    best = res[0]
    print(f"n={a.n} (Z_{a.n+1}): best circulant log ratio vs A_n* = {best['dlog']*1e3:+.4f}e-3")
    print("   distance-class weights d=1..:", [round(v, 4) for v in best["c"]])
    distinct = sorted({round(r['dlog'] * 1e3, 3) for r in res})[:5]
    print("   lowest distinct end values (e-3):", distinct)
    outdir = Path(__file__).resolve().parents[1] / "results"
    (outdir / f"exp052_n{a.n}_seed{a.seed}.json").write_text(json.dumps({"n": a.n, "results": res[:10]}, indent=1))
