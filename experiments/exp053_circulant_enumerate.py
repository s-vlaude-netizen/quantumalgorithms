"""Experiment 053 -- all two-generator circulant first-kind lattices on Z_{n+1}, ranked, best ones certified exactly.

rho_ij = 1 if i - j = +-a (mod n+1), = w if i - j = +-b, = 0 otherwise (a < b <= (n+1)/2); w is optimised on (0, 1.6] by golden-section
search on log w.  Weights equal to 0 are held at 1e-12 in the floating-point evaluation (see exp050) and are exactly 0 in the
exact certificate.  Output: for each n the ten best (a, b, w) with the floating-point and exact values of
log Iso - log Iso(A_n*).

usage: python experiments/exp053_circulant_enumerate.py --n 12 [--top 10]

For large n the pairs (a, b) can be split over processes (set OPENBLAS_NUM_THREADS=1 for each): run
  python experiments/exp053_circulant_enumerate.py --n 19 --shard I/K        for I = 0 .. K-1   (writes results/exp053_n19_shardIofK.json)
then
  python experiments/exp053_circulant_enumerate.py --n 19 --merge K          (certifies the best three, writes results/exp053_n19.json).
Every pair is evaluated by the same code as in the unsplit run; only the assignment of pairs to processes differs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp050_first_kind_search_numpy import Problem  # noqa: E402
from experiments.exp051_certify_first_kind import certify  # noqa: E402

ZERO = 1e-12


def f_of(prob, fa, cls_a, cls_b, w):
    rho = np.full(len(prob.pairs), ZERO)
    rho[cls_a] = 1.0
    rho[cls_b] = w
    return prob.value_grad(np.log(rho))[0] - fa


def golden(fun, lo, hi, iters=28):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = fun(c), fun(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = fun(c)
        else:
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = fun(d)
    x = (a + b) / 2
    return x, fun(x)


def setup(n):
    prob = Problem(n); N1 = prob.N1; fa = prob.f_an()
    dist = np.array([min((j - i) % N1, (i - j) % N1) for (i, j) in prob.pairs])
    return prob, N1, fa, dist


def evaluate_pairs(n, shard=0, shards=1):
    """Floating-point optimum over w for every pair (a, b) whose running index is congruent to `shard` mod `shards`."""
    prob, N1, fa, dist = setup(n)
    out = []
    idx = -1
    for a in range(1, N1 // 2 + 1):
        for b in range(a + 1, N1 // 2 + 1):
            idx += 1
            if idx % shards != shard:
                continue
            ca, cb = dist == a, dist == b
            fun = lambda lw: f_of(prob, fa, ca, cb, math.exp(lw))
            grid = np.linspace(math.log(0.05), math.log(1.6), 21)
            vals = [fun(g) for g in grid]
            k = int(np.argmin(vals)); lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
            lw, fv = golden(fun, lo, hi, iters=22)
            out.append({"a": a, "b": b, "w": math.exp(lw), "float": fv, "w_is_1": abs(math.exp(lw) - 1) < 1e-3})
    return out


def best_with_certificates(n, out, top=10):
    prob, N1, fa, dist = setup(n)
    out = sorted(out, key=lambda r: r["float"])
    res = out[:top]
    for r in res[:3]:
        w = Fraction(r["w"]).limit_denominator(10**5)
        rho = {}
        for k, (i, j) in enumerate(prob.pairs):
            if dist[k] == r["a"]:
                rho[(i, j)] = Fraction(1)
            elif dist[k] == r["b"]:
                rho[(i, j)] = w
        r["exact"] = float(certify(n, rho)); r["w_rational"] = str(w)
    return res


def enumerate_n(n, top=10):
    return best_with_certificates(n, evaluate_pairs(n), top)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--shard", type=str, default=None, help="I/K: evaluate only the pairs with running index = I mod K and write a shard file")
    ap.add_argument("--merge", type=int, default=None, help="K: merge the K shard files of this n, certify the best three, write the result")
    ap.add_argument("--out", type=str, default=None, help="output path (default results/exp053_n<n>.json; with --merge the shard files are read from the same directory)")
    a = ap.parse_args()
    results_dir = Path(__file__).resolve().parents[1] / "results"
    if a.shard is not None:
        i, k = (int(x) for x in a.shard.split("/"))
        part = evaluate_pairs(a.n, i, k)
        path = Path(a.out) if a.out else results_dir / f"exp053_n{a.n}_shard{i}of{k}.json"
        path.write_text(json.dumps({"n": a.n, "shard": i, "shards": k, "pairs": part}, indent=1))
        print(f"n={a.n} shard {i}/{k}: {len(part)} pairs -> {path}")
        sys.exit(0)
    if a.merge is not None:
        allp = []
        src = Path(a.out).parent if a.out else results_dir
        for i in range(a.merge):
            d = json.loads((src / f"exp053_n{a.n}_shard{i}of{a.merge}.json").read_text())
            assert d["n"] == a.n and d["shard"] == i and d["shards"] == a.merge
            allp += d["pairs"]
        res = best_with_certificates(a.n, allp, a.top)
    else:
        res = enumerate_n(a.n, a.top)
    print(f"n={a.n} (Z_{a.n+1}): best two-generator circulants (generators a, b; weight 1 on a, w on b)")
    for r in res:
        ex = f"   exact {r['exact']*1e3:+.4f}e-3 (w = {r['w_rational']})" if "exact" in r else ""
        print(f"   a={r['a']:2d} b={r['b']:2d} w={r['w']:.4f}   float {r['float']*1e3:+.4f}e-3{ex}")
    out = Path(a.out) if a.out else results_dir / f"exp053_n{a.n}.json"
    out.write_text(json.dumps({"n": a.n, "results": res}, indent=1))
