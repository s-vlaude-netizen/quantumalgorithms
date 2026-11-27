"""Experiment 054 -- every unweighted 4-regular circulant first-kind lattice on Z_{n+1}, certified exactly.

For each pair of generators a < b <= (n+1)/2 (rho = 1 on the differences +-a and +-b of Z_{n+1}, 0 elsewhere), up to the
equivalence given by multiplication with a unit of Z_{n+1} (which maps one such graph onto another), the exact log ratio
log Iso - log Iso(A_n*) is computed with exp051 (integer arithmetic, connectivity prefilter).  Only graphs that are connected
and 4-regular with distinct generators are considered.

usage: python experiments/exp054_unweighted_circulants_exact.py --n 15
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp051_certify_first_kind import certify  # noqa: E402


def circulant(n, gens):
    N1 = n + 1
    rho = {}
    for i in range(N1):
        for d in gens:
            j = (i + d) % N1
            rho[(min(i, j), max(i, j))] = Fraction(1)
    return rho


def canonical(N1, *gens):
    """Smallest representative of the generator set under multiplication by units and sign changes."""
    best = None
    for u in range(1, N1):
        if gcd(u, N1) != 1:
            continue
        key = tuple(sorted(min((u * g) % N1, (-u * g) % N1) for g in gens))
        if best is None or key < best:
            best = key
    return best


def main(n, k=2):
    N1 = n + 1
    seen = {}
    for gens in itertools.combinations(range(1, N1 // 2 + 1), k):
        if any(2 * g == N1 for g in gens):
            continue                                       # a difference of N/2 is a single (not doubled) class
        g0 = 0
        for g in gens:
            g0 = gcd(g0, g)
        if gcd(g0, N1) != 1:
            continue                                       # disconnected
        seen.setdefault(canonical(N1, *gens), gens)
    out = []
    for c, gens in sorted(seen.items()):
        v = float(certify(n, circulant(n, gens)))
        out.append({"gens": list(gens), "exact": v})
        print(f"   C_{N1}{tuple(gens)}: {v*1e3:+.4f}e-3", flush=True)
    out.sort(key=lambda r: r["exact"])
    print(f"n={n} (Z_{N1}), {2*k}-regular: best unweighted circulant: C_{N1}{tuple(out[0]['gens'])} at {out[0]['exact']*1e3:+.5f}e-3")
    Path(__file__).resolve().parents[1].joinpath("results", f"exp054_n{n}_k{k}.json").write_text(json.dumps({"n": n, "k": k, "results": out}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--k", type=int, default=2, help="number of generators (graph degree 2k)")
    a = ap.parse_args()
    main(a.n, a.k)
