"""Experiment 038 -- are small-molecule THC rank thresholds chemistry, or algebra?

Result 78 measured THC's chemical-accuracy rank as ``M = 4, 8, 18`` for
``N = 2, 4, 6`` orbitals and fitted ``M ~ N^1.33``; that exponent carries the
repository's claim that optimised THC "recovers the linear rank". The
re-measurement on converged fits (exp029) was never completed.

Before re-measuring an exponent, ask what the threshold *is*. A symmetric-``Z``
THC model has ``N M + M(M+1)/2`` parameters, ``M`` of them gauge (Result 86).
An 8-fold-symmetric two-electron tensor on ``N`` orbitals has ``P(P+1)/2``
independent entries, ``P = N(N+1)/2``. So the dimension count says the model can
represent a *generic* tensor exactly only once

    N M + M(M-1)/2  >=  P(P+1)/2 ,       i.e. M ~ N^2 / 2 asymptotically,

which is ``M = 3, 8, 17, 30`` for ``N = 2, 4, 6, 8`` -- and Result 78's thresholds
were ``4, 8, 18``. If the small-molecule threshold is where the fit becomes
*exact*, it is set by algebra, grows as ``N^2``, and says nothing about THC's
asymptotic ``M ~ O(N)``, which is a statement about the regime where chemical
accuracy is reached far *below* exactness (FeMoco: ``M ~ 8N`` against an
algebraic count of ~1 500 for ``N = 54``).

**The discriminating measurement.** Sweep the rank one step at a time and record,
next to the energy error, the fit's **residual**. If chemical accuracy arrives
together with a collapse of the residual to the level of an exact fit, the
threshold is algebraic. If chemical accuracy arrives while the residual is still
far from zero, the threshold is chemistry.

**Fits:** gauge-fixed, **unpenalised** (Result 90: the penalty only moves lambda
along exact minimisers and costs accuracy), best of 3 starts, 150 000-iteration
cap. Energy criterion as everywhere in this series: rebuild the Hamiltonian,
compare exact ground energies, chemical accuracy 1.6e-3 Ha.

One (molecule, rank) per invocation, so ranks can run as parallel processes:

    python -m experiments.exp038_thc_rank_regime --molecule H6 --ranks 10,11,12
    python -m experiments.exp038_thc_rank_regime --summarise
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qres.bench import RESULTS_DIR

RAW = RESULTS_DIR / "raw" / "exp038"
RESTARTS = 3
CHEMICAL_ACCURACY = 1.6e-3

#: a fit counts as exact when its residual (half the squared Frobenius error of
#: the tensor) is below this -- the converged exact fits in Results 89-90 sit
#: at 1e-9 to 1e-7, the inexact ones at 1e-4 and above
EXACT_RESIDUAL = 1e-6


def pairs(orbitals):
    return orbitals * (orbitals + 1) // 2


def independent_entries(orbitals):
    """Independent entries of an 8-fold-symmetric tensor on ``orbitals`` orbitals."""
    return pairs(orbitals) * (pairs(orbitals) + 1) // 2


def algebraic_rank(orbitals):
    """Smallest M whose non-gauge parameter count reaches the independent entries."""
    need = independent_entries(orbitals)
    rank = 1
    while orbitals * rank + rank * (rank - 1) // 2 < need:
        rank += 1
    return rank


def measure_rank(name, rank, restarts=RESTARTS, bond_length=None):
    from qres.factorization import molecular_integrals
    from qres.problems.chemistry import build_molecule

    from experiments.exp021_tensor_hypercontraction import energy_of
    from experiments.exp029_rank_exponent import MAX_ITERATIONS, gauge_thc_fit

    started = time.perf_counter()
    problem = (build_molecule(name) if bond_length is None
               else build_molecule(name, bond_length=bond_length))
    one_body, two_body, _ = molecular_integrals(problem)
    reference = energy_of(problem, one_body, two_body)
    fit = gauge_thc_fit(one_body, two_body, rank, alpha=0.0, restarts=restarts,
                        iterations=MAX_ITERATIONS)
    error = abs(energy_of(problem, one_body, fit["two_body"]) - reference)
    orbitals = one_body.shape[0]
    return {
        "molecule": label(name, bond_length), "orbitals": orbitals, "rank": rank,
        "bond_length": bond_length,
        "residual": fit["residual"], "energy_error": error,
        "within_chemical_accuracy": bool(error < CHEMICAL_ACCURACY),
        "exact": bool(fit["residual"] < EXACT_RESIDUAL),
        "one_norm": fit["one_norm"],
        "restart_residual_spread": fit["restart_residual_spread"],
        "all_converged": fit["all_converged"], "iterations": fit["iterations"],
        "algebraic_rank": algebraic_rank(orbitals),
        "seconds": time.perf_counter() - started,
    }


def pair_spectrum(two_body):
    """Eigenvalues of the ERI pair matrix in the metric the THC residual uses.

    The fit's residual is half the squared Frobenius error over all ``N^4``
    entries, where an off-diagonal pair ``(p, q)`` appears twice; weighting the
    pair-space matrix by the square roots of those multiplicities makes its
    eigenvalues the ones Eckart-Young applies to. They are invariant under
    orbital rotations.
    """
    import numpy as np

    orbitals = two_body.shape[0]
    index = [(p, q) for p in range(orbitals) for q in range(p, orbitals)]
    weight = np.sqrt([1.0 if p == q else 2.0 for p, q in index])
    matrix = np.array([[two_body[p, q, r, s] for r, s in index] for p, q in index])
    return np.sort(np.abs(np.linalg.eigvalsh(weight[:, None] * matrix * weight[None, :])))[::-1]


def eckart_young_bound(spectrum, rank):
    """Least residual any rank-``rank`` factorisation can reach -- THC included.

    ``X Z X^T`` has rank at most ``M``, so no choice of ``chi`` and ``Z`` can beat
    the truncated eigendecomposition.
    """
    return 0.5 * float((spectrum[rank:] ** 2).sum())


#: eigenvalues above this count as significant; on every equilibrium chain here
#: the last significant one is ~0.024 and the next ~1e-3
SIGNIFICANT = 1e-2


def spectrum_of(name, bond_length):
    from qres.factorization import molecular_integrals
    from qres.problems.chemistry import build_molecule

    problem = (build_molecule(name) if bond_length is None
               else build_molecule(name, bond_length=bond_length))
    return pair_spectrum(molecular_integrals(problem)[1])


def label(name, bond_length):
    return name if bond_length is None else f"{name}@{bond_length:g}"


def summarise():
    rows = [json.loads(path.read_text()) for path in sorted(RAW.glob("*.json"))]
    by_molecule = {}
    for row in rows:
        by_molecule.setdefault(row["molecule"], []).append(row)
    summary = []
    for name, entries in sorted(by_molecule.items(), key=lambda kv: kv[1][0]["orbitals"]):
        entries.sort(key=lambda r: r["rank"])
        orbitals = entries[0]["orbitals"]
        spectrum = spectrum_of(name.split("@")[0], entries[0].get("bond_length"))
        significant = int((spectrum > SIGNIFICANT).sum())
        for r in entries:
            r["eckart_young_bound"] = eckart_young_bound(spectrum, r["rank"])
        print(f"\n  {name} (N={orbitals}, algebraic count M >= {algebraic_rank(orbitals)}, "
              f"identifiability boundary {orbitals * (orbitals - 1) // 2 + 1})")
        print(f"  significant ERI eigenvalues (> {SIGNIFICANT:g}): {significant};"
              f"  2N - 1 = {2 * orbitals - 1}")
        print(f"  {'M':>4}{'residual':>12}{'Eckart-Young':>14}{'energy err':>12}"
              f"{'chem acc':>10}{'exact':>7}{'lambda':>9}{'converged':>11}")
        for r in entries:
            print(f"  {r['rank']:>4}{r['residual']:>12.2e}{r['eckart_young_bound']:>14.2e}"
                  f"{r['energy_error']:>12.2e}"
                  f"{'yes' if r['within_chemical_accuracy'] else 'no':>10}"
                  f"{'yes' if r['exact'] else 'no':>7}{r['one_norm']:>9.2f}"
                  f"{'yes' if r['all_converged'] else 'no':>11}")
        accurate = [r["rank"] for r in entries if r["within_chemical_accuracy"]]
        exact = [r["rank"] for r in entries if r["exact"]]
        summary.append({
            "molecule": name, "orbitals": orbitals,
            "first_chemically_accurate_rank": min(accurate) if accurate else None,
            "first_exact_rank": min(exact) if exact else None,
            "algebraic_rank": algebraic_rank(orbitals),
            "significant_eigenvalues": significant,
            "spectrum_head": [float(v) for v in spectrum[:2 * orbitals + 2]],
            "ranks_measured": [r["rank"] for r in entries], "rows": entries,
        })
        print(f"  first chemically accurate rank: {summary[-1]['first_chemically_accurate_rank']}"
              f",  first exact rank: {summary[-1]['first_exact_rank']}")
    path = RESULTS_DIR / "exp038_thc_rank_regime.json"
    with open(path, "w") as fh:
        json.dump({"exact_residual": EXACT_RESIDUAL, "chemical_accuracy": CHEMICAL_ACCURACY,
                   "molecules": summary}, fh, indent=2, default=float)
    print(f"\nsaved -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule")
    ap.add_argument("--ranks", help="comma-separated")
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--bond-length", type=float, default=None,
                    help="stretch the chain; default is each molecule's equilibrium")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()

    if args.summarise:
        print("=== experiment 038 :: small-molecule THC thresholds, chemistry or algebra ===")
        summarise()
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    for rank in (int(r) for r in args.ranks.split(",")):
        row = measure_rank(args.molecule, rank, args.restarts, args.bond_length)
        tag = label(args.molecule, args.bond_length)
        (RAW / f"{tag}_M{rank:03d}.json").write_text(json.dumps(row, default=float))
        print(f"  {tag} M={rank:>3}  residual {row['residual']:.2e}  "
              f"err {row['energy_error']:.2e}  lambda {row['one_norm']:.2f}  "
              f"({row['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
