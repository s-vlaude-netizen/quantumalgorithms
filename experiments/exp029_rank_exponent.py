"""Experiment 029 -- re-measuring the rank exponent on fits that actually converge.

Result 78 reported THC's rank scaling as ``M ~ N^1.33 +- 0.26`` and that number
carries the whole tensor-hypercontraction claim in this repository: THC's
argument is ``M ~ O(N)``, the selected-chi version of Result 77 gave
``N^2.26 +- 0.13``, and the non-overlapping intervals are what made "optimising
chi recovers the linear rank" a result rather than a hope.

**Result 80 then established that the fits underneath it mostly never
converged.** On H6 the iteration cap was *below the median requirement* -- 20 000
against 46 337 -- so the sweep's thresholds were read off fits stopped roughly a
third of the way through, and the `optimiser_converged` guard was rejecting
almost everything it was handed.

Result 80 also produced a better optimiser, and the improvement is not marginal.
Removing the model's exact gauge freedom -- rescaling a chi column against its Z
entries changes nothing -- is worth, on H6 at M = 18:

| | plain | gauge |
|---|---|---|
| converged | 7/8 | **8/8** |
| median iterations | 46 337 | **23 738** |
| lambda spread | 9.52 | **1.59** |
| energy error | 1.00e-4 | **2.33e-5** |

So the exponent is worth re-measuring rather than re-quoting, and this does it
with **everything else held fixed against Result 78**: the same molecules, the
same rank multiples, the same energy criterion (rebuild the Hamiltonian, compare
exact ground energies), and the same three guards -- stability under a perturbed
refit, convergence of lambda in the rank, and optimiser convergence.

Two things change, both from Result 80's recommendation:

* **the gauge-fixed parameterisation**, which is strictly better on every axis
  measured and has no hyperparameter
* **the 1-norm penalty at alpha = 1e-4**, which pins lambda (spread 1.00) at a
  cost of ~30x in energy error -- still inside chemical accuracy, and lambda is
  what a block encoding's cost is proportional to

**And one thing turned out to be wrong with Result 78's method, not just its
fits.** Its rank sweep tried only multiples of `N`, which quantises the threshold
to a multiple of `N` -- so a fitted exponent of exactly 1.0 can be an artefact of
the grid rather than a measurement, and a true threshold between `N` and `2N` is
invisible. Result 78 escaped this only because its thresholds happened to land in
different buckets, which is luck. This experiment therefore runs both grids: the
coarse one for comparability with Result 78, and a `--fine` grid stepping by one,
which is what the exponent should actually be read from. The fine grid is
affordable only because the gauge-fixed fits converge in a fraction of the
iterations the plain ones needed.

And one thing is *tested* rather than assumed. Result 78 needed six restarts
because a single start was not a measurement of anything. With gauge fixing and
the penalty, Result 80 measured a lambda spread of **1.00 across eight
independent starts** -- which would mean restarts have become unnecessary. This
runs three and records the spread, so that claim is checked on every rank rather
than inherited. If the spread stays at one, the multi-start machinery can go.

Run:  python -m experiments.exp029_rank_exponent
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize

from qres.bench import RESULTS_DIR
from qres.factorization import molecular_integrals, pauli_one_norm
from qres.problems.chemistry import build_molecule

from experiments.exp021_tensor_hypercontraction import (
    CHEMICAL_ACCURACY,
    CONVERGENCE_FACTOR,
    STABILITY_TOLERANCE,
    energy_of,
    thc_candidates,
)
from experiments.exp022_optimised_thc import RANK_MULTIPLES, thc_tensor
from experiments.exp023_penalised_thc import _objective, one_norm
from experiments.exp024_convergence_budget import gauge_objective, unpack_gauge

#: Result 80's recommendation: enough that gauge-fixed H6 converges with room to
#: spare (it needs 17-37k), and a fit that still hits it is reported as hitting it
MAX_ITERATIONS = 150000

#: Result 80's recommended penalty. 0 would reproduce the unpenalised fit.
PENALTY = 1e-4

#: down from Result 78's six -- and the spread is recorded on every rank so the
#: reduction is justified by measurement rather than by assumption
RESTARTS = 3

DEFAULT_MOLECULES = ("H2", "H4", "H6", "H8")


def _fit_once(one_body, two_body, rank, chi, alpha, iterations=MAX_ITERATIONS):
    """One gauge-fixed L-BFGS run from a given start."""
    orbitals = one_body.shape[0]
    packed = np.concatenate([chi.ravel(), np.zeros(rank * rank)])
    result = minimize(
        gauge_objective, packed, args=(orbitals, rank, two_body, alpha),
        jac=True, method="L-BFGS-B",
        options={"maxiter": iterations, "maxfun": iterations * 2,
                 "ftol": 1e-14, "gtol": 1e-10},
    )
    unit, coupling = unpack_gauge(result.x, orbitals, rank)
    coupling = 0.5 * (coupling + coupling.T)
    reconstructed = thc_tensor(unit, coupling)

    # The objective includes the penalty, so `result.fun` is not the residual.
    # Restart selection has to compare fit quality, not fit quality plus a
    # penalty term -- otherwise it would prefer a worse fit with a smaller
    # 1-norm, which is not what "best of several starts" should mean.
    residual, _ = _objective(
        np.concatenate([unit.ravel(), coupling.ravel()]), orbitals, rank,
        two_body, 0.0,
    )

    return {
        "rank": rank,
        "residual": float(residual),
        "penalised_objective": float(result.fun),
        "chi": unit,
        "coupling": coupling,
        "two_body": reconstructed,
        "tensor_error": float(np.max(np.abs(reconstructed - two_body))),
        "one_norm": one_norm(unit, coupling),
        "iterations": int(result.nit),
        "status": int(result.status),
        "converged_optimiser": bool(result.success and result.nit < iterations),
    }


def gauge_thc_fit(one_body, two_body, rank, alpha=PENALTY, restarts=RESTARTS,
                  iterations=MAX_ITERATIONS):
    """Best of several gauge-fixed starts, with the spread across them reported."""
    orbitals = one_body.shape[0]
    base = thc_candidates(one_body, two_body)[:rank].T.copy()
    rng = np.random.default_rng(20260907)
    if base.shape[1] < rank:
        base = np.hstack([base, rng.normal(scale=1e-3,
                                           size=(orbitals, rank - base.shape[1]))])

    attempts = []
    for index in range(max(1, restarts)):
        start = base if index == 0 else base + rng.normal(scale=0.1 * index,
                                                          size=base.shape)
        attempts.append(_fit_once(one_body, two_body, rank, start, alpha, iterations))

    best = min(attempts, key=lambda a: a["residual"])
    residuals = [a["residual"] for a in attempts]
    norms = [a["one_norm"] for a in attempts]
    best["restart_residual_spread"] = float(
        max(residuals) / max(min(residuals), 1e-300)
    )
    best["restart_one_norm_spread"] = float(max(norms) / max(min(norms), 1e-300))
    best["restart_one_norms"] = norms
    best["restarts"] = len(attempts)
    best["all_converged"] = all(a["converged_optimiser"] for a in attempts)
    return best


def _stability(one_body, two_body, rank, tight, alpha):
    """Refit from a perturbed start; a real optimum should not move.

    Identical tolerances and identical penalty to the fit being checked -- a
    refit run to a different stopping rule or a different objective would be
    measuring those instead of the optimum.
    """
    rng = np.random.default_rng(12345)
    orbitals = one_body.shape[0]
    perturbed = tight["chi"] + rng.normal(scale=1e-3, size=tight["chi"].shape)
    packed = np.concatenate([perturbed.ravel(), tight["coupling"].ravel()])
    result = minimize(
        gauge_objective, packed, args=(orbitals, rank, two_body, alpha),
        jac=True, method="L-BFGS-B",
        options={"maxiter": MAX_ITERATIONS, "maxfun": MAX_ITERATIONS * 2,
                 "ftol": 1e-14, "gtol": 1e-10},
    )
    unit, coupling = unpack_gauge(result.x, orbitals, rank)
    return one_norm(unit, 0.5 * (coupling + coupling.T))


def rank_grid(orbitals, multiples=RANK_MULTIPLES, fine=False):
    """Ranks to try.

    **The multiples grid builds the answer into the question.** Trying only
    ``M in {N, 2N, 3N, ...}`` quantises the threshold to a multiple of `N`, so if
    every molecule's threshold lands in the same bucket the fitted exponent is
    exactly 1.0 *by construction of the grid* rather than by measurement -- and a
    true threshold anywhere between `N` and `2N` cannot be seen at all.

    Result 78 escaped this only because its thresholds happened to land in
    *different* buckets. That is luck, not method.

    The fine grid steps by one from `N` to `3N`, so the threshold is resolved in
    absolute terms and the exponent is free to be whatever it is. It is
    affordable now only because the gauge-fixed fits converge in a fraction of
    the iterations the plain ones needed.
    """
    if fine:
        return list(range(max(2, orbitals), 3 * orbitals + 1))
    ranks, seen = [], set()
    for multiple in multiples:
        rank = multiple * orbitals
        if rank not in seen:
            seen.add(rank)
            ranks.append(rank)
    return ranks


def measure(name, multiples=RANK_MULTIPLES, alpha=PENALTY, restarts=RESTARTS,
            fine=False) -> dict:
    started = time.perf_counter()
    problem = build_molecule(name)
    one_body, two_body, _ = molecular_integrals(problem)
    orbitals = one_body.shape[0]
    reference = energy_of(problem, one_body, two_body)

    sweep = []
    for rank in rank_grid(orbitals, multiples, fine):

        fit = gauge_thc_fit(one_body, two_body, rank, alpha, restarts)
        error = abs(energy_of(problem, one_body, fit["two_body"]) - reference)

        perturbed_norm = _stability(one_body, two_body, rank, fit, alpha)
        largest = max(fit["one_norm"], perturbed_norm, 1e-12)
        stable = abs(fit["one_norm"] - perturbed_norm) / largest < STABILITY_TOLERANCE

        row = {
            "rank": rank,
            "rank_over_orbitals": rank / orbitals,
            "energy_error": error,
            "tensor_error": fit["tensor_error"],
            "one_norm": fit["one_norm"],
            "one_norm_perturbed": perturbed_norm,
            "stable": bool(stable),
            "iterations": fit["iterations"],
            "optimiser_converged": fit["converged_optimiser"],
            "all_restarts_converged": fit["all_converged"],
            "restart_residual_spread": fit["restart_residual_spread"],
            "restart_one_norm_spread": fit["restart_one_norm_spread"],
            "restart_one_norms": fit["restart_one_norms"],
            "restarts": fit["restarts"],
            "within_chemical_accuracy": bool(error < CHEMICAL_ACCURACY),
        }
        sweep.append(row)

        valid = [
            index for index, entry in enumerate(sweep)
            if entry["within_chemical_accuracy"] and entry["stable"]
            and entry["optimiser_converged"]
        ]
        # On the fine grid the neighbouring ranks are one apart, so two of them
        # is a much weaker check of lambda's convergence in the rank than two
        # steps of N was. Widen the window so the guard still compares against a
        # meaningfully larger rank.
        stop_after = valid[0] + (4 if fine else 2) if valid else None
        print(
            f"  {name:<4} M={rank:>4} (M/N={rank / orbitals:>3.0f})  "
            f"err {error:>9.2e}  lambda {fit['one_norm']:>8.2f}  "
            f"iters {fit['iterations']:>7}  "
            f"lam-spread {fit['restart_one_norm_spread']:>7.2f}"
            f"{'' if stable else ' UNSTABLE'}"
            f"{'' if fit['converged_optimiser'] else ' NOT-CONVERGED'}",
            flush=True,
        )
        if stop_after is not None and len(sweep) > stop_after:
            print(f"  {name:<4} stopping: two ranks past the first valid one",
                  flush=True)
            break

    smallest = None
    for index, row in enumerate(sweep):
        if not (row["within_chemical_accuracy"] and row["stable"]
                and row["optimiser_converged"]):
            continue
        larger = [r["one_norm"] for r in sweep[index + 1:] if r["stable"]]
        if larger and row["one_norm"] > CONVERGENCE_FACTOR * min(larger):
            row["converged"] = False
            continue
        row["converged"] = True
        smallest = row
        break

    return {
        "molecule": name,
        "orbitals": orbitals,
        "pauli_one_norm": pauli_one_norm(problem.hamiltonian),
        "sweep": sweep,
        "smallest_sufficient": smallest,
        "seconds": time.perf_counter() - started,
    }


def fit_exponent(orbitals, ranks):
    """Log-log slope with a standard error; never a slope on its own."""
    x, y = np.log(np.asarray(orbitals, float)), np.log(np.asarray(ranks, float))
    if len(x) < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    dof = len(x) - 2
    variance = float(np.sum(residual**2) / dof)
    stderr = float(np.sqrt(variance / np.sum((x - x.mean()) ** 2)))
    return float(slope), stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default=",".join(DEFAULT_MOLECULES))
    ap.add_argument("--penalty", type=float, default=PENALTY)
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--fine", action="store_true",
                    help="step the rank by 1 instead of by N -- see rank_grid")
    args = ap.parse_args()

    print("=== experiment 029 :: the rank exponent, on fits that converge ===")
    print(f"gauge-fixed parameterisation, penalty {args.penalty:.0e}, "
          f"{args.restarts} restarts, "
          f"{'fine (step 1)' if args.fine else 'coarse (multiples of N)'} rank grid.")
    print("Result 78 measured N^1.33 +- 0.26 on fits that mostly hit the cap.\n")

    rows = []
    for name in args.molecules.split(","):
        rows.append(measure(name, alpha=args.penalty, restarts=args.restarts,
                            fine=args.fine))

    print("\n--- thresholds ---")
    print(f"{'molecule':>9}{'orbitals':>10}{'M':>6}{'M/N':>6}{'lambda':>9}"
          f"{'vs Pauli':>10}{'iters':>9}")
    print("-" * 59)
    orbitals, thresholds = [], []
    for row in rows:
        best = row["smallest_sufficient"]
        if best is None:
            print(f"{row['molecule']:>9}{row['orbitals']:>10}  no rank passed all guards")
            continue
        orbitals.append(row["orbitals"])
        thresholds.append(best["rank"])
        print(f"{row['molecule']:>9}{row['orbitals']:>10}{best['rank']:>6}"
              f"{best['rank_over_orbitals']:>6.1f}{best['one_norm']:>9.2f}"
              f"{best['one_norm'] / row['pauli_one_norm']:>10.2f}"
              f"{best['iterations']:>9}")

    slope, stderr = fit_exponent(orbitals, thresholds)
    print(f"\n--- the exponent ---")
    print(f"  this experiment (gauge + penalty): M ~ N^{slope:.2f} +- {stderr:.2f}")
    print(f"  Result 78 (plain, mostly unconverged): M ~ N^1.33 +- 0.26")
    print(f"  Result 77 (selected chi):              M ~ N^2.26 +- 0.13")
    print(f"  THC's claim:                           M ~ N^1")
    if np.isfinite(slope) and np.isfinite(stderr):
        low, high = slope - 2 * stderr, slope + 2 * stderr
        print(f"\n  +-2 sigma interval: [{low:.2f}, {high:.2f}]")
        print(f"  contains linear (1.0): {'YES' if low <= 1.0 <= high else 'NO'}")
        print(f"  excludes Result 77's 2.26: "
              f"{'YES' if high < 2.26 else 'NO'}")

    # did restarts turn out to matter?
    spreads = [entry["restart_one_norm_spread"]
               for row in rows for entry in row["sweep"]]
    if spreads:
        print(f"\n--- were the restarts necessary? ---")
        print(f"  lambda spread across restarts: max {max(spreads):.3f}, "
              f"median {np.median(spreads):.3f}")
        if max(spreads) < 1.05:
            print("  Every rank returned the same lambda from every start. The")
            print("  multi-start machinery is now unnecessary, and Result 78 needed")
            print("  it only because the plain parameterisation made the landscape")
            print("  look multi-modal when the flat direction was the problem.")
        else:
            print("  Restarts still matter; the spread has not collapsed everywhere.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_fine" if args.fine else ""
    path = RESULTS_DIR / f"exp029_rank_exponent{suffix}.json"
    with open(path, "w") as fh:
        json.dump({
            "penalty": args.penalty, "restarts": args.restarts,
            "fine_grid": args.fine,
            "max_iterations": MAX_ITERATIONS,
            "chemical_accuracy": CHEMICAL_ACCURACY,
            "stability_tolerance": STABILITY_TOLERANCE,
            "convergence_factor": CONVERGENCE_FACTOR,
            "rows": rows,
            "threshold_orbitals": orbitals,
            "threshold_ranks": thresholds,
            "exponent": slope, "exponent_stderr": stderr,
            "result_78_exponent": 1.33, "result_78_stderr": 0.26,
            "result_77_exponent": 2.26, "result_77_stderr": 0.13,
        }, fh, indent=2, default=float)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
