"""Experiment 023 -- does penalising the 1-norm make the THC fit reproducible?

Result 78 got THC's linear rank out of an optimised fit (`M ~ N^1.33 +- 0.26`)
and then undercut its own headline: **the fit barely reproduces.**  H4 at M=12
gave 3.4e-8 on one run and 9.3e-3 on the next, both converged; lambda moved
non-monotonically across ranks (8.2, 13.4, 9.1, 30.0, 17.7 on H6); and the
spread across six restarts reached 1.6e9.

The diagnosis it left was specific: **nothing in the objective steers the
optimiser towards the low-lambda optimum among the many near-degenerate ones.**
Published THC penalises the 1-norm; this one minimised the residual alone, so of
the many (chi, Z) that fit the tensor equally well, it returned whichever one
floating-point noise happened to reach.

That matters because **lambda is not a diagnostic, it is the cost**: a block
encoding's walk count is proportional to it (Result 75).  An unreproducible
lambda is an unreproducible runtime estimate.

So this measures one thing, on the rank where Result 78 saw the worst of it:

    with a penalty on lambda, does the SPREAD across independent fits shrink?

and one thing it must not cost:

    does the penalty push the energy outside chemical accuracy?

A penalty trades bias for variance.  Reporting only the variance it buys would
be the same selective quotation this project keeps catching; both are measured.

The penalty is a smooth |Z|::

    lambda(chi, Z) = 1/2 sum_mn h(Z_mn) |chi_m|^2 |chi_n|^2,  h(z) = sqrt(z^2+d^2) - d

differentiable everywhere, and its gradient was checked against finite
differences to 3.4e-8 before use -- the same check that caught a wrong gradient
in Result 78.

Run:  python -m experiments.exp023_penalised_thc
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
from qres.factorization import molecular_integrals
from qres.problems.chemistry import build_molecule

from experiments.exp021_tensor_hypercontraction import CHEMICAL_ACCURACY, energy_of
from experiments.exp022_optimised_thc import thc_candidates, thc_tensor

#: smoothing for |Z|; far below any coupling that matters, so the penalty is |Z|
#: everywhere it is active
DELTA = 1e-6

MAX_ITERATIONS = 20000

#: independent fits per configuration -- the spread across these IS the result
REPEATS = 8

#: penalty strengths to compare. 0 reproduces Result 78.
PENALTIES = (0.0, 1e-6, 1e-4, 1e-2)


def one_norm(chi: np.ndarray, coupling: np.ndarray) -> float:
    """The quantity the algorithm cost is proportional to."""
    scale = np.sum(chi**2, axis=0)
    return float(0.5 * np.abs(coupling * np.outer(scale, scale)).sum())


def _objective(packed, orbitals, rank, target, alpha):
    """Residual plus ``alpha`` times a smooth 1-norm, with analytic gradient."""
    split = orbitals * rank
    chi = packed[:split].reshape(orbitals, rank)
    coupling = packed[split:].reshape(rank, rank)

    outer = np.einsum("pm,qm->mpq", chi, chi)
    residual = np.einsum("mpq,mn,nrs->pqrs", outer, coupling, outer) - target

    value = 0.5 * float(np.sum(residual**2))
    grad_coupling = np.einsum("pqrs,mpq,nrs->mn", residual, outer, outer)
    forward = np.einsum("mn,nrs->mrs", coupling, outer)
    backward = np.einsum("npq,nm->mpq", outer, coupling)
    grad_chi = 2 * np.einsum("aqrs,qm,mrs->am", residual, chi, forward) + 2 * np.einsum(
        "pqas,sm,mpq->am", residual, chi, backward
    )

    if alpha:
        scale = np.sum(chi**2, axis=0)
        root = np.sqrt(coupling**2 + DELTA**2)
        smoothed = root - DELTA
        value += alpha * 0.5 * float(np.sum(smoothed * np.outer(scale, scale)))
        grad_coupling += alpha * 0.5 * (coupling / root) * np.outer(scale, scale)
        # both index positions of a column contribute
        grad_chi += alpha * chi * (smoothed @ scale + smoothed.T @ scale)[None, :]

    return value, np.concatenate([grad_chi.ravel(), grad_coupling.ravel()])


def fit(one_body, two_body, rank, alpha, start, iterations=MAX_ITERATIONS):
    orbitals = one_body.shape[0]
    packed = np.concatenate([start.ravel(), np.zeros(rank * rank)])
    result = minimize(
        _objective, packed, args=(orbitals, rank, two_body, alpha),
        jac=True, method="L-BFGS-B",
        options={"maxiter": iterations, "maxfun": iterations * 2,
                 "ftol": 1e-14, "gtol": 1e-10},
    )
    split = orbitals * rank
    chi = result.x[:split].reshape(orbitals, rank)
    coupling = result.x[split:].reshape(rank, rank)
    coupling = 0.5 * (coupling + coupling.T)
    return {
        "chi": chi,
        "coupling": coupling,
        "two_body": thc_tensor(chi, coupling),
        "one_norm": one_norm(chi, coupling),
        "iterations": int(result.nit),
        "converged": bool(result.success and result.nit < iterations),
    }


def spread(values) -> float:
    """Max over min -- the factor a reader would be wrong by, picking one run."""
    values = np.asarray([v for v in values if np.isfinite(v) and v > 0])
    if len(values) < 2:
        return float("nan")
    return float(values.max() / values.min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H4")
    ap.add_argument("--rank", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()

    print("=== experiment 023 :: does penalising lambda make the fit reproducible? ===")
    print(f"{args.molecule} at M={args.rank}, {args.repeats} independent fits per penalty.")
    print("Result 78 saw 3.4e-8 and 9.3e-3 on two runs of this configuration.\n")

    problem = build_molecule(args.molecule)
    one_body, two_body, _ = molecular_integrals(problem)
    orbitals = one_body.shape[0]
    reference = energy_of(problem, one_body, two_body)

    base = thc_candidates(one_body, two_body)[: args.rank].T.copy()
    if base.shape[1] < args.rank:
        pad = np.random.default_rng(0).normal(
            scale=1e-3, size=(orbitals, args.rank - base.shape[1])
        )
        base = np.hstack([base, pad])

    header = (
        f"{'penalty':>10}{'median err':>13}{'err spread':>12}"
        f"{'median lam':>12}{'lam spread':>12}{'in chem acc':>13}{'converged':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for alpha in PENALTIES:
        started = time.perf_counter()
        errors, norms, converged = [], [], 0
        rng = np.random.default_rng(4242)
        for repeat in range(args.repeats):
            start = base if repeat == 0 else base + rng.normal(
                scale=0.1 * repeat, size=base.shape
            )
            outcome = fit(one_body, two_body, args.rank, alpha, start)
            error = abs(energy_of(problem, one_body, outcome["two_body"]) - reference)
            errors.append(error)
            norms.append(outcome["one_norm"])
            converged += int(outcome["converged"])

        inside = sum(1 for e in errors if e < CHEMICAL_ACCURACY)
        row = {
            "penalty": alpha,
            "median_error": float(np.median(errors)),
            "error_spread": spread(errors),
            "median_one_norm": float(np.median(norms)),
            "one_norm_spread": spread(norms),
            "inside_chemical_accuracy": inside,
            "repeats": args.repeats,
            "converged": converged,
            "errors": [float(e) for e in errors],
            "one_norms": [float(n) for n in norms],
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(
            f"{alpha:>10.0e}{row['median_error']:>13.2e}{row['error_spread']:>12.1e}"
            f"{row['median_one_norm']:>12.2f}{row['one_norm_spread']:>12.1e}"
            f"{inside:>8}/{args.repeats:<4}{converged:>8}/{args.repeats:<3}",
            flush=True,
        )

    # --------------------------------------------------------------- verdict
    print("\n--- what the penalty bought and what it cost ---")
    baseline = rows[0]
    print(f"unpenalised: lambda spread {baseline['one_norm_spread']:.1e}, "
          f"{baseline['inside_chemical_accuracy']}/{args.repeats} inside chemical accuracy")

    usable = [r for r in rows[1:]
              if r["inside_chemical_accuracy"] >= baseline["inside_chemical_accuracy"]]
    if usable:
        best = min(usable, key=lambda r: r["one_norm_spread"])
        improvement = baseline["one_norm_spread"] / best["one_norm_spread"]
        print(f"best penalty {best['penalty']:.0e}: lambda spread "
              f"{best['one_norm_spread']:.1e} (**{improvement:.1f}x tighter**), "
              f"{best['inside_chemical_accuracy']}/{args.repeats} inside")
        print(f"median lambda moves {baseline['median_one_norm']:.2f} -> "
              f"{best['median_one_norm']:.2f}")
        if improvement < 2:
            print("\nThat is not a real improvement. The penalty does not fix the")
            print("reproducibility problem, and the open item stands.")
    else:
        print("No penalty kept as many fits inside chemical accuracy as the")
        print("unpenalised baseline. On this evidence the penalty trades away")
        print("more accuracy than it buys in reproducibility.")

    print("\nRead the spreads, not the medians: a median over eight fits hides")
    print("exactly the failure this experiment exists to measure.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # One path per configuration.  A shared filename means a second run silently
    # overwrites the first, and any test reading it then checks a different
    # molecule than its docstring claims.
    path = RESULTS_DIR / f"exp023_penalised_thc_{args.molecule}_M{args.rank}.json"
    with open(path, "w") as fh:
        json.dump({
            "molecule": args.molecule, "rank": args.rank, "repeats": args.repeats,
            "delta": DELTA, "chemical_accuracy": CHEMICAL_ACCURACY,
            "reference_energy": reference, "rows": rows,
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
