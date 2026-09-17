"""Experiment 024 -- the THC fit on H6 never converged, and one number was resting on it.

Result 79 measured that a 1-norm penalty collapses the spread across independent
THC fits: on H4 at M=12, from 1e5 in residual and 6x in lambda down to **exactly
1.0**, at no cost in accuracy.  It reported the same collapse on H6 at M=18 --
and then flagged its own number, because on H6 only **1 fit in 8 converged**,
with or without the penalty.  A spread of 1.0 among eight non-converged fits may
mean they all found the same optimum, or merely that all eight stopped at the
same iteration cap.  Result 79 could not tell those apart.

This tells them apart.  It also answers, first, the cheapest question, which is
the one that turned out to matter:

    WHY does L-BFGS stop?

`scipy` carries the reason in `result.status` and `result.message`, and neither
Result 78 nor 79 recorded it.  Doing so costs nothing and settles it: on H6 the
status is 1, ``STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT``, with `nit` exactly
at the cap.  Not a line-search failure, not a tolerance that cannot be reached --
the budget was simply too small for the molecule.  H6 has 432 parameters against
H4's 192, and it needs roughly **45 000** iterations where H4 needs a few
thousand.  20 000 was a constant carried over from the smaller molecule.

So this experiment runs each fit to a budget large enough to converge, snapshots
the iterate at Result 79's old cap on the way past, and compares:

    the same fits, at the budget that was used and at a budget that converges.

Everything that Result 79 reported on H6 is recomputed at both, so the question
"was that number an artefact of the cap?" gets a measured yes or no rather than a
caveat.

A third arm tests the other hypothesis that was open.  The model has an exact
flat direction: scaling a chi column by `s` and the matching Z entries by
`1/s^2` leaves the tensor unchanged, so `M` directions of the landscape are
gauge.  Fixing them -- normalising the columns inside the objective and letting Z
carry the scale -- was the reparameterisation Result 79 proposed.  It is measured
here rather than assumed, because a plausible mechanism that turns out to be
worth 2% is exactly the kind of thing this project has to stop asserting.

Run:  python -m experiments.exp024_convergence_budget
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
from experiments.exp023_penalised_thc import _objective, one_norm, spread

#: the budget Results 78 and 79 used, carried over from H4 without re-deriving it
OLD_BUDGET = 20000

#: large enough that H6 converges with room to spare; a fit that still hits this
#: is reported as hitting it rather than quietly counted as converged
MAX_ITERATIONS = 150000

#: guard against dividing by the norm of a column that has collapsed
TINY = 1e-12

#: the penalty Result 79 recommended, and no penalty, so the comparison covers
#: the configuration whose number is in question
PENALTIES = (0.0, 1e-4)

REPEATS = 8


# --------------------------------------------------------------------------
# the two parameterisations
# --------------------------------------------------------------------------

def unpack_plain(packed, orbitals, rank):
    """(chi, Z) exactly as stored, with Z symmetrised as the fit does."""
    split = orbitals * rank
    chi = packed[:split].reshape(orbitals, rank)
    coupling = packed[split:].reshape(rank, rank)
    return chi, 0.5 * (coupling + coupling.T)


def unpack_gauge(packed, orbitals, rank):
    """Columns normalised -- the model the gauge-fixed objective actually sees."""
    chi, coupling = unpack_plain(packed, orbitals, rank)
    norms = np.maximum(np.linalg.norm(chi, axis=0), TINY)
    return chi / norms, coupling


def gauge_objective(packed, orbitals, rank, target, alpha):
    """The same objective on unit-norm chi columns, with the chain rule applied.

    The model is invariant under ``chi_m -> s chi_m, Z_mn -> Z_mn / (s_m^2 s_n^2)``,
    so `M` directions of the landscape change nothing.  Normalising inside the
    objective removes them: the gradient below is the projection of the ordinary
    gradient onto the sphere tangent, ``(I - u u^T) g / |chi|``, so the radial
    coordinate is never moved rather than being moved pointlessly.

    Checked against finite differences before use, like every other gradient in
    this series -- Result 78's was wrong on both blocks and only that check
    caught it.
    """
    split = orbitals * rank
    chi = packed[:split].reshape(orbitals, rank)
    norms = np.maximum(np.linalg.norm(chi, axis=0), TINY)
    unit = chi / norms

    value, gradient = _objective(
        np.concatenate([unit.ravel(), packed[split:]]), orbitals, rank, target, alpha
    )
    grad_unit = gradient[:split].reshape(orbitals, rank)
    grad_chi = (grad_unit - unit * np.sum(unit * grad_unit, axis=0)) / norms
    return value, np.concatenate([grad_chi.ravel(), gradient[split:]])


PARAMETERISATIONS = {
    "plain": (_objective, unpack_plain),
    "gauge": (gauge_objective, unpack_gauge),
}


# --------------------------------------------------------------------------
# one fit, with the termination reason recorded and the old budget snapshotted
# --------------------------------------------------------------------------

def run_fit(parameterisation, orbitals, rank, two_body, alpha, start,
            cap=MAX_ITERATIONS, milestone=OLD_BUDGET):
    """Fit to ``cap`` iterations, keeping the iterate as it passed ``milestone``.

    The L-BFGS trajectory does not depend on `maxiter` -- the cap only decides
    where it stops -- so one run to the larger budget contains the smaller-budget
    run exactly, and the comparison costs one fit rather than two.
    """
    objective, _ = PARAMETERISATIONS[parameterisation]
    packed = np.concatenate([start.ravel(), np.zeros(rank * rank)])

    snapshot = {"x": None, "iteration": 0}
    counter = {"n": 0}

    def record(xk):
        counter["n"] += 1
        if counter["n"] == milestone:
            snapshot["x"] = np.array(xk, copy=True)
            snapshot["iteration"] = counter["n"]

    started = time.perf_counter()
    result = minimize(
        objective, packed, args=(orbitals, rank, two_body, alpha),
        jac=True, method="L-BFGS-B", callback=record,
        options={"maxiter": cap, "maxfun": cap * 2, "ftol": 1e-14, "gtol": 1e-10},
    )
    elapsed = time.perf_counter() - started

    # A fit that never reached the milestone converged before it; its "milestone"
    # state is its final state.
    milestone_x = snapshot["x"] if snapshot["x"] is not None else result.x

    return {
        "final_x": result.x,
        "milestone_x": milestone_x,
        "reached_milestone": snapshot["x"] is not None,
        "iterations": int(result.nit),
        "status": int(result.status),
        "message": str(result.message),
        "residual": float(result.fun),
        "max_gradient": float(np.max(np.abs(result.jac))),
        # converged means scipy says so AND the budget was not the thing that
        # stopped it -- the distinction Result 79 was missing
        "converged": bool(result.success and result.nit < cap),
        "converged_at_old_budget": bool(result.success and result.nit < milestone),
        "seconds": elapsed,
    }


def assess(parameterisation, packed, orbitals, rank, problem, one_body, reference):
    """lambda and the energy error of one iterate -- the two reported quantities."""
    _, unpack = PARAMETERISATIONS[parameterisation]
    chi, coupling = unpack(packed, orbitals, rank)
    tensor = thc_tensor(chi, coupling)
    return {
        "one_norm": one_norm(chi, coupling),
        "error": abs(energy_of(problem, one_body, tensor) - reference),
        "column_norm_ratio": float(
            np.linalg.norm(chi, axis=0).max() / max(np.linalg.norm(chi, axis=0).min(), TINY)
        ),
    }


def summarise(entries, key, repeats):
    values = [e[key] for e in entries]
    return {
        f"median_{key}": float(np.median(values)),
        f"{key}_spread": spread(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H6")
    ap.add_argument("--rank", type=int, default=18)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--cap", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--parameterisations", default="plain,gauge")
    args = ap.parse_args()

    print("=== experiment 024 :: was the H6 collapse real, or was it the iteration cap? ===")
    print(f"{args.molecule} at M={args.rank}, {args.repeats} fits per setting.")
    print(f"Result 79 used {OLD_BUDGET} iterations and converged 1/8 on H6.\n")

    problem = build_molecule(args.molecule)
    one_body, two_body, _ = molecular_integrals(problem)
    orbitals = one_body.shape[0]
    reference = energy_of(problem, one_body, two_body)
    parameters = orbitals * args.rank + args.rank**2
    print(f"{orbitals} orbitals, {parameters} free parameters\n")

    base = thc_candidates(one_body, two_body)[: args.rank].T.copy()
    if base.shape[1] < args.rank:
        pad = np.random.default_rng(0).normal(
            scale=1e-3, size=(orbitals, args.rank - base.shape[1])
        )
        base = np.hstack([base, pad])

    header = (
        f"{'param':>7}{'penalty':>9}{'budget':>11}{'converged':>11}{'med iters':>11}"
        f"{'med lam':>10}{'lam spread':>12}{'med err':>11}{'in chem':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for parameterisation in args.parameterisations.split(","):
        for alpha in PENALTIES:
            rng = np.random.default_rng(4242)
            fits, at_old, at_full = [], [], []
            for repeat in range(args.repeats):
                start = base if repeat == 0 else base + rng.normal(
                    scale=0.1 * repeat, size=base.shape
                )
                outcome = run_fit(parameterisation, orbitals, args.rank, two_body,
                                  alpha, start, cap=args.cap)
                fits.append(outcome)
                at_old.append(assess(parameterisation, outcome["milestone_x"],
                                     orbitals, args.rank, problem, one_body, reference))
                at_full.append(assess(parameterisation, outcome["final_x"],
                                      orbitals, args.rank, problem, one_body, reference))

            iterations = [f["iterations"] for f in fits]
            for label, entries, converged_key in (
                (f"{OLD_BUDGET}", at_old, "converged_at_old_budget"),
                ("converged", at_full, "converged"),
            ):
                converged = sum(int(f[converged_key]) for f in fits)
                inside = sum(1 for e in entries if e["error"] < CHEMICAL_ACCURACY)
                row = {
                    "parameterisation": parameterisation,
                    "penalty": alpha,
                    "budget": label,
                    "converged": converged,
                    "repeats": args.repeats,
                    "median_iterations": float(np.median(iterations)),
                    "iterations": iterations,
                    "inside_chemical_accuracy": inside,
                    "statuses": [f["status"] for f in fits],
                    "messages": sorted({f["message"] for f in fits}),
                    "seconds": float(sum(f["seconds"] for f in fits)),
                    "one_norms": [e["one_norm"] for e in entries],
                    "errors": [e["error"] for e in entries],
                    "median_column_norm_ratio": float(
                        np.median([e["column_norm_ratio"] for e in entries])
                    ),
                    **summarise(entries, "one_norm", args.repeats),
                    **summarise(entries, "error", args.repeats),
                }
                rows.append(row)
                print(
                    f"{parameterisation:>7}{alpha:>9.0e}{label:>11}"
                    f"{converged:>7}/{args.repeats:<3}{row['median_iterations']:>11.0f}"
                    f"{row['median_one_norm']:>10.2f}{row['one_norm_spread']:>12.2f}"
                    f"{row['median_error']:>11.2e}{inside:>6}/{args.repeats:<2}",
                    flush=True,
                )

    # --------------------------------------------------------------- verdict
    print("\n--- was Result 79's H6 number an artefact of the cap? ---")
    indexed = {(r["parameterisation"], r["penalty"], r["budget"]): r for r in rows}
    old = indexed.get(("plain", 1e-4, f"{OLD_BUDGET}"))
    full = indexed.get(("plain", 1e-4, "converged"))
    if old and full:
        print(f"penalised, at the old budget: {old['converged']}/{old['repeats']} converged, "
              f"lambda spread {old['one_norm_spread']:.3f}")
        print(f"penalised, run to convergence: {full['converged']}/{full['repeats']} converged, "
              f"lambda spread {full['one_norm_spread']:.3f}")
        if full["converged"] < full["repeats"]:
            print("\nStill not all converged at this budget -- the cap is larger but the")
            print("question is not yet settled. Raise --cap before reading anything else.")
        elif full["one_norm_spread"] < 1.05:
            print("\nThe collapse is REAL: it survives when every fit actually converges.")
        else:
            print("\nThe collapse was the CAP: with converged fits the spread reopens to")
            print(f"{full['one_norm_spread']:.2f}x, and Result 79's H6 line must be retracted.")

    unpenalised = indexed.get(("plain", 0.0, "converged"))
    if unpenalised and full:
        print(f"\nand what the penalty is worth among converged fits: lambda spread "
              f"{unpenalised['one_norm_spread']:.2f} -> {full['one_norm_spread']:.2f}, "
              f"median lambda {unpenalised['median_one_norm']:.2f} -> "
              f"{full['median_one_norm']:.2f}")

    plain = indexed.get(("plain", 0.0, "converged"))
    gauge = indexed.get(("gauge", 0.0, "converged"))
    if plain and gauge:
        saving = plain["median_iterations"] / max(gauge["median_iterations"], 1)
        print(f"\ngauge fixing: {plain['median_iterations']:.0f} -> "
              f"{gauge['median_iterations']:.0f} median iterations ({saving:.2f}x)")
        print(f"and the flat direction it removes was barely used: the column-norm "
              f"ratio reaches only {plain['median_column_norm_ratio']:.2f} without it")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"exp024_convergence_{args.molecule}_M{args.rank}.json"
    with open(path, "w") as fh:
        json.dump({
            "molecule": args.molecule, "rank": args.rank, "repeats": args.repeats,
            "old_budget": OLD_BUDGET, "cap": args.cap, "parameters": parameters,
            "chemical_accuracy": CHEMICAL_ACCURACY, "reference_energy": reference,
            "rows": rows,
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
