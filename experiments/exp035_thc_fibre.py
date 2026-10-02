"""Experiment 035 -- lowering THC's lambda without moving the tensor at all.

Result 89 proved that past ``M = N(N-1)/2 + 1`` the THC model has exactly flat
directions beyond the gauge -- the rank-one columns can slide inside their span
-- and that lambda, the quantity the block encoding's cost is proportional to,
is **not** constant along them. Every chemical-accuracy rank this repository
fitted (H4 at 8, H6 at 18) is past that boundary.

That makes a prediction, and it is testable:

    at those ranks, lambda can be lowered along the manifold of exact
    minimisers, with the reconstructed tensor -- hence the Hamiltonian, hence
    the energy -- unchanged to machine precision.

And it bears on a price this repository paid. Result 80 found that the lambda
penalty collapses the restart spread (1.59 -> 1.00 on H6 at M=18) but **costs
30x in energy error** (2.33e-5 -> 7.41e-4), and recommended the penalty "only
when lambda must be pinned". If the penalty's benefit comes from the slides, the
same lambda is available from an unpenalised fit by sliding, at the unpenalised
fit's accuracy.

**The construction.** At a fitted ``(chi, Z)`` with ``V0 = X Z X^T``, keep the
span ``U`` of the columns fixed: every column must stay a rank-one matrix inside
``U``, i.e. ``chi_m^T B_k chi_m = 0`` for a basis ``B_k`` of ``U``'s complement,
and ``Z`` is re-solved exactly as ``X^+ V0 X^+T``. Columns are kept at unit
norm, which removes the gauge (lambda is invariant along it anyway). Then
lambda is minimised over what is left -- the slides -- by SLSQP, and the tensor
change is measured, not assumed.

**Three regimes, and the middle one was a surprise.** Rank-one matrices in the
span ``U`` are the real solutions ``x`` of the ``P - M`` quadrics
``x^T B_k x = 0`` in ``P^(N-1)``.

* ``M < P - N + 1`` (H4 at 6): more quadrics than dimensions, so generically the
  only solutions are the ``M`` columns themselves. Nothing can move -- **the
  control**: the slide must return the lambda it started from.
* ``M = P - N + 1`` (H4 at 7): ``N - 1`` quadrics in ``P^(N-1)``, which by Bezout
  meet in ``2^(N-1)`` points over C -- 8 for H4. Seven are the real columns;
  complex solutions pair up, so **the eighth is real too**. Any 7 of the 8 that
  span ``U`` represent the same tensor exactly. Result 89's count of
  *continuous* flat directions is untouched (there are none beyond the gauge),
  but the fibre is not a point: it is a **finite set** of exact representations,
  enumerated here completely.
* ``P - N + 1 < M < P`` (H4 at 8, H6 at 18): positive-dimensional -- the slides.

I first used H4 at 7 as the control, because Result 89 puts it inside the
gauge-only regime; the slide then lowered lambda on 3 of 8 starts with the
tensor unchanged to 1e-15, which is how the finite alternatives were found.

Run:  python -m experiments.exp035_thc_fibre [--repeats 8]
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

from experiments.exp021_tensor_hypercontraction import (
    CHEMICAL_ACCURACY,
    energy_of,
    thc_candidates,
)
from experiments.exp022_optimised_thc import thc_tensor
from experiments.exp029_rank_exponent import MAX_ITERATIONS, PENALTY, _fit_once

#: (molecule, rank): H4 at 6 is strictly inside (the control), H4 at 7 sits on
#: the boundary (finitely many alternatives), H4 at 8 and H6 at 18 -- Result 78's
#: chemical-accuracy ranks -- are past it (continuous slides).
CASES = (("H4", 6), ("H4", 7), ("H4", 8), ("H6", 18))

#: random starts for finding every real rank-one point in the span; 8 are
#: expected on H4 at M=7 and the search is required to stop finding new ones
POINT_SEARCH_STARTS = 3000

#: Result 80's protocol: eight starts, perturbation 0.1 x repeat, seed 4242
REPEATS = 8
START_SEED = 4242

#: a slide is accepted only if the tensor moved by less than this (absolute,
#: entries of order 1): anything larger would be a different Hamiltonian
TENSOR_TOLERANCE = 1e-9

#: starts for the slide itself; the first is the fitted point
SLIDE_STARTS = 4

#: smoothing of |Z| inside the optimiser only -- every reported lambda is exact
SMOOTHING = 1e-9


def pair_index(orbitals):
    return [(p, q) for p in range(orbitals) for q in range(p, orbitals)]


def columns(chi):
    """``X``: column m is ``chi_m chi_m^T`` read on pairs ``p <= q``."""
    orbitals, rank = chi.shape
    return np.array([[chi[p, m] * chi[q, m] for m in range(rank)]
                     for p, q in pair_index(orbitals)])


def one_norm_exact(chi, coupling):
    weight = np.sum(chi**2, axis=0)
    return 0.5 * float(np.sum(np.abs(coupling) * np.outer(weight, weight)))


def complement_forms(chi):
    """Symmetric ``B_k`` with ``x^T B_k x = 0`` iff ``x x^T`` has no component
    along the k-th direction orthogonal to the span of the columns."""
    orbitals, rank = chi.shape
    basis, _, _ = np.linalg.svd(columns(chi), full_matrices=True)
    complement = basis[:, rank:]
    forms = []
    for k in range(complement.shape[1]):
        form = np.zeros((orbitals, orbitals))
        for i, (p, q) in enumerate(pair_index(orbitals)):
            if p == q:
                form[p, p] = complement[i, k]
            else:
                form[p, q] = form[q, p] = complement[i, k] / 2
        forms.append(form)
    return forms


def slide(chi, coupling, starts=SLIDE_STARTS, seed=0):
    """Minimise lambda over the fibre through ``(chi, coupling)``.

    Returns the lowest-lambda point whose reconstructed tensor is within
    ``TENSOR_TOLERANCE`` of the start's, plus every attempt for the record.
    Only meaningful for ``M < P``: at ``M >= P`` the span is everything and
    ``Z`` is no longer determined by the columns.
    """
    orbitals, rank = chi.shape
    if rank >= orbitals * (orbitals + 1) // 2:
        raise ValueError("slide() needs M < P; at M >= P the coupling is free too")
    target = columns(chi) @ coupling @ columns(chi).T
    forms = complement_forms(chi)

    def resolve(flat):
        c = flat.reshape(orbitals, rank)
        pseudo = np.linalg.pinv(columns(c))
        return c, pseudo @ target @ pseudo.T

    def smoothed(flat):
        c, z = resolve(flat)
        weight = np.sum(c**2, axis=0)
        return 0.5 * float(np.sum(np.sqrt(z**2 + SMOOTHING**2) * np.outer(weight, weight)))

    def constraints(flat):
        c = flat.reshape(orbitals, rank)
        inside = [c[:, m] @ form @ c[:, m] for form in forms for m in range(rank)]
        return np.concatenate([inside, np.sum(c**2, axis=0) - 1.0])

    rng = np.random.default_rng(seed)
    unit = chi / np.linalg.norm(chi, axis=0)
    attempts, best = [], None
    for index in range(starts):
        start = unit.ravel() if index == 0 else (
            unit + 0.05 * index * rng.normal(size=unit.shape)).ravel()
        # back onto the constraint set before optimising
        start = minimize(lambda x: float(np.sum(constraints(x) ** 2)), start,
                         method="BFGS", options={"gtol": 1e-14, "maxiter": 5000}).x
        result = minimize(smoothed, start, method="SLSQP",
                          constraints=[{"type": "eq", "fun": constraints}],
                          options={"maxiter": 3000, "ftol": 1e-13})
        c, z = resolve(result.x)
        moved = float(np.max(np.abs(columns(c) @ z @ columns(c).T - target)))
        attempt = {"one_norm": one_norm_exact(c, z), "tensor_change": moved,
                   "constraint": float(np.max(np.abs(constraints(result.x)))),
                   "success": bool(result.success)}
        attempts.append(attempt)
        if moved < TENSOR_TOLERANCE and (best is None or attempt["one_norm"] < best[0]):
            best = (attempt["one_norm"], c, z, moved)
    return best, attempts


def regime(orbitals, rank):
    edge = orbitals * (orbitals + 1) // 2 - orbitals + 1
    return "inside" if rank < edge else "boundary" if rank == edge else "slides"


def rank_one_points(chi, starts=POINT_SEARCH_STARTS, seed=0):
    """Every real unit ``x`` (up to sign) with ``x x^T`` in the span of the columns."""
    from scipy.optimize import least_squares

    forms = complement_forms(chi)
    orbitals = chi.shape[0]

    def equations(x):
        return np.array([x @ form @ x for form in forms] + [x @ x - 1.0])

    rng = np.random.default_rng(seed)
    points = []
    for _ in range(starts):
        found = least_squares(equations, rng.normal(size=orbitals),
                              xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.max(np.abs(found.fun)) > 1e-12:
            continue
        x = found.x / np.linalg.norm(found.x)
        x = x * np.sign(x[np.argmax(np.abs(x))])
        if not any(np.linalg.norm(x - p) < 1e-7 for p in points):
            points.append(x)
    return points


def alternatives(chi, coupling):
    """On the boundary: every exact representation built from the rank-one points.

    Returns the lowest-lambda one within ``TENSOR_TOLERANCE``, the number of
    real points found, and every subset's lambda for the record.
    """
    from itertools import combinations

    orbitals, rank = chi.shape
    target = columns(chi) @ coupling @ columns(chi).T
    points = rank_one_points(chi)
    options, best = [], None
    for subset in combinations(range(len(points)), rank):
        candidate = np.array([points[i] for i in subset]).T
        block = columns(candidate)
        if np.linalg.matrix_rank(block, tol=1e-10) < rank:
            continue
        pseudo = np.linalg.pinv(block)
        z = pseudo @ target @ pseudo.T
        moved = float(np.max(np.abs(block @ z @ block.T - target)))
        value = one_norm_exact(candidate, z)
        options.append({"one_norm": value, "tensor_change": moved})
        if moved < TENSOR_TOLERANCE and (best is None or value < best[0]):
            best = (value, candidate, z, moved)
    return best, len(points), options


def lower_one_norm(chi, coupling):
    """Slide, enumerate or leave alone, by regime; same return shape throughout."""
    kind = regime(*chi.shape)
    if kind == "boundary":
        best, count, options = alternatives(chi, coupling)
        return best, {"regime": kind, "rank_one_points": count, "options": options}
    best, attempts = slide(chi, coupling)
    record = {"regime": kind, "attempts": attempts}
    if kind == "inside":
        record["rank_one_points"] = len(rank_one_points(chi))
    return best, record


def starts_like_result_80(one_body, two_body, rank, repeats):
    orbitals = one_body.shape[0]
    base = thc_candidates(one_body, two_body)[:rank].T.copy()
    if base.shape[1] < rank:
        pad = np.random.default_rng(0).normal(scale=1e-3,
                                              size=(orbitals, rank - base.shape[1]))
        base = np.hstack([base, pad])
    rng = np.random.default_rng(START_SEED)
    return [base if repeat == 0 else base + rng.normal(scale=0.1 * repeat, size=base.shape)
            for repeat in range(repeats)]


def spread(values):
    return float(max(values) / min(values))


def measure(name, rank, repeats=REPEATS):
    problem = build_molecule(name)
    one_body, two_body, _ = molecular_integrals(problem)
    reference = energy_of(problem, one_body, two_body)
    starts = starts_like_result_80(one_body, two_body, rank, repeats)

    arms = {"unpenalised": [], "lowered": [], "penalised": []}
    for index, start in enumerate(starts):
        began = time.perf_counter()
        plain = _fit_once(one_body, two_body, rank, start, 0.0, MAX_ITERATIONS)
        penal = _fit_once(one_body, two_body, rank, start, PENALTY, MAX_ITERATIONS)
        best, record = lower_one_norm(plain["chi"], plain["coupling"])

        plain_error = abs(energy_of(problem, one_body, plain["two_body"]) - reference)
        penal_error = abs(energy_of(problem, one_body, penal["two_body"]) - reference)
        slid_tensor = thc_tensor(best[1], best[2])
        slid_error = abs(energy_of(problem, one_body, slid_tensor) - reference)

        arms["unpenalised"].append({"one_norm": plain["one_norm"], "error": plain_error,
                                    "residual": plain["residual"],
                                    "converged": plain["converged_optimiser"]})
        arms["penalised"].append({"one_norm": penal["one_norm"], "error": penal_error,
                                  "residual": penal["residual"],
                                  "converged": penal["converged_optimiser"]})
        arms["lowered"].append({"one_norm": best[0], "error": slid_error,
                             "tensor_change": best[3],
                             "tensor_change_full": float(np.max(np.abs(
                                 slid_tensor - plain["two_body"]))),
                             **record})
        print(f"    start {index}: lambda {plain['one_norm']:8.4f} -> lowered {best[0]:8.4f}"
              f" (tensor moved {best[3]:.1e})  penalised {penal['one_norm']:8.4f}   "
              f"err {plain_error:.2e} / {slid_error:.2e} / {penal_error:.2e}"
              f"  ({time.perf_counter() - began:.0f}s)", flush=True)

    summary = {}
    for arm, rows in arms.items():
        norms = [r["one_norm"] for r in rows]
        errors = [r["error"] for r in rows]
        summary[arm] = {
            "median_one_norm": float(np.median(norms)),
            "one_norm_spread": spread(norms),
            "median_error": float(np.median(errors)),
            "inside_chemical_accuracy": int(sum(e < CHEMICAL_ACCURACY for e in errors)),
        }
    return {"molecule": name, "rank": rank, "orbitals": one_body.shape[0],
            "regime": regime(one_body.shape[0], rank),
            "repeats": repeats, "arms": arms, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--cases", default=",".join(f"{n}:{m}" for n, m in CASES),
                    help="comma-separated molecule:rank pairs")
    args = ap.parse_args()
    cases = [(spec.split(":")[0], int(spec.split(":")[1])) for spec in args.cases.split(",")]

    print("=== experiment 035 :: lowering THC's lambda without moving the tensor ===")
    print("Result 80's protocol (8 starts, seed 4242); each unpenalised fit is then")
    print("moved to the lowest-lambda exact representation. Energy errors: unpenalised /")
    print("lowered / penalised.\n")

    results = []
    for name, rank in cases:
        print(f"--- {name} at M={rank} ---", flush=True)
        row = measure(name, rank, args.repeats)
        results.append(row)
        print(f"  {'arm':<13}{'median lambda':>15}{'spread':>9}{'median err':>13}{'in chem':>9}")
        for arm, s in row["summary"].items():
            print(f"  {arm:<13}{s['median_one_norm']:>15.4f}{s['one_norm_spread']:>9.3f}"
                  f"{s['median_error']:>13.2e}{s['inside_chemical_accuracy']:>6}/{args.repeats}")
        # after every case, not once at the end: the first full run lost its H6
        # arm to a container restart two starts in, with nothing on disk
        path = save(results)
        print(f"  saved {len(results)} case(s) -> {path}", flush=True)
    return 0


def save(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp035_thc_fibre.json"
    with open(path, "w") as fh:
        json.dump({"cases": results, "penalty": PENALTY,
                   "tensor_tolerance": TENSOR_TOLERANCE}, fh, indent=2, default=float)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
