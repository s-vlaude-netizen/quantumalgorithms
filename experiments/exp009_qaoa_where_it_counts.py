"""Experiment 009 -- QAOA at the sizes where a MaxCut number means anything.

Result 51 pinned down where classical certainty ends: below n = 40 two
independent classical methods are still exactly optimal, so every approximation
ratio measured there is measuring nothing.  A QAOA result has to live at
**n >= 60**, and that is exactly where statevector simulation dies (2^60).

``qres/lightcone.py`` gets past that.  At depth p, ``<Z_u Z_v>`` depends only on
the vertices within p hops of the edge, so the energy of a 1 000-vertex instance
is a sum of independent 6- or 14-qubit simulations.  It is exact, not
approximate -- verified against full statevector simulation to 9e-15.

Three questions, in the order they have to be answered:

1. **Does QAOA's expected cut beat the classical champion at n >= 60?**  The
   champion there is not Goemans-Williamson (Result 51: it never wins once at a
   matched budget) but iterated local search.
2. **Do the optimal angles transfer across instance size?**  On a 3-regular
   graph the light cone does not depend on n at all, which *predicts* that the
   optimal angles are n-independent.  If so, the expensive part of QAOA -- the
   instance-specific outer loop -- is free, and the fixed-angle literature has a
   mechanism rather than a coincidence.
3. **What would the quantum computer have to be to run it?**  An expected cut is
   free here; on hardware it is shots and gates.

Run:  python -m experiments.exp009_qaoa_where_it_counts
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
from qres.classical_optimization import iterated_local_search, local_search_maxcut
from qres.lightcone import LightConeEnergy
from qres.problems.optimization import random_regular_maxcut

#: instance the angles are trained on -- small enough that p=2 optimisation is
#: affordable, large enough to be past the region where the problem is trivial
TRAINING_SIZE = 60


def optimise_angles(problem, depth: int, starts: int = 8, seed: int = 0):
    """Best (gamma, beta) for one instance, by multi-start Nelder-Mead.

    The objective is exact and noiseless, so this is ordinary smooth
    optimisation -- none of the shot-noise machinery elsewhere in this project
    applies.  Multi-start because the QAOA landscape is famously multi-modal.
    """
    energy = LightConeEnergy(problem, depth)
    rng = np.random.default_rng(seed)
    best_value, best_params = np.inf, None

    for attempt in range(starts):
        if attempt == 0:
            # the standard shallow-ramp guess
            start = np.concatenate([
                np.linspace(0.3, 0.6, depth), np.linspace(0.5, 0.2, depth)
            ])
        else:
            start = rng.uniform(-np.pi / 2, np.pi / 2, 2 * depth)
        result = minimize(energy, start, method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
        if result.fun < best_value:
            best_value, best_params = float(result.fun), np.asarray(result.x)

    return best_params, -best_value, energy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="60,100,200,500,1000")
    ap.add_argument("--depths", default="1,2")
    ap.add_argument("--reference-seconds", type=float, default=2.0)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    depths = [int(d) for d in args.depths.split(",")]

    print("=== experiment 009 :: QAOA where a MaxCut number means anything ===")
    print("exact light-cone energies; classical champion is iterated local search\n")

    # --- 1. train the angles once, on one instance ------------------------
    print(f"training angles on one {TRAINING_SIZE}-vertex 3-regular instance")
    trained = {}
    for depth in depths:
        problem = random_regular_maxcut(TRAINING_SIZE, 3, seed=0, exact=False)
        t0 = time.perf_counter()
        params, cut, energy = optimise_angles(problem, depth)
        seconds = time.perf_counter() - t0
        trained[depth] = params
        print(
            f"  p={depth}: cut {cut:.3f}, cone {energy.max_cone} qubits, "
            f"{energy.distinct_cones} distinct, {seconds:.1f}s"
        )
        print(f"       gamma {np.round(params[:depth], 4).tolist()}  "
              f"beta {np.round(params[depth:], 4).tolist()}")
    print()

    # --- 2. apply them everywhere, and score against the classical champion --
    header = (
        f"{'n':>6}{'p':>3}{'QAOA <cut>':>12}{'ILS cut':>10}{'ratio':>9}"
        f"{'1ms local':>11}{'QAOA s':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n in sizes:
        problem = random_regular_maxcut(n, 3, seed=1, exact=False)

        reference = iterated_local_search(
            problem, seed=1, time_budget=args.reference_seconds
        )
        cheap = local_search_maxcut(problem, restarts=10, seed=1)

        for depth in depths:
            energy = LightConeEnergy(problem, depth)
            t0 = time.perf_counter()
            transferred = energy.expected_cut(trained[depth])
            seconds = time.perf_counter() - t0

            ratio = transferred / reference.cut_value
            print(
                f"{n:>6}{depth:>3}{transferred:>12.2f}{reference.cut_value:>10.0f}"
                f"{ratio:>9.4f}{cheap.cut_value / reference.cut_value:>11.4f}"
                f"{seconds:>9.2f}",
                flush=True,
            )
            rows.append(
                {
                    "n": n,
                    "depth": depth,
                    "qaoa_expected_cut": transferred,
                    "reference_cut": reference.cut_value,
                    "ratio": ratio,
                    "cheap_ratio": cheap.cut_value / reference.cut_value,
                    "qaoa_seconds": seconds,
                    "reference_seconds": reference.seconds,
                    "cheap_ms": cheap.seconds * 1e3,
                }
            )

    # --- 3. what does re-optimising per instance actually buy? --------------
    print("\nre-optimising the angles on each instance instead of transferring:")
    print(f"{'n':>6}{'p':>3}{'transferred':>13}{'re-optimised':>14}{'gain':>9}")
    transfer_rows = []
    for n in (100, 200):
        problem = random_regular_maxcut(n, 3, seed=1, exact=False)
        for depth in depths:
            energy = LightConeEnergy(problem, depth)
            base = energy.expected_cut(trained[depth])
            local_params, tuned, _ = optimise_angles(problem, depth, starts=4, seed=n)
            print(
                f"{n:>6}{depth:>3}{base:>13.3f}{tuned:>14.3f}"
                f"{(tuned - base) / base:>8.3%}",
                flush=True,
            )
            transfer_rows.append(
                {
                    "n": n,
                    "depth": depth,
                    "transferred": base,
                    "reoptimised": tuned,
                    "gain": (tuned - base) / base,
                    "angles": local_params.tolist(),
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp009_qaoa_where_it_counts.json"
    with open(path, "w") as fh:
        json.dump(
            {
                "trained_angles": {str(d): trained[d].tolist() for d in trained},
                "rows": rows,
                "transfer": transfer_rows,
            },
            fh,
            indent=2,
        )
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
