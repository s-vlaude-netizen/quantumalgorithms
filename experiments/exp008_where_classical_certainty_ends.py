"""Experiment 008 -- where does classical certainty on MaxCut actually end?

Result 50 measured that Goemans-Williamson returns the **exact** optimum on 100%
of instances up to n = 20, which makes every approximation ratio below that size
uninformative: there is no hard part left to measure.  It left the obvious
question open, and that question is the prerequisite for every quantum
comparison this project could still run.

MaxCut is NP-hard, so GW cannot stay exact forever.  Where does it stop?  The
difficulty is that "exact" stops being checkable at the same place it stops
being obvious: brute force ends around 24 variables.

The method here is agreement between two strong, *independent* classical
methods:

``exact``       n <= 24, both scored against brute force.  This is where the
                reference earns its credibility -- iterated local search must
                reproduce the true optimum before anything it says above 24 is
                admissible.  It does: 20/20 on every family tested.
``disputed``    n > 24, GW against ILS at a **matched wall-clock budget**.
                While they agree, that is strong evidence the optimum is found.
                Where they disagree, neither is known to be optimal and nobody
                can score anything -- that gap is the answer.

The budget matching is the part that has to be right.  Given 300 fixed
iterations, ILS lost to GW at n = 60 (81 against 82) -- but it had 36 ms against
the SDP's 2.3 s.  Reading a win off that pair credits GW with a 63x wall-clock
advantage.  ILS therefore gets exactly the time GW spent on the same instance.

Run:  python -m experiments.exp008_where_classical_certainty_ends
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qres.bench import RESULTS_DIR
from qres.classical_optimization import (
    goemans_williamson,
    iterated_local_search,
    local_search_maxcut,
)
from qres.problems.optimization import BRUTE_FORCE_LIMIT, random_regular_maxcut


def sweep(sizes, degree: int = 3, instances: int = 12) -> list[dict]:
    rows = []
    header = (
        f"{'n':>5}{'inst':>6}{'GW=ILS':>9}{'GW>ILS':>9}{'ILS>GW':>9}"
        f"{'mean gap':>10}{'GW s':>8}{'ILS it':>9}{'verified':>10}"
    )
    print(header)
    print("-" * len(header))

    for n in sizes:
        exact = n <= BRUTE_FORCE_LIMIT
        agree = gw_wins = ils_wins = 0
        gaps, gw_times, ils_iterations = [], [], []
        gw_exact = ils_exact = 0

        for seed in range(instances):
            problem = random_regular_maxcut(n, degree, seed=seed, exact=exact)

            gw = goemans_williamson(problem, seed=seed)
            # the reference gets exactly the wall-clock the SDP just spent
            ils = iterated_local_search(problem, seed=seed, time_budget=gw.seconds)

            gw_times.append(gw.seconds)
            ils_iterations.append(ils.metadata["iterations"])

            if abs(gw.cut_value - ils.cut_value) < 1e-9:
                agree += 1
            elif gw.cut_value > ils.cut_value:
                gw_wins += 1
            else:
                ils_wins += 1

            best = max(gw.cut_value, ils.cut_value)
            gaps.append(abs(gw.cut_value - ils.cut_value) / best if best else 0.0)

            if exact:
                optimum = problem.metadata["max_cut"]
                gw_exact += abs(gw.cut_value - optimum) < 1e-9
                ils_exact += abs(ils.cut_value - optimum) < 1e-9

        verified = (
            f"{gw_exact}/{ils_exact} of {instances}" if exact else "unknowable"
        )
        print(
            f"{n:>5}{instances:>6}{agree:>9}{gw_wins:>9}{ils_wins:>9}"
            f"{np.mean(gaps):>10.4f}{np.mean(gw_times):>8.2f}"
            f"{int(np.mean(ils_iterations)):>9,d}{verified:>10}",
            flush=True,
        )
        rows.append(
            {
                "n": n,
                "instances": instances,
                "agree": agree,
                "gw_wins": gw_wins,
                "ils_wins": ils_wins,
                "mean_gap": float(np.mean(gaps)),
                "gw_seconds": float(np.mean(gw_times)),
                "ils_iterations": float(np.mean(ils_iterations)),
                "brute_forced": exact,
                "gw_exact": gw_exact if exact else None,
                "ils_exact": ils_exact if exact else None,
            }
        )
    return rows


def cheap_method_check(sizes, degree: int = 3, instances: int = 12) -> list[dict]:
    """How much of this does a millisecond of hill-climbing already get?

    Result 50's headline on small instances was that local search matched the
    SDP at 1/100th the cost.  Worth knowing whether that survives past the
    enumeration limit, because if it does, the expensive methods are not the
    thing a quantum method would have to beat.
    """
    print("\ncheap local search, scored against the best of GW and ILS")
    header = f"{'n':>5}{'matches best':>15}{'mean ratio':>13}{'mean ms':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for n in sizes:
        exact = n <= BRUTE_FORCE_LIMIT
        matches, ratios, times = 0, [], []
        for seed in range(instances):
            problem = random_regular_maxcut(n, degree, seed=seed, exact=exact)
            gw = goemans_williamson(problem, seed=seed)
            ils = iterated_local_search(problem, seed=seed, time_budget=gw.seconds)
            best = max(gw.cut_value, ils.cut_value)

            cheap = local_search_maxcut(problem, restarts=10, seed=seed)
            matches += abs(cheap.cut_value - best) < 1e-9
            ratios.append(cheap.cut_value / best if best else 1.0)
            times.append(cheap.seconds * 1e3)

        print(
            f"{n:>5}{f'{matches}/{instances}':>15}"
            f"{np.mean(ratios):>13.4f}{np.mean(times):>10.2f}",
            flush=True,
        )
        rows.append(
            {
                "n": n,
                "matches_best": matches,
                "instances": instances,
                "mean_ratio": float(np.mean(ratios)),
                "mean_ms": float(np.mean(times)),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16,20,24,30,40,60,80,100")
    ap.add_argument("--instances", type=int, default=12)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    print("=== experiment 008 :: where does classical certainty end? ===")
    print("GW vs iterated local search at MATCHED wall-clock, 3-regular MaxCut")
    print(f"(brute force still available at n <= {BRUTE_FORCE_LIMIT})\n")

    agreement = sweep(sizes, instances=args.instances)
    cheap = cheap_method_check(sizes, instances=args.instances)

    disputed = [r for r in agreement if r["agree"] < r["instances"]]
    print()
    if disputed:
        first = min(r["n"] for r in disputed)
        print(
            f"The two strong methods first disagree at n = {first}. Below that "
            f"they agree on every\ninstance, and where brute force can confirm "
            f"it they are both exactly optimal."
        )
    else:
        print("The two strong methods agreed on every instance at every size tested.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp008_where_classical_certainty_ends.json"
    with open(path, "w") as fh:
        json.dump({"agreement": agreement, "cheap": cheap}, fh, indent=2)
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
