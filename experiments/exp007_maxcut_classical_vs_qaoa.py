"""Experiment 007 -- what does QAOA add over a classical MaxCut heuristic?

The chemistry side of this project has a classical anchor (RESEARCH_LOG Result
42) and the optimisation side did not, which left every QAOA number here as
unmoored as the VQE ones once were.

The comparison has a different shape, and that is why it is worth running
separately.  Molecular ground states have CCSD(T): polynomial and very accurate,
so VQE races a fast near-exact method.  MaxCut is NP-hard, so the classical
competitor is an *approximation* algorithm with a proven 0.878 ratio, and the
question becomes whether QAOA beats a **guarantee** rather than an exact answer.
That framing predicts a narrow but real opening for the quantum side.

Two parts, because the first alone cannot settle it:

``head_to_head``  QAOA against greedy / local search / Goemans-Williamson on
                  3-regular MaxCut, charging QAOA the shots it actually spends.
``hardness``      whether *any* instance family at a brute-forceable size defeats
                  the classical heuristics.  If they never fail, there is nothing
                  for QAOA to win, and an approximation ratio measured here says
                  nothing about MaxCut's hard part.

The denominator in ``hardness`` is instances **actually built**, not attempted.
Silently skipping a failed generation and dividing by the attempt count is what
turned 3 successes out of 10 into "optimal on 30% of instances" (Result 50).

Run:  python -m experiments.exp007_maxcut_classical_vs_qaoa
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
    greedy_maxcut,
    local_search_maxcut,
)
from qres.noise import make_environment
from qres.problems.optimization import (
    erdos_renyi_maxcut,
    maxcut,
    random_regular_maxcut,
)
from qres.qaoa import run_qaoa


def _weighted_dense(n: int, density: float, seed: int):
    rng = np.random.default_rng(seed)
    edges = [
        (i, j, float(rng.normal(0, 1)))
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < density
    ]
    return maxcut(edges, num_nodes=n, name=f"w{density:g}")


def head_to_head(sizes=(10, 14, 18), depths=(1, 3), seed: int = 0) -> list[dict]:
    """QAOA vs the classical baselines, with QAOA charged for its shots."""
    environment = make_environment("ideal", seed=seed)
    rows = []

    print("=== part 1 :: head to head on 3-regular MaxCut ===\n")
    for n in sizes:
        problem = random_regular_maxcut(n, 3, seed=seed)
        optimum = problem.metadata["max_cut"]
        print(f"--- n={n}, max_cut={optimum:g} ---")

        for result in (
            greedy_maxcut(problem),
            local_search_maxcut(problem, seed=seed),
            goemans_williamson(problem, seed=seed),
        ):
            print(
                f"      {result.method:<26}{result.cut_value:>7.0f}"
                f"{result.approximation_ratio:>8.4f}{result.seconds * 1e3:>13.2f} ms"
            )
            rows.append(
                {
                    "n": n,
                    "method": result.method,
                    "cut": result.cut_value,
                    "ratio": result.approximation_ratio,
                    "seconds": result.seconds,
                }
            )

        for depth in depths:
            outcome = run_qaoa(problem, reps=depth, environment=environment, seed=seed)
            # `best_sampled` is already <H> + offset for the best bitstring seen,
            # so negating it *is* the cut.  Subtracting the offset again here is
            # the Result 50 bug that pushed ratios below random guessing.
            cut = -outcome.best_sampled
            ratio = cut / optimum if optimum else float("nan")
            shots = int(outcome.ledger["shots"])
            print(
                f"      {'QAOA p=' + str(depth) + ' (best sample)':<26}{cut:>7.0f}"
                f"{ratio:>8.4f}{shots:>11,d} shots"
            )
            rows.append(
                {
                    "n": n,
                    "method": f"qaoa_p{depth}",
                    "cut": cut,
                    "ratio": ratio,
                    "shots": shots,
                }
            )
        print(flush=True)

    return rows


def hardness(sizes=(12, 16, 20), instances: int = 20) -> list[dict]:
    """Does any family at a brute-forceable size defeat the classical heuristics?"""
    families = [
        ("3-regular", lambda n, s: random_regular_maxcut(n, 3, seed=s)),
        ("5-regular", lambda n, s: random_regular_maxcut(n, 5, seed=s)),
        ("9-regular", lambda n, s: random_regular_maxcut(n, 9, seed=s)),
        ("Erdos-Renyi p=.5", lambda n, s: erdos_renyi_maxcut(n, 0.5, seed=s)),
        ("weighted p=.8", lambda n, s: _weighted_dense(n, 0.8, s)),
    ]

    print("=== part 2 :: how often is each method exactly optimal? ===")
    print("(denominator is instances actually BUILT, not attempted)\n")
    header = (
        f"{'family':<20}{'n':>4}{'built':>7}{'greedy':>9}"
        f"{'local':>8}{'GW':>8}{'worst local':>13}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for label, factory in families:
        for n in sizes:
            built = 0
            hits = {"greedy": 0, "local": 0, "gw": 0}
            worst = 1.0
            for s in range(instances):
                try:
                    problem = factory(n, s)
                except (RuntimeError, ValueError):
                    continue  # counted by omission from `built`, never divided into
                built += 1
                optimum = problem.metadata["max_cut"]
                for key, result in (
                    ("greedy", greedy_maxcut(problem)),
                    ("local", local_search_maxcut(problem, seed=s)),
                    ("gw", goemans_williamson(problem, seed=s)),
                ):
                    if abs(result.cut_value - optimum) < 1e-6:
                        hits[key] += 1
                    if key == "local" and optimum:
                        worst = min(worst, result.cut_value / optimum)
            if not built:
                continue
            rate = {k: v / built for k, v in hits.items()}
            print(
                f"{label:<20}{n:>4}{built:>7}"
                f"{rate['greedy']:>8.0%}{rate['local']:>8.0%}{rate['gw']:>8.0%}"
                f"{worst:>13.4f}",
                flush=True,
            )
            rows.append(
                {"family": label, "n": n, "built": built, "worst_local": worst, **rate}
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--skip-qaoa", action="store_true")
    args = ap.parse_args()

    head = [] if args.skip_qaoa else head_to_head()
    hard = hardness(instances=args.instances)

    exact = [r for r in hard if r["gw"] >= 1.0]
    print(
        f"\nGoemans-Williamson was exactly optimal on every instance in "
        f"{len(exact)} of {len(hard)} rows."
    )
    print(
        "Where that holds, an approximation ratio measured at these sizes is not\n"
        "measuring MaxCut's hard part -- a ~100 ms classical algorithm solves it."
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp007_maxcut_classical_vs_qaoa.json"
    with open(path, "w") as fh:
        json.dump({"head_to_head": head, "hardness": hard}, fh, indent=2)
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
