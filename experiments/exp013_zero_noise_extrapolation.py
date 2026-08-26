"""Experiment 013 -- does zero-noise extrapolation beat spending the same shots plainly?

Result 55 established the target.  On a Heron-class noise model the error is
*bias*: 16x the shots buys 4.3%, and the measured bias equals the median error
to three digits.  Every variance-side improvement in this repository is
therefore optimising the wrong quantity there, and mitigation is the only tool
that attacks the right one.

That also makes the comparison cheap to state honestly.  ZNE buys accuracy with
extra circuit executions, so it is usually reported against an unmitigated run
at the *same shots per circuit*, which quietly gives it three times the budget.
Here every arm is charged the same **total** shots: the unmitigated arm gets all
of them, and ZNE splits its allowance across its scale factors.  If mitigation
wins under that rule it is a real win.

Numbers to beat, from Result 55 on H4 at 400k shots under `heron`, measured
from the Hartree-Fock reference where the exact answer is known:

    general commuting   6.9e-2 Hartree
    qubit-wise commuting 5.1e-2 Hartree

chemical accuracy being 1.6e-3, i.e. both are ~35x out of range before any
ansatz or optimisation is involved.

Run:  python -m experiments.exp013_zero_noise_extrapolation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit.quantum_info import Statevector

from qres.ansatz import hartree_fock_state
from qres.bench import RESULTS_DIR
from qres.estimator import ShotEstimator
from qres.mitigation import readout_mitigated_energy, zne_energy
from qres.noise import make_environment
from qres.problems.chemistry import build_molecule

#: every arm is charged this many shots in total, mitigation overhead included
DEFAULT_SHOTS = 300_000


def exact_energy(problem) -> float:
    state = Statevector(hartree_fock_state(problem.hf_bitstring))
    return float(np.real(state.expectation_value(problem.hamiltonian)))


def unmitigated(problem, environment, shots, seed, grouping):
    ansatz = hartree_fock_state(problem.hf_bitstring)
    estimator = ShotEstimator(
        problem.hamiltonian,
        ansatz,
        make_environment(environment, seed=seed),
        grouping=grouping,
        allocation="uniform",
    )
    return estimator.estimate([], shots).value


def mitigated(problem, environment, shots, seed, grouping, scales, method):
    ansatz = hartree_fock_state(problem.hf_bitstring)
    result = zne_energy(
        problem.hamiltonian,
        ansatz,
        make_environment(environment, seed=seed),
        params=[],
        total_shots=shots,
        scales=scales,
        method=method,
        grouping=grouping,
        allocation="uniform",
    )
    return result.value


def readout(problem, environment, shots, seed, grouping, fraction):
    ansatz = hartree_fock_state(problem.hf_bitstring)
    return readout_mitigated_energy(
        problem.hamiltonian,
        ansatz,
        make_environment(environment, seed=seed),
        params=[],
        total_shots=shots,
        calibration_fraction=fraction,
        grouping=grouping,
        allocation="uniform",
    ).value


#: (label, kind, argument).  Every arm is charged the same total shots.
ARMS = (
    ("unmitigated", "plain", None),
    ("ZNE linear (1,3)", "zne", ((1, 3), "linear")),
    ("ZNE linear (1,3,5)", "zne", ((1, 3, 5), "linear")),
    ("ZNE richardson (1,3,5)", "zne", ((1, 3, 5), "richardson")),
    ("readout (10% calib)", "readout", 0.10),
    ("readout (25% calib)", "readout", 0.25),
)


def run(problem, environment, shots, seeds, grouping):
    exact = exact_energy(problem)
    rows = []

    for label, kind, argument in ARMS:
        errors, values = [], []
        started = time.perf_counter()
        for seed in range(seeds):
            if kind == "plain":
                value = unmitigated(problem, environment, shots, 5000 + seed, grouping)
            elif kind == "zne":
                scales, method = argument
                value = mitigated(
                    problem, environment, shots, 5000 + seed, grouping, scales, method
                )
            else:
                value = readout(
                    problem, environment, shots, 5000 + seed, grouping, argument
                )
            values.append(value)
            errors.append(abs(value - exact))
        seconds = time.perf_counter() - started

        rows.append(
            {
                "arm": label,
                "median_error": float(np.median(errors)),
                "iqr": float(np.subtract(*np.percentile(errors, [75, 25]))),
                "bias": float(np.mean(values) - exact),
                "seconds": seconds / max(seeds, 1),
            }
        )
    return rows, exact


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default="H4")
    ap.add_argument("--environment", default="heron")
    ap.add_argument("--grouping", default="qwc")
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--seeds", type=int, default=16)
    args = ap.parse_args()

    print("=== experiment 013 :: zero-noise extrapolation at matched total shots ===")
    print(f"{args.shots:,} shots per estimate for EVERY arm, {args.seeds} seeds, "
          f"{args.environment}, {args.grouping} grouping\n")

    all_rows = []
    for name in args.molecules.split(","):
        problem = build_molecule(name)
        rows, exact = run(problem, args.environment, args.shots, args.seeds, args.grouping)

        print(f"--- {name} (exact {exact:.6f}) ---")
        header = f"{'arm':<24}{'median error':>14}{'IQR':>11}{'bias':>12}{'vs plain':>10}{'s':>7}"
        print(header)
        print("-" * len(header))

        baseline = rows[0]["median_error"]
        for row in rows:
            ratio = row["median_error"] / baseline if baseline else float("nan")
            print(
                f"{row['arm']:<24}{row['median_error']:>14.3e}{row['iqr']:>11.2e}"
                f"{row['bias']:>12.3e}{ratio:>10.2f}{row['seconds']:>7.1f}",
                flush=True,
            )
            all_rows.append({"molecule": name, "ratio": ratio, **row})

        best = min(rows[1:], key=lambda r: r["median_error"], default=None)
        if best is not None:
            change = best["median_error"] / baseline
            verdict = "helps" if change < 1 else "does NOT help"
            print(
                f"\nbest mitigation {verdict}: {best['arm']} at {change:.2f}x "
                f"the unmitigated error ({best['median_error']:.3e} against "
                f"{baseline:.3e})"
            )
            print(f"chemical accuracy is 1.6e-3 -- still "
                  f"{best['median_error'] / 1.6e-3:.0f}x away\n", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp013_zero_noise_extrapolation.json"
    with open(path, "w") as fh:
        json.dump({"shots": args.shots, "seeds": args.seeds,
                   "environment": args.environment, "rows": all_rows}, fh, indent=2)
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
