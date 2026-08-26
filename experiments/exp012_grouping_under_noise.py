"""Experiment 012 -- does general-commuting grouping still win once its basis
changes are charged for on a noisy device?

Result 38 ranked measurement schemes on total estimator variance under Neyman
allocation, and experiment 011 confirmed that ranking holds and *widens* out to
2 949 terms.  But that metric is structurally blind to the thing that decides it
on hardware.

Qubit-wise commuting groups need only single-qubit rotations to reach their
measurement basis.  General-commuting groups need a **Clifford diagonalisation**
-- entangling gates, on a device where entangling gates are the error source.
The variance formula counts none of that.  So the scheme that wins the formula
by 9x on H2O could still lose the measurement, and the only way to find out is
to run both through a real device noise model at matched shots.

Both arms are charged the same total shots and both are compared against the
exact expectation value of the same state, so any difference is the measurement
scheme and nothing else.

Run:  python -m experiments.exp012_grouping_under_noise
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
from qres.noise import make_environment
from qres.problems.chemistry import build_molecule

SCHEMES = ("commuting", "qwc")


def two_qubit_gates(circuit) -> int:
    return sum(count for name, count in circuit.count_ops().items()
               if name in {"cx", "cz", "ecr", "cy", "swap", "rzz"})


def measure(problem, scheme: str, environment, shots: int, seeds: int):
    """Median absolute error over ``seeds`` independent estimates."""
    ansatz = hartree_fock_state(problem.hf_bitstring)
    exact = float(
        np.real(Statevector(ansatz).expectation_value(problem.hamiltonian))
    )

    errors, wall = [], []
    circuit_depth = two_qubits = groups = 0

    for seed in range(seeds):
        env = make_environment(environment, seed=1000 + seed)
        estimator = ShotEstimator(
            problem.hamiltonian, ansatz, env,
            grouping=scheme, allocation="uniform",
        )
        if seed == 0:
            groups = len(estimator.groups)
            # what the basis changes actually cost, after transpilation
            circuit_depth = max((c.depth() for c in estimator._isa), default=0)
            two_qubits = max((two_qubit_gates(c) for c in estimator._isa), default=0)

        started = time.perf_counter()
        result = estimator.estimate([], shots)
        wall.append(time.perf_counter() - started)
        errors.append(abs(result.value - exact))

    return {
        "scheme": scheme,
        "groups": groups,
        "max_depth": circuit_depth,
        "max_two_qubit_gates": two_qubits,
        "median_error": float(np.median(errors)),
        "iqr": float(np.subtract(*np.percentile(errors, [75, 25]))),
        "median_seconds": float(np.median(wall)),
        "exact": exact,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default="H4,LiH,H2O")
    ap.add_argument("--environments", default="ideal,heron")
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=24)
    args = ap.parse_args()

    print("=== experiment 012 :: grouping under a real device noise model ===")
    print(f"{args.shots:,} shots per estimate, {args.seeds} seeds, "
          f"median absolute error against the exact value\n")

    header = (
        f"{'molecule':<8}{'device':<9}{'scheme':<11}{'groups':>7}{'depth':>7}"
        f"{'2q':>6}{'median err':>12}{'IQR':>11}{'vs QWC':>9}{'s':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in args.molecules.split(","):
        problem = build_molecule(name)
        for environment in args.environments.split(","):
            measured = {}
            for scheme in SCHEMES:
                measured[scheme] = measure(
                    problem, scheme, environment, args.shots, args.seeds
                )

            baseline = measured["qwc"]["median_error"]
            for scheme in SCHEMES:
                row = measured[scheme]
                ratio = row["median_error"] / baseline if baseline else float("nan")
                print(
                    f"{name:<8}{environment:<9}{scheme:<11}{row['groups']:>7}"
                    f"{row['max_depth']:>7}{row['max_two_qubit_gates']:>6}"
                    f"{row['median_error']:>12.3e}{row['iqr']:>11.2e}"
                    f"{ratio:>9.2f}{row['median_seconds']:>7.1f}",
                    flush=True,
                )
                rows.append({"molecule": name, "environment": environment,
                             "ratio_vs_qwc": ratio, **row})
            print(flush=True)

    print("A ratio below 1.00 means general-commuting grouping is the better")
    print("measurement; above 1.00 means its Clifford basis changes cost more")
    print("under device noise than the variance reduction is worth.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp012_grouping_under_noise.json"
    with open(path, "w") as fh:
        json.dump({"shots": args.shots, "seeds": args.seeds, "rows": rows}, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
