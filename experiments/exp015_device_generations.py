"""Experiment 015 -- how much better does the hardware have to get?

Results 60 and 61 measured the wall and priced it: a two-qubit gate costs ~0.1%
of the observable's scale, so the budget is ``required accuracy / 0.001`` -- under
one gate for chemistry, against the 665-1300 an ansatz needs.  Every conclusion
in that chain was measured on a single device model (`fake_torino`, Heron r1).

That leaves the question the whole project has been circling: **is this a fact
about quantum computing, or about this generation of hardware?**  The fake
provider ships calibration snapshots from real devices spanning several
generations, with median two-qubit error rates from 0.0013 to 0.044 -- a 34x
range.  Measuring the per-gate cost across that range turns "the hardware is not
good enough" into a number: how much better it must get, and whether the trend
is heading there.

Method is Result 60's, unchanged: the Hartree-Fock reference padded with
cancelling ``CX CX`` pairs, so the ideal state never moves and only the gate
count does.  The slope of error against gate count is the per-gate cost, and it
is fitted per device and compared against that device's own calibrated
two-qubit error rate.

Run:  python -m experiments.exp015_device_generations
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
from qres.mitigation import readout_mitigated_energy
from qres.noise import device_environment
from qres.problems.chemistry import build_molecule

from experiments.exp014_gate_budget import CHEMICAL_ACCURACY, padded_reference

#: spanning the generations the fake provider ships, best-calibrated first
DEVICES = (
    "fake_boston",      # 156q, median 2q error 0.00127
    "fake_aachen",      # 156q, 0.00154
    "fake_marrakesh",   # 156q, 0.00329
    "fake_fez",         # 156q, 0.00390
    "fake_torino",      # 133q Heron r1 -- every earlier result in this log
    "fake_brisbane",    # 127q Eagle r3
    "fake_kolkata",     # 27q Falcon r5.11
    "fake_cambridge",   # 28q, 0.03016
)


def device_error_rates(name: str) -> tuple[float, float]:
    """Median two-qubit gate error and readout error from the calibration."""
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    backend = next(b for b in FakeProviderForBackendV2().backends() if b.name == name)
    target = backend.target

    gate_errors: list[float] = []
    for gate in ("cz", "ecr", "cx"):
        if gate in target.operation_names:
            gate_errors = [
                properties.error
                for properties in target[gate].values()
                if properties is not None and properties.error is not None
            ]
            if gate_errors:
                break

    readout = [
        target["measure"][key].error
        for key in target["measure"]
        if target["measure"][key] is not None and target["measure"][key].error is not None
    ] if "measure" in target.operation_names else []

    return float(np.median(gate_errors)), float(np.median(readout)) if readout else float("nan")


def per_gate_cost(problem, device: str, pair_counts, shots: int, seeds: int) -> dict:
    """Fit error against two-qubit gate count on one device."""
    exact = float(
        np.real(
            Statevector(hartree_fock_state(problem.hf_bitstring)).expectation_value(
                problem.hamiltonian
            )
        )
    )

    gates, errors = [], []
    for pairs in pair_counts:
        circuit = padded_reference(problem, pairs)
        values = []
        for seed in range(seeds):
            environment = device_environment(device, seed=17000 + seed)
            values.append(
                readout_mitigated_energy(
                    problem.hamiltonian, circuit, environment, params=[],
                    total_shots=shots, calibration_fraction=0.05,
                    calibration="tensored", grouping="qwc", allocation="uniform",
                ).value
            )
        gates.append(2 * pairs)
        errors.append(float(np.median(np.abs(np.array(values) - exact))))

    # slope through the non-zero points; the intercept is the gate-free floor
    slope = float(np.polyfit(gates[1:], errors[1:], 1)[0]) if len(gates) > 2 else (
        (errors[-1] - errors[0]) / (gates[-1] - gates[0])
    )
    return {"gate_counts": gates, "errors": errors, "per_gate": slope, "floor": errors[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H4")
    ap.add_argument("--devices", default=",".join(DEVICES))
    ap.add_argument("--pairs", default="0,4,16")
    ap.add_argument("--shots", type=int, default=60_000)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    problem = build_molecule(args.molecule)
    pair_counts = [int(p) for p in args.pairs.split(",")]

    print("=== experiment 015 :: the per-gate cost across device generations ===")
    print(f"{args.molecule}, cancelling-CX padding, {args.shots:,} shots, "
          f"{args.seeds} seeds, readout-mitigated\n")

    header = (
        f"{'device':<18}{'2q error':>10}{'readout':>9}{'floor':>11}"
        f"{'per gate':>11}{'budget':>9}{'s':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    path = RESULTS_DIR / "exp015_device_generations.json"
    for device in args.devices.split(","):
        try:
            gate_error, readout_error = device_error_rates(device)
        except StopIteration:
            print(f"{device:<18}  not available in this fake provider", flush=True)
            continue

        started = time.perf_counter()
        try:
            fit = per_gate_cost(problem, device, pair_counts, args.shots, args.seeds)
        except Exception as exc:
            print(f"{device:<18}  failed: {str(exc)[:44]}", flush=True)
            continue
        seconds = time.perf_counter() - started

        budget = CHEMICAL_ACCURACY / fit["per_gate"] if fit["per_gate"] > 0 else float("inf")
        print(
            f"{device:<18}{gate_error:>10.5f}{readout_error:>9.5f}{fit['floor']:>11.3e}"
            f"{fit['per_gate']:>11.3e}{budget:>9.2f}{seconds:>7.0f}",
            flush=True,
        )
        rows.append({"device": device, "two_qubit_error": gate_error,
                     "readout_error": readout_error, "budget_gates": budget, **fit})

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"molecule": args.molecule, "shots": args.shots,
                       "seeds": args.seeds, "rows": rows}, fh, indent=2)

    if len(rows) >= 3:
        errors = np.array([r["two_qubit_error"] for r in rows])
        costs = np.array([r["per_gate"] for r in rows])
        usable = costs > 0
        if usable.sum() >= 3:
            exponent, intercept = np.polyfit(np.log(errors[usable]), np.log(costs[usable]), 1)
            print(f"\nper-gate cost ~ (2q error)^{exponent:.2f}")
            correlation = np.corrcoef(np.log(errors[usable]), np.log(costs[usable]))[0, 1]
            print(f"correlation of logs: {correlation:.3f}")

            # what 2q error would put 665 gates (batched ADAPT) inside budget?
            needed_cost = CHEMICAL_ACCURACY / 665
            needed_error = float(np.exp((np.log(needed_cost) - intercept) / exponent))
            best = errors.min()
            print(f"\nBatched ADAPT on {args.molecule} needs 665 two-qubit gates.")
            print(f"Fitting inside chemical accuracy needs a per-gate cost of "
                  f"{needed_cost:.2e},")
            print(f"which this fit puts at a two-qubit error rate of {needed_error:.2e}.")
            print(f"The best device measured here is {best:.5f} -- "
                  f"a factor of {best / needed_error:,.0f} away.")

    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
