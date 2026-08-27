"""Experiment 014 -- how many two-qubit gates does a Heron-class device actually buy?

Result 59 measured the constraint but not the number: a four-parameter ansatz
costing 317 two-qubit gates carried **1.4 Hartree** of bias, 26x more error than
every mitigation technique in this package recovers.  So the gate budget is what
binds, and it is worth knowing exactly how large it is.

The measurement needs a knob that changes gate count and nothing else.  Ansatz
depth does not qualify -- a deeper ansatz prepares a different state, so the
exact reference moves with the circuit.  Instead the circuit here is the
Hartree-Fock reference padded with pairs of CX gates, ``CX CX = I``, separated by
barriers so the transpiler cannot cancel them:

* the ideal state is *exactly* Hartree-Fock at every gate count (verified to
  1e-12), so one exact reference serves the whole sweep
* every added gate is a real gate on the device, carrying real error
* the bias measured is therefore purely the cost of circuit depth

The number this produces -- the gate count at which bias crosses chemical
accuracy -- is the budget any ansatz has to fit inside.  Compare against
RESEARCH_LOG Result 47: reaching chemical accuracy on H4 needs ~1300 two-qubit
gates with UCCSD, or 665 with the batched ADAPT ansatz.

Run:  python -m experiments.exp014_gate_budget
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
from qres.mitigation import readout_mitigated_energy
from qres.noise import make_environment
from qres.problems.chemistry import build_molecule

CHEMICAL_ACCURACY = 1.6e-3
TWO_QUBIT_GATES = {"cx", "cz", "ecr", "cy", "swap", "rzz"}


def padded_reference(problem, pairs: int, seed: int = 0):
    """Hartree-Fock plus ``pairs`` cancelling CX pairs, barriers between.

    Barriers are load-bearing: without them the transpiler removes ``CX CX``
    and the sweep silently measures nothing at all.
    """
    circuit = hartree_fock_state(problem.hf_bitstring)
    rng = np.random.default_rng(seed)
    for _ in range(pairs):
        control = int(rng.integers(circuit.num_qubits - 1))
        circuit.barrier()
        circuit.cx(control, control + 1)
        circuit.barrier()
        circuit.cx(control, control + 1)
    circuit.barrier()
    return circuit


def two_qubit_count(estimator) -> int:
    return max(
        (
            sum(n for gate, n in circuit.count_ops().items() if gate in TWO_QUBIT_GATES)
            for circuit in estimator._isa
        ),
        default=0,
    )


def measure(problem, circuit, environment, shots, seeds, mitigate: bool):
    exact = float(
        np.real(
            Statevector(hartree_fock_state(problem.hf_bitstring)).expectation_value(
                problem.hamiltonian
            )
        )
    )
    values, gates = [], 0
    for seed in range(seeds):
        env = make_environment(environment, seed=15000 + seed)
        if mitigate:
            values.append(
                readout_mitigated_energy(
                    problem.hamiltonian, circuit, env, params=[], total_shots=shots,
                    calibration_fraction=0.05, calibration="tensored",
                    grouping="qwc", allocation="uniform",
                ).value
            )
            if seed == 0:
                estimator = ShotEstimator(
                    problem.hamiltonian, circuit, env, grouping="qwc", allocation="uniform"
                )
                gates = two_qubit_count(estimator)
        else:
            estimator = ShotEstimator(
                problem.hamiltonian, circuit, env, grouping="qwc", allocation="uniform"
            )
            if seed == 0:
                gates = two_qubit_count(estimator)
            values.append(estimator.estimate([], shots).value)

    values = np.array(values)
    return {
        "two_qubit_gates": gates,
        "median_error": float(np.median(np.abs(values - exact))),
        "bias": float(values.mean() - exact),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H4")
    ap.add_argument("--environment", default="heron")
    ap.add_argument("--pairs", default="0,2,4,8,16,32,64,128")
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    problem = build_molecule(args.molecule)

    print("=== experiment 014 :: the two-qubit gate budget ===")
    print(f"{args.molecule} Hartree-Fock reference padded with cancelling CX pairs")
    print(f"ideal state is exactly HF at every point; {args.environment}, "
          f"{args.shots:,} shots, {args.seeds} seeds\n")

    header = (
        f"{'2q gates':>10}{'unmitigated':>14}{'mitigated':>13}"
        f"{'vs chem acc':>13}{'s':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for pairs in [int(p) for p in args.pairs.split(",")]:
        circuit = padded_reference(problem, pairs)
        started = time.perf_counter()
        plain = measure(problem, circuit, args.environment, args.shots, args.seeds, False)
        fixed = measure(problem, circuit, args.environment, args.shots, args.seeds, True)
        seconds = time.perf_counter() - started

        ratio = fixed["median_error"] / CHEMICAL_ACCURACY
        print(
            f"{plain['two_qubit_gates']:>10}{plain['median_error']:>14.3e}"
            f"{fixed['median_error']:>13.3e}{ratio:>12.1f}x{seconds:>7.0f}",
            flush=True,
        )
        rows.append(
            {
                "pairs": pairs,
                "two_qubit_gates": plain["two_qubit_gates"],
                "unmitigated_error": plain["median_error"],
                "mitigated_error": fixed["median_error"],
                "mitigated_bias": fixed["bias"],
                "vs_chemical_accuracy": ratio,
            }
        )

    over = [r for r in rows if r["mitigated_error"] > CHEMICAL_ACCURACY]
    under = [r for r in rows if r["mitigated_error"] <= CHEMICAL_ACCURACY]
    print()
    if under and over:
        last_good = max(r["two_qubit_gates"] for r in under)
        first_bad = min(r["two_qubit_gates"] for r in over)
        print(f"Chemical accuracy survives to {last_good} two-qubit gates and is lost")
        print(f"by {first_bad}, with the best mitigation this package has.")
    elif not under:
        print("Chemical accuracy is not reached at any gate count tested, "
              f"including {min(r['two_qubit_gates'] for r in rows)}.")
    else:
        print("Chemical accuracy held at every gate count tested; extend the sweep.")
    print("For scale: UCCSD on H4 needs ~1300 two-qubit gates, batched ADAPT ~665.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp014_gate_budget.json"
    with open(path, "w") as fh:
        json.dump({"molecule": args.molecule, "environment": args.environment,
                   "shots": args.shots, "seeds": args.seeds, "rows": rows}, fh, indent=2)
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
