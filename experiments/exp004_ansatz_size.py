"""Experiment 004 -- how many parameters can a shot budget afford?

Session 1's recurring finding was that VQE runs here are *optimiser*-limited,
not shot-noise-limited: on H4 at a 150k shot budget the error is 5e-2 Ha while
the estimator's own standard error is ~8e-3 Ha, so measurement-side improvements
have nothing to act on.

There is a blunt reason.  A budget of 150k shots at 3072 shots per evaluation
buys 48 energy evaluations, and COBYLA needs ``n+1 = 47`` of them just to build
its initial simplex on a 46-parameter ansatz.  The optimiser never starts.

So there should be an interior optimum in ansatz size at fixed budget, for the
same reason there is one in shots-per-evaluation (experiment 002): more
parameters buy expressibility but cost convergence.  This sweeps it.

Also included for contrast: UCCSD, which has *fewer* parameters (26 on H4) but
is 100x deeper -- 1826 depth and 1096 two-qubit gates against 14 and 10 for the
hardware-efficient ansatz.  Under device noise that is not a trade, it is a
disqualification, and the numbers should show it.

Run:  python -m experiments.exp004_ansatz_size --molecule H4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit.quantum_info import Statevector

from qres.ansatz import build_ansatz
from qres.bench import Study, run_over_seeds
from qres.problems.chemistry import CHEMICAL_ACCURACY_HA, build_molecule
from qres.vqe import run_vqe

MOLECULE_KWARGS = {
    "H2": {},
    "H4": {},
    "LiH": {},
    "BeH2": dict(active_electrons=4, active_orbitals=6),
}


def _run(seed: int, molecule: str, molecule_kwargs: dict, **kwargs):
    problem = build_molecule(molecule, **molecule_kwargs)
    return run_vqe(problem, seed=seed, **kwargs)


def ansatz_facts(spec: str, problem) -> dict:
    """Parameters, depth and two-qubit count -- the hardware-relevant size."""
    circuit = build_ansatz(spec, problem)
    zero = Statevector(circuit.assign_parameters(np.zeros(circuit.num_parameters)))
    return {
        "parameters": circuit.num_parameters,
        "depth": circuit.depth(),
        "two_qubit_gates": sum(1 for inst in circuit.data if len(inst.qubits) == 2),
        "energy_at_zero": float(zero.expectation_value(problem.hamiltonian).real),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H4")
    ap.add_argument("--env", default="ideal")
    ap.add_argument("--budget", type=int, default=600_000)
    ap.add_argument("--shots", type=int, default=3072)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--optimizer", default="cobyla")
    ap.add_argument(
        "--ansatze",
        default="hea:1:linear:cry,hea:2:linear:cry,hea:3:linear:cry,hea:4:linear:cry",
        help="comma-separated ansatz specs; add 'uccsd' for the deep-circuit contrast",
    )
    args = ap.parse_args()

    problem = build_molecule(args.molecule, **MOLECULE_KWARGS[args.molecule])
    seeds = list(range(args.seeds))
    study = Study(name=f"exp004_{args.molecule}_{args.env.replace('@', '')}", seeds=seeds)

    evaluations = args.budget // args.shots
    print(f"=== experiment 004 :: {args.molecule} ({problem.num_qubits}q, "
          f"{len(problem.hamiltonian)} terms) env={args.env} ===")
    print(f"budget={args.budget:,} at {args.shots} shots/eval = {evaluations} energy evaluations")
    print(f"chemical accuracy = {CHEMICAL_ACCURACY_HA:.2e} Ha,  "
          f"correlation energy = {problem.correlation_energy:.2e} Ha\n")

    header = (
        f"{'ansatz':<22}{'params':>7}{'depth':>7}{'2q':>6}"
        f"{'evals/param':>13}{'med_err':>11}{'best':>11}"
    )
    print(header)
    print("-" * len(header))

    for spec in args.ansatze.split(","):
        facts = ansatz_facts(spec, problem)
        t0 = time.perf_counter()
        runs = run_over_seeds(
            _run,
            seeds,
            molecule=args.molecule,
            molecule_kwargs=MOLECULE_KWARGS[args.molecule],
            ansatz=spec,
            environment=args.env,
            grouping="commuting",
            allocation="adaptive",
            optimizer=args.optimizer,
            shots=args.shots,
            shot_budget=args.budget,
            maxiter=100_000,
            label=f"{args.molecule}|{spec}",
        )
        study.add(f"{args.molecule}|{spec}", runs)
        study.save()
        errors = [r.noiseless_error for r in runs]
        per_param = evaluations / max(1, facts["parameters"])
        print(
            f"{spec:<22}{facts['parameters']:>7}{facts['depth']:>7}"
            f"{facts['two_qubit_gates']:>6}{per_param:>13.1f}"
            f"{np.median(errors):>11.3e}{np.min(errors):>11.3e}"
            f"   ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    print(f"\nsaved -> {study.save()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
