"""Experiment 005 -- which excitation ansatz is both expressive and shallow enough?

The central obstruction found in session 1 (RESEARCH_LOG Result 10): a
hardware-efficient ansatz has *exactly zero* gradient at the Hartree-Fock
determinant at any depth, because its first-order response reaches only single
excitations and Brillouin's theorem decouples those.  Correlation energy is in
the doubles.  So only coupled-cluster-family ansätze can move -- and those are
deep enough that device noise destroys them (Result 12).

This measures the trade directly, for each candidate:

* ``|grad|`` at Hartree-Fock -- can the optimiser move at all?
* best exact-optimisation error -- is it expressive enough to matter?
* two-qubit gates transpiled to a real device, and the resulting fidelity --
  will it survive?

A candidate is only interesting if it clears all three.  So far nothing does:
UCCSD reaches chemical accuracy at ~1350 two-qubit gates (0.01% fidelity on
Heron), PUCCD survives better but stalls at 1.6e-2 Ha.

Run:  python -m experiments.exp005_excitation_ansatz --molecule H4
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
from scipy.optimize import minimize

from qres.ansatz import build_ansatz
from qres.bench import RESULTS_DIR
from qres.noise import make_environment
from qres.problems.chemistry import CHEMICAL_ACCURACY_HA, build_molecule
from qres.resources import circuit_duration_seconds

MOLECULE_KWARGS = {
    "H2": {},
    "H4": {},
    "LiH": {},
    "BeH2": dict(active_electrons=4, active_orbitals=6),
}

DEFAULT_ANSATZE = (
    "hea:2:linear:cry,hea:4:linear:cry,"
    "puccd,puccd:2,puccd:3,succd,succd:2,puccsd,ucc-d:1,ucc-d:2,uccsd"
)


def device_fidelity(circuit, env) -> dict:
    """Rough surviving signal: gate infidelity times T1 decay over the duration.

    Deliberately crude -- it is a screen, not a simulation.  A candidate that
    scores 1e-4 here does not need a careful study to be ruled out.
    """
    isa = env.prepare(circuit, optimization_level=3)
    target = env.backend.target
    two_q = sum(
        1 for inst in isa.data if len(inst.qubits) == 2 and inst.operation.name != "barrier"
    )
    errors = [
        prop.error
        for name, qargs_map in target.items()
        if name not in ("delay", "reset", "barrier", "measure")
        for qargs, prop in qargs_map.items()
        if prop is not None and prop.error is not None and qargs and len(qargs) == 2
    ]
    t1s = [q.t1 for q in target.qubit_properties if q is not None and q.t1]
    duration = circuit_duration_seconds(isa, target)

    gate_fidelity = (1 - float(np.median(errors))) ** two_q
    t1_fidelity = float(np.exp(-duration / np.median(t1s)) ** circuit.num_qubits)
    return {
        "two_qubit_gates": two_q,
        "depth": isa.depth(),
        "duration_us": duration * 1e6,
        "gate_fidelity": gate_fidelity,
        "t1_fidelity": t1_fidelity,
        "total_fidelity": gate_fidelity * t1_fidelity,
    }


def gradient_norm(circuit, hamiltonian) -> tuple[float, int]:
    """Finite-difference gradient at theta = 0 -- the Hartree-Fock reference."""
    n = circuit.num_parameters

    def energy(x):
        return float(Statevector(circuit.assign_parameters(x)).expectation_value(hamiltonian).real)

    grad = np.zeros(n)
    h = 1e-5
    base = np.zeros(n)
    for i in range(n):
        plus, minus = base.copy(), base.copy()
        plus[i] += h
        minus[i] -= h
        grad[i] = (energy(plus) - energy(minus)) / (2 * h)
    return float(np.linalg.norm(grad)), int((np.abs(grad) > 1e-8).sum())


def best_exact_energy(circuit, hamiltonian, restarts: int = 1, maxiter: int = 300) -> float:
    def energy(x):
        return float(Statevector(circuit.assign_parameters(x)).expectation_value(hamiltonian).real)

    best = minimize(energy, np.zeros(circuit.num_parameters), method="BFGS",
                    options={"maxiter": maxiter}).fun
    for seed in range(restarts):
        start = np.random.default_rng(seed).normal(0, 0.2, circuit.num_parameters)
        best = min(best, minimize(energy, start, method="BFGS", options={"maxiter": maxiter}).fun)
    return float(best)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H4")
    ap.add_argument("--device", default="heron")
    ap.add_argument("--ansatze", default=DEFAULT_ANSATZE)
    ap.add_argument("--maxiter", type=int, default=300)
    args = ap.parse_args()

    problem = build_molecule(args.molecule, **MOLECULE_KWARGS[args.molecule])
    env = make_environment(args.device)

    print(f"=== experiment 005 :: {args.molecule} ({problem.num_qubits}q) "
          f"on {env.backend.name} ===")
    print(f"FCI={problem.fci_energy:.6f}  Ecorr={problem.correlation_energy:.5f}  "
          f"chemical accuracy={CHEMICAL_ACCURACY_HA:.1e}")
    print("a candidate must clear all three: gradient at HF, accuracy, fidelity\n")

    header = (
        f"{'ansatz':<20}{'params':>7}{'2q':>6}{'|grad@0|':>10}"
        f"{'exact_err':>11}{'chem?':>6}{'fidelity':>10}{'verdict':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for spec in args.ansatze.split(","):
        t0 = time.perf_counter()
        try:
            circuit = build_ansatz(spec, problem, env)
            grad, nonzero = gradient_norm(circuit, problem.hamiltonian)
            energy = best_exact_energy(circuit, problem.hamiltonian, maxiter=args.maxiter)
            error = abs(energy - problem.fci_energy)
            device = device_fidelity(circuit, env)

            moves = grad > 1e-8
            accurate = error < CHEMICAL_ACCURACY_HA
            survives = device["total_fidelity"] > 0.5
            verdict = "PASS" if (moves and accurate and survives) else (
                "stuck" if not moves else ("inexact" if not accurate else "noisy")
            )
            rows.append({"ansatz": spec, "parameters": circuit.num_parameters,
                         "gradient_norm": grad, "nonzero_gradients": nonzero,
                         "exact_error": error, "verdict": verdict, **device})
            print(
                f"{spec:<20}{circuit.num_parameters:>7}{device['two_qubit_gates']:>6}"
                f"{grad:>10.3e}{error:>11.3e}{('YES' if accurate else 'no'):>6}"
                f"{device['total_fidelity']:>10.4f}{verdict:>9}"
                f"   ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        except Exception as exc:  # a family may not exist for this problem
            print(f"{spec:<20} FAILED: {type(exc).__name__}: {str(exc)[:50]}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"exp005_{args.molecule}_{env.backend.name}.json"
    with open(path, "w") as fh:
        json.dump({"molecule": args.molecule, "device": env.backend.name, "rows": rows}, fh, indent=2)
    print(f"\nverdicts: stuck = zero gradient at HF, inexact = cannot reach chemical "
          f"accuracy,\n          noisy = fidelity below 0.5, PASS = clears all three")
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
