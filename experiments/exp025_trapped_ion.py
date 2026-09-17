"""Experiment 025 -- what does a trapped-ion machine actually change?

Every noise result in this repository is measured on IBM calibration snapshots,
so the conclusions are superconducting conclusions: heavy-hex connectivity and
~1.3e-3 two-qubit error.  Trapped-ion machines differ on both axes, and the
question is whether the difference is a constant factor or a change of verdict.

The occasion is Quantinuum's Helios, announced with figures that are easy to
read as much stronger than they are.  What is published:

* **98 physical qubits, all-to-all connected**
* **99.921% two-qubit gate fidelity** -- so p = 7.9e-4
* **"48 fully error-corrected logical qubits"** at a 2:1 encoding via the
  Iceberg code, quoted with **99.99% state-preparation-and-measurement**
* **"50 error-detected logical qubits"** better than break-even

Three things that matter for sizing an algorithm are *not* published: a logical
two-qubit gate error rate, a logical gate count, and the post-selection
acceptance rate.  The headline "99.9% at 50 logical qubits" is the **physical**
two-qubit fidelity beside a **different** configuration's logical qubit count;
they are not a logical error rate, and nothing here can be sized on them
directly.

So this measures the two things that *are* determinable from public numbers plus
this repository's own circuits:

1. **Connectivity.**  All-to-all removes every SWAP the heavy-hex transpiler
   inserts.  This repo's gate counts -- and therefore Result 66's whole
   threshold table -- are post-routing, so the saving is real and has never been
   measured here.  Held fixed: the basis gate set, the optimisation level, the
   circuit.  The *only* difference between the two arms is the coupling map, so
   the ratio is the routing overhead and nothing else.

2. **What the two together buy against the threshold.**  Result 66 established
   that the binding quantity is (two-qubit gates) x (per-gate error), and that
   chemical accuracy needs that product below 1.6e-3.  Both factors move on a
   trapped-ion machine; the product is what decides anything.

And one thing the headline hides:

3. **Error *detection* is not error correction.**  A distance-2 code detects a
   single fault and cannot correct it, so the machine discards the shot.  The
   accepted fraction falls as (1-p)^(gates), which is a **shot multiplier**, and
   this repository already measured shots to be the binding cost of VQE.  A
   configuration described as "50 error-detected logical qubits" therefore buys
   fidelity at an exponentially growing price in runtime, and the exponent is
   the circuit's gate count.

Sources for the device figures are in the module docstring above; they are
Quantinuum's own published announcement, not a measurement made here, and are
labelled as such in the output.

Run:  python -m experiments.exp025_trapped_ion
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import transpile

from qres.ansatz import build_ansatz
from qres.bench import RESULTS_DIR
from qres.fermionic import two_qubit_count
from qres.noise import get_fake_backend
from qres.problems.chemistry import build_molecule

CHEMICAL_ACCURACY = 1.6e-3

#: the basis both arms are transpiled to, so the comparison isolates routing.
#: Transpiling one arm to a device target and the other to a generic basis would
#: confound connectivity with the native gate set -- ecr against cx -- and the
#: ratio would no longer mean what it says.
BASIS = ("rz", "sx", "x", "cx")

#: SABRE routing is stochastic; a single seed is not a gate count.
SEEDS = (0, 1, 2, 3, 4)

#: best median two-qubit error among the seven devices measured in Result 65,
#: which is `fake_boston`
IBM_BEST_TWO_QUBIT_ERROR = 1.2747211439835399e-3

#: Quantinuum Helios, published: 99.921% two-qubit gate fidelity.  Not measured
#: here, and not a *logical* error rate -- see the module docstring.
HELIOS_TWO_QUBIT_ERROR = 7.9e-4

MOLECULES = ("H2", "LiH", "H4", "BeH2", "H6")
ANSATZE = ("uccsd", "puccd")


def routed_two_qubit_count(circuit, coupling_map, seed):
    """Two-qubit gates after routing to ``coupling_map`` (None = all-to-all)."""
    isa = transpile(
        circuit,
        basis_gates=list(BASIS),
        coupling_map=coupling_map,
        optimization_level=3,
        seed_transpiler=seed,
    )
    return sum(
        1 for inst in isa.data
        if len(inst.qubits) == 2 and inst.operation.name != "barrier"
    )


def count_across_seeds(circuit, coupling_map):
    counts = [routed_two_qubit_count(circuit, coupling_map, seed) for seed in SEEDS]
    return {
        "median": float(np.median(counts)),
        "min": int(min(counts)),
        "max": int(max(counts)),
        "counts": counts,
    }


def survival(gates, error):
    """Fraction of shots a distance-2 detecting code accepts.

    Every fault is detected and the shot is thrown away, so acceptance is the
    probability that no gate faulted.  This is the cost the word "detected"
    carries and the announcement does not quote.
    """
    return float((1.0 - error) ** gates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default=",".join(MOLECULES))
    ap.add_argument("--ansatze", default=",".join(ANSATZE))
    ap.add_argument("--backend", default="fake_torino",
                    help="heavy-hex device supplying the coupling map")
    args = ap.parse_args()

    print("=== experiment 025 :: what does a trapped-ion machine actually change? ===")
    print(f"routing arm: {args.backend} coupling map; free arm: all-to-all")
    print(f"same basis {BASIS} and optimisation level in both, {len(SEEDS)} transpiler seeds\n")

    backend = get_fake_backend(args.backend)
    coupling_map = backend.coupling_map

    header = (
        f"{'molecule':>9}{'ansatz':>8}{'qubits':>8}{'heavy-hex':>11}{'all-to-all':>12}"
        f"{'routing':>9}{'x error':>9}{'total':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    error_ratio = IBM_BEST_TWO_QUBIT_ERROR / HELIOS_TWO_QUBIT_ERROR
    for name in args.molecules.split(","):
        problem = build_molecule(name)
        for spec in args.ansatze.split(","):
            started = time.perf_counter()
            try:
                circuit = build_ansatz(spec, problem)
            except Exception as exc:                       # noqa: BLE001
                print(f"{name:>9}{spec:>8}   skipped: {type(exc).__name__}: {exc}")
                continue

            hexagonal = count_across_seeds(circuit, coupling_map)
            free = count_across_seeds(circuit, None)
            routing = hexagonal["median"] / max(free["median"], 1)
            total = routing * error_ratio

            row = {
                "molecule": name,
                "ansatz": spec,
                "qubits": int(circuit.num_qubits),
                "parameters": int(circuit.num_parameters),
                "heavy_hex": hexagonal,
                "all_to_all": free,
                "routing_factor": float(routing),
                "error_factor": float(error_ratio),
                "total_factor": float(total),
                # the product Result 66 showed to be the binding quantity
                "ibm_bias_budget": float(hexagonal["median"] * IBM_BEST_TWO_QUBIT_ERROR),
                "ion_bias_budget": float(free["median"] * HELIOS_TWO_QUBIT_ERROR),
                "ion_detection_survival": survival(free["median"], HELIOS_TWO_QUBIT_ERROR),
                "seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(
                f"{name:>9}{spec:>8}{row['qubits']:>8}{hexagonal['median']:>11.0f}"
                f"{free['median']:>12.0f}{routing:>9.2f}{error_ratio:>9.2f}{total:>8.2f}",
                flush=True,
            )

    # ------------------------------------------------------------ the verdict
    print("\n--- does it change the verdict, or the constant? ---")
    if rows:
        routings = [r["routing_factor"] for r in rows]
        print(f"routing overhead removed: median {np.median(routings):.2f}x "
              f"(range {min(routings):.2f}-{max(routings):.2f})")
        print(f"per-gate error: {error_ratio:.2f}x better (published, not measured here)")
        print(f"combined: median {np.median([r['total_factor'] for r in rows]):.2f}x\n")

        print(f"{'molecule':>9}{'ansatz':>8}{'IBM bias':>11}{'ion bias':>11}"
              f"{'vs 1.6e-3':>11}{'shots kept':>12}")
        print("-" * 62)
        for row in rows:
            print(f"{row['molecule']:>9}{row['ansatz']:>8}"
                  f"{row['ibm_bias_budget']:>11.2e}{row['ion_bias_budget']:>11.2e}"
                  f"{row['ion_bias_budget'] / CHEMICAL_ACCURACY:>11.1f}"
                  f"{row['ion_detection_survival']:>12.2e}")

        inside = [r for r in rows if r["ion_bias_budget"] < CHEMICAL_ACCURACY]
        print(f"\ninside chemical accuracy on the ion machine: {len(inside)}/{len(rows)}")
        if inside:
            print("  " + ", ".join(f"{r['molecule']}/{r['ansatz']}" for r in inside))
        print("\n'shots kept' is the fraction a distance-2 DETECTING code accepts.")
        print("Where that column reads 1e-3, the machine is discarding 999 shots in")
        print("1000, and this project already measured shots to be VQE's binding cost.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp025_trapped_ion.json"
    with open(path, "w") as fh:
        json.dump({
            "backend": args.backend,
            "basis": list(BASIS),
            "seeds": list(SEEDS),
            "ibm_best_two_qubit_error": IBM_BEST_TWO_QUBIT_ERROR,
            "helios_two_qubit_error": HELIOS_TWO_QUBIT_ERROR,
            "helios_figures_are_published_not_measured": True,
            "chemical_accuracy": CHEMICAL_ACCURACY,
            "rows": rows,
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
