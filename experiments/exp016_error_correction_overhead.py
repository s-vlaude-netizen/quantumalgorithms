"""Experiment 016 -- how many physical qubits buy the fidelity the hardware lacks?

Result 65 measured that the cost of a two-qubit gate *is* the device's error
rate, and Result 66 turned that into a threshold per molecule: H4 needs ~1.6e-6
against the best available 0.00127, a factor of ~800.  NEXT_STEPS called error
correction "the unmeasured question" and "a different project".

It is a different project to *build*, but not to size, and the sizing is the
question a reader actually has: error correction trades **qubits** for
**fidelity**, so the 800x has a price in qubits and that price is what decides
whether any of this is reachable.

**This is an estimate, not a measurement, and the distinction matters here.**
Everything else in RESEARCH_LOG is measured; this combines two measured inputs --
the physical error rate from calibration data, and the gate counts from
transpiling real ansaetze -- with a standard surface-code model:

    logical error per round   p_L ~ A (p / p_th)^((d+1)/2)
    physical qubits per logical qubit   ~ 2 d^2

with ``p_th = 1e-2`` and ``A = 0.1``, the conventional values.  The exponent is
the part that carries the conclusion and it is not sensitive to A; the threshold
is, and a factor of two in ``p_th`` moves the answer by roughly one code
distance.  Both are reported so a reader can re-run with their own numbers.

Run:  python -m experiments.exp016_error_correction_overhead
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qres.bench import RESULTS_DIR

#: surface-code model constants (conventional values, not measured here)
THRESHOLD = 1e-2
PREFACTOR = 0.1

#: measured in Result 65: the best median two-qubit error in the fake provider
BEST_PHYSICAL_ERROR = 0.00127

#: measured in Result 66 by transpiling UCCSD: (orbitals, qubits, two-qubit gates)
MOLECULES = (
    ("H2", 2, 2, 4),
    ("H4", 4, 6, 1_471),
    ("LiH", 6, 10, 9_103),
    ("H2O", 7, 12, 24_156),
    ("BeH2", 7, 12, 26_111),
    ("NH3", 8, 14, 50_570),
)

CHEMICAL_ACCURACY = 1.6e-3


def logical_error(distance: int, physical: float, threshold: float = THRESHOLD) -> float:
    """Surface-code logical error per round."""
    return PREFACTOR * (physical / threshold) ** ((distance + 1) / 2)


def required_distance(
    target: float, physical: float, threshold: float = THRESHOLD, limit: int = 199
) -> int | None:
    """Smallest odd distance whose logical error is below ``target``."""
    for distance in range(3, limit + 1, 2):
        if logical_error(distance, physical, threshold) < target:
            return distance
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physical-error", type=float, default=BEST_PHYSICAL_ERROR)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    threshold = args.threshold
    physical = args.physical_error

    print("=== experiment 016 :: the qubit price of the missing fidelity ===")
    print("ESTIMATE, not a measurement: measured error rate and measured gate")
    print("counts, combined with a standard surface-code model")
    print(f"physical two-qubit error {physical:.5f} (measured, Result 65), "
          f"threshold {threshold:.0e}, prefactor {PREFACTOR}\n")

    header = (
        f"{'molecule':<8}{'logical qubits':>15}{'gates':>10}{'needed p_L':>13}"
        f"{'distance':>10}{'physical qubits':>17}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name, orbitals, qubits, gates in MOLECULES:
        # the whole calculation must land inside chemical accuracy, so each
        # logical gate may contribute at most accuracy/gates of error
        target = CHEMICAL_ACCURACY / gates
        distance = required_distance(target, physical, threshold)
        if distance is None:
            print(f"{name:<8}{qubits:>15}{gates:>10,d}{target:>13.1e}"
                  f"{'--':>10}{'beyond the model':>17}", flush=True)
            continue

        physical_qubits = qubits * 2 * distance**2
        print(
            f"{name:<8}{qubits:>15}{gates:>10,d}{target:>13.1e}"
            f"{distance:>10}{physical_qubits:>17,d}",
            flush=True,
        )
        rows.append({
            "molecule": name, "orbitals": orbitals, "logical_qubits": qubits,
            "gates": gates, "target_logical_error": target,
            "code_distance": distance, "physical_qubits": physical_qubits,
        })

    # and the extrapolated drug-sized case, using Result 66's N^5.12 gate scaling
    print()
    reference_orbitals, reference_gates = 4, 1_471
    for orbitals in (20, 50):
        gates = reference_gates * (orbitals / reference_orbitals) ** 5.12
        logical_qubits = 2 * orbitals - 2  # parity mapping with two-qubit reduction
        target = CHEMICAL_ACCURACY / gates
        distance = required_distance(target, physical, threshold)
        if distance is None:
            print(f"{orbitals} orbitals: beyond the model at this physical error")
            continue
        physical_qubits = logical_qubits * 2 * distance**2
        label = "a small fragment" if orbitals == 20 else "a modest drug molecule"
        print(f"{orbitals} orbitals ({label}): {gates:,.0f} gates, "
              f"distance {distance}, **{physical_qubits:,.0f} physical qubits**")
        rows.append({
            "molecule": f"extrapolated/{orbitals}", "orbitals": orbitals,
            "logical_qubits": logical_qubits, "gates": float(gates),
            "target_logical_error": target, "code_distance": distance,
            "physical_qubits": int(physical_qubits),
        })

    print(f"\nFor scale: the largest device in the fake provider is 156 qubits.")
    print("The physical error rate enters only through log(p/p_th), so the")
    print("distance -- and the qubit count -- is remarkably insensitive to it:")
    for candidate in (0.00127, 0.0005, 0.0001):
        distance = required_distance(CHEMICAL_ACCURACY / 1_471, candidate, threshold)
        qubits = 6 * 2 * distance**2 if distance else None
        print(f"  H4 at physical error {candidate:.5f}: distance {distance}, "
              f"{qubits:,} physical qubits")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp016_error_correction_overhead.json"
    with open(path, "w") as fh:
        json.dump({"physical_error": physical, "threshold": threshold,
                   "prefactor": PREFACTOR, "rows": rows}, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
