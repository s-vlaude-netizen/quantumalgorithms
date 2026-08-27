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

#: measured by transpiling UCCSD: (orbitals, qubits, two-qubit gates, non-Clifford rotations)
#: The rotation count is what drives magic-state cost, and it is far smaller than
#: the gate count -- UCCSD carries one rotation per excitation wrapped in Clifford
#: basis changes, so NH3's 32 318 two-qubit gates need only 2 340 rotations.
MOLECULES = (
    ("H2", 2, 2, 4, 4),
    ("H4", 4, 6, 1_471, 152),
    ("LiH", 6, 10, 9_103, 640),
    ("H2O", 7, 12, 24_156, 1_000),
    ("BeH2", 7, 12, 26_111, 1_000),
    ("NH3", 8, 14, 50_570, 2_340),
)

#: Ross-Selinger synthesis: a rotation to precision eps costs ~3 log2(1/eps) T gates
SYNTHESIS_CONSTANT = 3.0

#: 15-to-1 distillation maps input error p to ~35 p^3, consuming 15 states
DISTILLATION_INPUTS = 15
DISTILLATION_CONSTANT = 35.0

#: a 15-to-1 factory occupies roughly this many logical qubits' worth of area
FACTORY_LOGICAL_QUBITS = 12


def t_count(rotations: int, total_budget: float) -> tuple[int, float]:
    """T gates needed, and the per-rotation synthesis precision they buy."""
    if rotations == 0:
        return 0, 0.0
    # synthesis error is shared across rotations, so each gets budget/rotations
    epsilon = total_budget / (2 * rotations)
    per_rotation = SYNTHESIS_CONSTANT * np.log2(1 / epsilon)
    return int(np.ceil(rotations * per_rotation)), epsilon


def distillation_rounds(physical: float, target: float, limit: int = 6) -> int | None:
    """Rounds of 15-to-1 needed to bring magic-state error below ``target``."""
    error = physical
    for rounds in range(1, limit + 1):
        error = DISTILLATION_CONSTANT * error**3
        if error < target:
            return rounds
    return None

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
        f"{'distance':>10}{'data qubits':>17}{'T count':>11}{'rounds':>8}"
        f"{'total':>13}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name, orbitals, qubits, gates, rotations in MOLECULES:
        # the whole calculation must land inside chemical accuracy, so each
        # logical gate may contribute at most accuracy/gates of error
        target = CHEMICAL_ACCURACY / gates
        distance = required_distance(target, physical, threshold)
        if distance is None:
            print(f"{name:<8}{qubits:>15}{gates:>10,d}{target:>13.1e}"
                  f"{'--':>10}{'beyond the model':>17}", flush=True)
            continue

        data_qubits = qubits * 2 * distance**2

        # magic states: every non-Clifford rotation synthesises into T gates, and
        # every T gate consumes a distilled state good enough not to spend the
        # whole error budget on its own
        gates_needed, _ = t_count(rotations, CHEMICAL_ACCURACY)
        magic_target = CHEMICAL_ACCURACY / max(gates_needed, 1)
        rounds = distillation_rounds(physical, magic_target)
        factory_qubits = (
            FACTORY_LOGICAL_QUBITS * 2 * distance**2 * (rounds or 0)
        )

        print(
            f"{name:<8}{qubits:>15}{gates:>10,d}{target:>13.1e}"
            f"{distance:>10}{data_qubits:>17,d}{gates_needed:>11,d}"
            f"{rounds if rounds else '--':>8}{data_qubits + factory_qubits:>13,d}",
            flush=True,
        )
        rows.append({
            "molecule": name, "orbitals": orbitals, "logical_qubits": qubits,
            "gates": gates, "rotations": rotations, "target_logical_error": target,
            "code_distance": distance, "data_qubits": data_qubits,
            "t_count": gates_needed, "distillation_rounds": rounds,
            "factory_qubits": factory_qubits,
            "physical_qubits": data_qubits + factory_qubits,
        })

    # and the extrapolated drug-sized case, using Result 66's N^5.12 gate scaling
    # plus the measured rotation scaling from the table above
    print()
    measured_orbitals = np.array([m[1] for m in MOLECULES[1:]], dtype=float)
    measured_rotations = np.array([m[4] for m in MOLECULES[1:]], dtype=float)
    rotation_fit = np.polyfit(np.log(measured_orbitals), np.log(measured_rotations), 1)
    print(f"non-Clifford rotations scale as N^{rotation_fit[0]:.2f} (measured, 5 molecules)")

    reference_orbitals, reference_gates = 4, 1_471
    for orbitals in (20, 50):
        gates = reference_gates * (orbitals / reference_orbitals) ** 5.12
        rotations = int(np.exp(np.polyval(rotation_fit, np.log(orbitals))))
        logical_qubits = 2 * orbitals - 2  # parity mapping with two-qubit reduction
        target = CHEMICAL_ACCURACY / gates
        distance = required_distance(target, physical, threshold)
        if distance is None:
            print(f"{orbitals} orbitals: beyond the model at this physical error")
            continue

        data_qubits = logical_qubits * 2 * distance**2
        gates_needed, _ = t_count(rotations, CHEMICAL_ACCURACY)
        magic_target = CHEMICAL_ACCURACY / max(gates_needed, 1)
        rounds = distillation_rounds(physical, magic_target)
        factory_qubits = FACTORY_LOGICAL_QUBITS * 2 * distance**2 * (rounds or 0)
        total = data_qubits + factory_qubits

        label = "a small fragment" if orbitals == 20 else "a modest drug molecule"
        print(f"{orbitals} orbitals ({label}): {gates:,.0f} gates, "
              f"{rotations:,} rotations, {gates_needed:,} T gates, distance "
              f"{distance}, {rounds} distillation rounds")
        print(f"     data {data_qubits:,} + factory {factory_qubits:,} = "
              f"**{total:,} physical qubits**")
        rows.append({
            "molecule": f"extrapolated/{orbitals}", "orbitals": orbitals,
            "logical_qubits": logical_qubits, "gates": float(gates),
            "rotations": rotations, "target_logical_error": target,
            "code_distance": distance, "data_qubits": data_qubits,
            "t_count": gates_needed, "distillation_rounds": rounds,
            "factory_qubits": factory_qubits, "physical_qubits": int(total),
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
