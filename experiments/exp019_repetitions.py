"""Experiment 019 -- the repetition count, which is where Result 71 left the
error-corrected estimate, and which turns out to be a VQE artefact.

Result 71 computed the wall-clock of *one* error-corrected circuit execution
(59 minutes at the parallel floor for a drug-sized molecule) and then closed on
the quantity it could not bound: how many executions an algorithm needs.  At
10^3 executions that is 41 days and at 10^6 it is 113 years, and this project had
measured H4 consuming ~10^8 shots without reaching chemical accuracy.  So the
open question was whether the repetition count destroys the qubit estimate.

**It does, for VQE, and VQE is the wrong algorithm to error-correct.**

The two families put the accuracy in different places:

* **VQE** is shot-noise-limited.  This repository *measured* the law
  ``shots ~ (Sigma|c|)^2 * n / eps^2`` (Result 42), so precision costs
  **repetitions**, quadratically.
* **Qubitization + phase estimation** is Heisenberg-limited.  Precision costs
  **coherent circuit depth**, linearly: ~``pi * lambda / (2 eps)`` applications of
  a walk operator, and O(1) repetitions.

Error correction is precisely the thing that makes depth affordable and does
nothing about repetitions.  So the comparison is not close, and the point of
this experiment is to put *this repository's own measured inputs* into it rather
than to quote the conclusion.

**Provenance of every number, because Result 72 is the standing lesson here:**

* ``lambda = Sigma|c|`` and the Pauli term count ``L`` -- **measured**, from the
  molecular Hamiltonians in this repo (Results 42, 53).
* the VQE shot law -- **measured** in this repo (Result 42).
* ansatz T counts -- **measured** by transpiling, via exp016.
* the qubitization query count and the LCU block-encoding cost -- **standard
  theory, not measured and not novel.**  Resource estimates of exactly this kind
  are a well-developed literature (Babbush, Berry, von Burg and others), and
  their constructions are far better than the naive LCU used here.  This is a
  reproduction with local inputs, and the naive encoding makes the qubitization
  side look *worse* than the state of the art, which is the safe direction.

Run:  python -m experiments.exp019_repetitions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qres.bench import RESULTS_DIR

CHEMICAL_ACCURACY = 1.6e-3

#: (molecule, spatial orbitals, qubits, Pauli terms L, Sigma|c|, ansatz parameters,
#: ansatz T gates).  L and Sigma|c| are measured from the Hamiltonians; the T
#: counts come from exp016's transpiled UCCSD; parameters from exp018.
MOLECULES = (
    ("H2",  2,  2,     5,  0.99,   3,     148),
    ("H4",  4,  6,   165,  8.47,  26,   7_997),
    ("LiH", 6, 10,   631, 12.34,  92,  37_651),
    ("H6",  6, 10,   919, 21.19, 117,  37_651),
    ("H2O", 7, 12, 1_086, 72.00, 100,  60_761),
    ("NH3", 8, 14, 2_949, 66.09, 200, 150_790),
    ("H8",  8, 14, 2_913, 40.01, 360, 150_790),
)

#: A Toffoli costs 4 T gates in the standard decomposition.
T_PER_TOFFOLI = 4

#: Naive LCU: SELECT over L terms costs ~L Toffolis, PREPARE a comparable amount.
#: Real constructions do far better -- this is deliberately the pessimistic side.
TOFFOLIS_PER_WALK_PER_TERM = 2.0

#: Phase estimation with a qubitized walk needs ~pi*lambda/(2*eps) applications.
QPE_CONSTANT = np.pi / 2

#: Repetitions to push the phase-estimation failure probability down. O(1) --
#: this is the whole point, against VQE's 1/eps^2.
QPE_REPETITIONS = 3

#: Result 71's floor: seconds of wall-clock per T gate on the data block at the
#: parallel floor, for a distance-23 code at a 1 us cycle.
SECONDS_PER_T_GATE = 23e-6


def qpe_resources(lam: float, terms: int, epsilon: float) -> dict:
    """Walk applications and T gates for qubitized phase estimation."""
    walks = QPE_CONSTANT * lam / epsilon
    toffolis_per_walk = TOFFOLIS_PER_WALK_PER_TERM * terms
    t_gates = walks * toffolis_per_walk * T_PER_TOFFOLI
    return {
        "walks": walks,
        "t_gates": t_gates * QPE_REPETITIONS,
        "repetitions": QPE_REPETITIONS,
    }


def vqe_resources(lam: float, parameters: int, t_per_circuit: int, epsilon: float) -> dict:
    """Shots and T gates for a variational loop, from the measured shot law."""
    shots = (lam**2) * parameters / (epsilon**2)
    return {
        "shots": shots,
        "t_gates": shots * t_per_circuit,
        "repetitions": shots,
    }


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 172_800:
        return f"{seconds / 3600:.1f} h"
    if seconds < 86400 * 400:
        return f"{seconds / 86400:.1f} days"
    return f"{seconds / (86400 * 365.25):.3g} years"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon", type=float, default=CHEMICAL_ACCURACY)
    args = ap.parse_args()
    epsilon = args.epsilon

    print("=== experiment 019 :: repetitions, and why VQE is the wrong thing to correct ===")
    print("ESTIMATE. lambda, L, ansatz T counts and the shot law are measured in")
    print("this repo; the qubitization cost model is standard theory, deliberately")
    print(f"taken at its naive (pessimistic) end.  target accuracy {epsilon:.1e} Ha\n")

    header = (
        f"{'molecule':<8}{'L':>7}{'lambda':>8}{'QPE reps':>10}{'QPE T':>12}"
        f"{'VQE reps':>12}{'VQE T':>12}{'T ratio':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name, orbitals, qubits, terms, lam, parameters, t_per_circuit in MOLECULES:
        qpe = qpe_resources(lam, terms, epsilon)
        vqe = vqe_resources(lam, parameters, t_per_circuit, epsilon)
        ratio = vqe["t_gates"] / qpe["t_gates"]

        print(
            f"{name:<8}{terms:>7,d}{lam:>8.2f}{qpe['repetitions']:>10,d}"
            f"{qpe['t_gates']:>12.2e}{vqe['repetitions']:>12.2e}"
            f"{vqe['t_gates']:>12.2e}{ratio:>11.1e}",
            flush=True,
        )
        rows.append({
            "molecule": name, "orbitals": orbitals, "qubits": qubits,
            "pauli_terms": terms, "lambda": lam, "ansatz_parameters": parameters,
            "ansatz_t_gates": t_per_circuit,
            "qpe_walks": qpe["walks"], "qpe_t_gates": qpe["t_gates"],
            "qpe_repetitions": qpe["repetitions"],
            "vqe_shots": vqe["shots"], "vqe_t_gates": vqe["t_gates"],
            "t_gate_ratio_vqe_over_qpe": ratio,
        })

    print("\n--- what that is in wall-clock, at Result 71's floor ---")
    print(f"({SECONDS_PER_T_GATE:.0e} s per T gate on the data block)\n")
    print(f"{'molecule':<8}{'QPE':>16}{'VQE':>18}")
    print("-" * 42)
    for row in rows:
        print(f"{row['molecule']:<8}"
              f"{human_time(row['qpe_t_gates'] * SECONDS_PER_T_GATE):>16}"
              f"{human_time(row['vqe_t_gates'] * SECONDS_PER_T_GATE):>18}")

    # ------------------------------------------------------------- the scaling
    print("\n--- the reason, which is a scaling difference and not a constant ---")
    print("VQE:  shots ~ lambda^2 / eps^2   -- precision costs REPETITIONS")
    print("QPE:  walks ~ lambda   / eps     -- precision costs DEPTH, reps are O(1)")
    print("\nSo the ratio grows as lambda/eps. Tightening the target by 10x")
    print("costs VQE 100x and QPE 10x, and error correction is exactly the")
    print("technology that makes depth affordable and does nothing for shots.\n")

    for factor in (1.0, 0.1, 0.01):
        target = CHEMICAL_ACCURACY * factor
        lam, terms, parameters, t_circ = 8.47, 165, 26, 7_997  # H4, measured
        q = qpe_resources(lam, terms, target)
        v = vqe_resources(lam, parameters, t_circ, target)
        print(f"  H4 at {target:.1e} Ha:  ratio {v['t_gates'] / q['t_gates']:.2e}")

    # the drug-sized extrapolation, using Result 53's measured basis-limit scaling
    print("\n--- and the drug-sized case ---")
    # lambda ~ N^2.86 in the basis-limit direction (Result 53), anchored on H4
    lam_drug = 8.47 * (50 / 4) ** 2.86
    terms_drug = 165 * (50 / 4) ** 4  # L ~ N^4 for a molecular Hamiltonian
    qpe_drug = qpe_resources(lam_drug, terms_drug, CHEMICAL_ACCURACY)
    print(f"50 orbitals: lambda ~ {lam_drug:,.0f} (N^2.86, Result 53), "
          f"L ~ {terms_drug:,.0f} (N^4)")
    print(f"  QPE: {qpe_drug['walks']:.2e} walks, {qpe_drug['t_gates']:.2e} T gates")
    print(f"  at the floor: {human_time(qpe_drug['t_gates'] * SECONDS_PER_T_GATE)}")
    print("\nResult 71 asked whether repetitions destroy the qubit estimate.")
    print("For VQE they do. For the algorithm one would actually run on a")
    print("fault-tolerant machine, the repetition count is 3 -- and what is left")
    print("is a depth problem, which is the thing the qubits were bought for.")

    # ------------------------------------------------------------------------
    # And the part that matters more than anything above: the block encoding.
    # ------------------------------------------------------------------------
    print("\n--- WHERE THIS ESTIMATE SITS IN THE LITERATURE ---")
    print("The naive LCU used here is the 2017-era construction, and the number")
    print("it produces says so.  Published estimates for FeMoco, a comparably")
    print("sized problem:\n")
    anchors = (
        ("Reiher et al. 2017, Trotter", 1e14, "years"),
        ("Berry/Gidney 2019, sparse qubitization", 1e10, "-"),
        ("Lee et al. 2021, tensor hypercontraction", 5.3e9 * T_PER_TOFFOLI, "~4 days"),
    )
    print(f"{'construction':<42}{'T gates':>12}{'reported':>12}")
    print("-" * 66)
    for label, t_gates, runtime in anchors:
        print(f"{label:<42}{t_gates:>12.1e}{runtime:>12}")
    print(f"{'THIS experiment, naive LCU':<42}{qpe_drug['t_gates']:>12.1e}"
          f"{human_time(qpe_drug['t_gates'] * SECONDS_PER_T_GATE):>12}")

    behind = qpe_drug["t_gates"] / (5.3e9 * T_PER_TOFFOLI)
    print(f"\nSo this model is ~{behind:.0e}x behind the current state of the art,")
    print("and it reproduces roughly where the field stood in 2017.  The gap is")
    print("not hardware and not measurement -- it is the Hamiltonian")
    print("representation inside the block encoding.")
    print("\n**That single algorithmic line is worth more than every constant")
    print("factor in this repository combined**, by about two orders of")
    print("magnitude, and none of it was invented here.")

    rows.append({
        "molecule": "extrapolated/50", "orbitals": 50, "lambda": lam_drug,
        "pauli_terms": terms_drug, "qpe_walks": qpe_drug["walks"],
        "qpe_t_gates": qpe_drug["t_gates"], "qpe_repetitions": QPE_REPETITIONS,
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp019_repetitions.json"
    with open(path, "w") as fh:
        json.dump({
            "epsilon": epsilon,
            "toffolis_per_walk_per_term": TOFFOLIS_PER_WALK_PER_TERM,
            "qpe_repetitions": QPE_REPETITIONS,
            "seconds_per_t_gate": SECONDS_PER_T_GATE,
            "rows": rows,
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
