"""Experiment 003 -- does covariance-aware Pauli grouping beat count-minimising?

Standard grouping minimises the number of measurement groups.  Under Neyman
shot allocation the quantity that actually sets the shot cost is

    total variance = (sum_g sqrt(Var_g))^2,    Var_g = sum_{i,j in g} c_i c_j Cov_ij

and the cross terms mean group *count* and total variance can disagree: putting
two anticorrelated observables together makes a group cheaper than either term
alone suggests.

Three things are measured, and the third is the one that decides it:

1. **Predicted variance** of each grouping, scored against the covariance of the
   *exact ground state* -- the state a converged VQE would sit at.
2. **Reference sensitivity.** Covariances have to come from somewhere cheap.
   Hartree-Fock is free; the exact ground state is an oracle nobody has. If the
   oracle is far better, the idea needs a better cheap reference to be useful.
3. **The group-count control.** Covariance-aware grouping produces *more* groups
   than count-minimising, and more groups also means finer-grained Neyman
   allocation.  So a grouping that is merely split into the same number of
   groups at random is included: if that captures the benefit, the covariance
   machinery is doing nothing and the result is an artefact.

Run:  python -m experiments.exp003_covariance_grouping
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
from qres.covariance import (
    covariance_grouping,
    covariance_matrix,
    group_variance,
    predicted_total_variance,
    refined_covariance_grouping,
)
from qres.measurement import group_paulis
from qres.problems.chemistry import build_molecule

MOLECULE_KWARGS = {
    "H2": {},
    "H4": {},
    "LiH": {},
    "BeH2": dict(active_electrons=4, active_orbitals=6),
}


def exact_ground_state(hamiltonian) -> Statevector:
    matrix = hamiltonian.to_matrix()
    _, vectors = np.linalg.eigh(matrix)
    # eigh returns a column view; Statevector needs a contiguous array
    return Statevector(np.ascontiguousarray(vectors[:, 0]))


def random_split_control(groups, target, coeffs, cov, seed) -> float:
    """Split the count-minimising grouping at random until it has ``target`` groups.

    Isolates "does having more groups help" from "does knowing the covariance
    help".  Splitting destroys whatever beneficial anticorrelation the greedy
    cover happened to capture, so if the real method only matched this, it would
    be measuring nothing.
    """
    rng = np.random.default_rng(seed)
    parts = [list(g.indices) for g in groups]
    while len(parts) < target:
        biggest = max(range(len(parts)), key=lambda i: len(parts[i]))
        if len(parts[biggest]) < 2:
            break
        members = parts[biggest][:]
        rng.shuffle(members)
        cut = len(members) // 2
        parts[biggest], extra = members[:cut], members[cut:]
        parts.append(extra)
    return float(sum(np.sqrt(max(group_variance(p, coeffs, cov), 0.0)) for p in parts) ** 2)


def study_molecule(name: str, method: str = "commuting") -> dict:
    problem = build_molecule(name, **MOLECULE_KWARGS[name])
    coeffs = np.asarray(problem.hamiltonian.coeffs).real

    ground = exact_ground_state(problem.hamiltonian)
    reference_hf = Statevector(hartree_fock_state(problem.hf_bitstring))
    # scoring always uses the TRUE covariance, whatever the grouping was built from
    cov_true = covariance_matrix(problem.hamiltonian, ground)

    baseline, _ = group_paulis(problem.hamiltonian, method)
    var_base = predicted_total_variance(baseline, cov_true, coeffs)

    row = {
        "molecule": name,
        "num_qubits": problem.num_qubits,
        "num_terms": len(problem.hamiltonian),
        "baseline_groups": len(baseline),
        "baseline_variance": var_base,
        "variants": {},
    }

    variants = [
        ("hartree_fock", reference_hf, False),
        ("oracle_ground_state", ground, False),
        ("hartree_fock+refined", reference_hf, True),
    ]
    for label, reference, refine in variants:
        t0 = time.perf_counter()
        if refine:
            groups, _, moves = refined_covariance_grouping(
                problem.hamiltonian, reference, method, max_sweeps=400
            )
        else:
            groups, _ = covariance_grouping(problem.hamiltonian, reference, method)
            moves = 0
        variance = predicted_total_variance(groups, cov_true, coeffs)
        controls = [
            random_split_control(baseline, len(groups), coeffs, cov_true, s) for s in range(20)
        ]
        row["variants"][label] = {
            "groups": len(groups),
            "variance": variance,
            "ratio": variance / var_base,
            "refinement_moves": moves,
            "control_ratio_median": float(np.median(controls)) / var_base,
            "control_ratio_min": float(np.min(controls)) / var_base,
            "seconds": time.perf_counter() - t0,
        }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default="H2,H4,LiH")
    ap.add_argument("--method", default="commuting", choices=["commuting", "qwc"])
    args = ap.parse_args()

    print(f"=== experiment 003 :: covariance-aware grouping ({args.method}) ===")
    print("ratio < 1 means lower variance at equal shots, i.e. fewer shots for equal error\n")
    header = (
        f"{'molecule':<10}{'terms':>7}{'groups':>8}{'variance':>11}"
        f"{'ratio':>8}{'control':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in args.molecules.split(","):
        row = study_molecule(name, args.method)
        rows.append(row)
        print(
            f"{row['molecule']:<10}{row['num_terms']:>7}{row['baseline_groups']:>8}"
            f"{row['baseline_variance']:>11.4f}{1.0:>8.3f}{'--':>10}   count-minimising",
            flush=True,
        )
        for label, v in row["variants"].items():
            print(
                f"{'':<10}{'':>7}{v['groups']:>8}{v['variance']:>11.4f}"
                f"{v['ratio']:>8.3f}{v['control_ratio_median']:>10.3f}   cov-aware / {label}",
                flush=True,
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"exp003_covariance_{args.method}.json"
    with open(path, "w") as fh:
        json.dump({"method": args.method, "rows": rows}, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
