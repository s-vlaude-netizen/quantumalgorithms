"""Experiment 011 -- does the measurement-scheme ranking survive past LiH?

RESEARCH_LOG Result 38 ranked five measurement schemes on one metric -- total
estimator variance under Neyman allocation, ``(sum_g sqrt(Var_g))^2``, at the
Hartree-Fock reference -- and the repository's default (general-commuting
grouping) won on all three molecules tested.  All three were tiny: H2, H4, LiH,
at most 631 Pauli terms.

NEXT_STEPS has carried "check whether the rankings are small-Hamiltonian
artefacts" as the top open item ever since, and it could not be run because
there was nothing bigger in the molecule set.  Result 53 added nine molecules,
so it can be run now.  The systems here reach 2 949 terms -- 4.7x past anything
the ranking was established on, and as far as the metric can be evaluated at
all.

Two things are measured, because a structural ranking is not a runtime claim:

*The Result 38 metric* on every molecule that fits, and *the wall-clock each
scheme costs to plan*.  Planning is not free: general-commuting grouping is a
graph colouring over every pair of terms, and the covariance matrix the metric
needs is worse still.  Both are reported, and the second turns out to set the
ceiling on the whole approach.

Run:  python -m experiments.exp011_scheme_ranking_at_scale
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
from qres.covariance import covariance_matrix, predicted_total_variance
from qres.measurement import group_paulis
from qres.problems.chemistry import build_molecule
from qres.shadows import random_pauli_shadow_variance

#: The covariance matrix is the binding cost, and it is worse than its memory
#: footprint suggests: measured, it takes ``seconds ~ terms^2.9``.  H2O's 1 086
#: terms need 60 s, NH3's 2 949 need 1 175 s, and CH4's 6 892 would need close to
#: four hours.  So the default stops just past NH3 -- and that ceiling is itself
#: a finding: the metric Result 38 ranks schemes on cannot be evaluated at all
#: for molecules of practical size.
MAX_TERMS = 3500

ORIGINAL = ("H2", "H4", "LiH")
LARGER = ("BeH2", "HF", "H2O", "NH3", "CH4")


def score_molecule(name: str, max_terms: int = MAX_TERMS) -> dict | None:
    problem = build_molecule(name)
    hamiltonian = problem.hamiltonian
    terms = len(hamiltonian)
    if terms > max_terms:
        print(f"{name:<8}{terms:>7}  skipped: over the {max_terms}-term memory limit", flush=True)
        return None

    reference = Statevector(hartree_fock_state(problem.hf_bitstring))
    coefficients = np.asarray(hamiltonian.coeffs).real

    started = time.perf_counter()
    covariance = covariance_matrix(hamiltonian, reference)
    covariance_seconds = time.perf_counter() - started

    row: dict = {
        "molecule": name,
        "orbitals": problem.num_spatial_orbitals,
        "qubits": hamiltonian.num_qubits,
        "terms": terms,
        "covariance_seconds": covariance_seconds,
    }

    for label, method in (("commuting", "commuting"), ("qwc", "qwc")):
        started = time.perf_counter()
        groups, _ = group_paulis(hamiltonian, method)
        plan_seconds = time.perf_counter() - started
        row[f"{label}_groups"] = len(groups)
        row[f"{label}_variance"] = predicted_total_variance(groups, covariance, coefficients)
        row[f"{label}_plan_seconds"] = plan_seconds

    row["random_shadow_variance"] = random_pauli_shadow_variance(hamiltonian)

    base = row["commuting_variance"]
    for key in ("qwc", "random_shadow"):
        source = f"{key}_variance"
        row[f"{key}_ratio"] = row[source] / base if base else float("nan")
    return row


def _save(rows, path) -> None:
    """Write after every molecule.

    A single write at the end loses everything to a timeout, and the runs this
    experiment makes are long enough that timeouts are the normal ending -- the
    first attempt was killed four hours into CH4 with nothing on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"rows": rows}, fh, indent=2)


def ranking(names, max_terms: int = MAX_TERMS, path=None) -> list[dict]:
    print("=== part 1 :: the Result 38 metric, past the sizes it was set on ===")
    print("ratios are against general-commuting grouping (1.00 = the default wins)\n")
    header = (
        f"{'molecule':<8}{'terms':>7}{'qubits':>7}{'groups':>8}{'QWC':>7}"
        f"{'QWC x':>8}{'shadow x':>10}{'plan s':>8}{'cov s':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in names:
        try:
            row = score_molecule(name, max_terms)
        except MemoryError:
            print(f"{name:<8}  out of memory building the covariance matrix", flush=True)
            continue
        if row is None:
            continue
        print(
            f"{row['molecule']:<8}{row['terms']:>7}{row['qubits']:>7}"
            f"{row['commuting_groups']:>8}{row['qwc_groups']:>7}"
            f"{row['qwc_ratio']:>8.2f}{row['random_shadow_ratio']:>10.1f}"
            f"{row['commuting_plan_seconds']:>8.1f}{row['covariance_seconds']:>8.1f}",
            flush=True,
        )
        rows.append(row)
        if path is not None:
            _save(rows, path)
    return rows


def planning_cost(rows) -> None:
    """Does the winner's planning cost grow faster than the problem?"""
    if len(rows) < 4:
        return
    print("\n=== planning cost: is the winner still affordable at scale? ===\n")
    terms = np.array([r["terms"] for r in rows], dtype=float)
    plan = np.array([r["commuting_plan_seconds"] for r in rows], dtype=float)
    covariance = np.array([r["covariance_seconds"] for r in rows], dtype=float)

    usable = plan > 1e-3
    if usable.sum() >= 3:
        exponent = np.polyfit(np.log(terms[usable]), np.log(plan[usable]), 1)[0]
        print(f"  general-commuting grouping: seconds ~ terms^{exponent:.2f}")
    if (covariance > 1e-3).sum() >= 3:
        mask = covariance > 1e-3
        exponent = np.polyfit(np.log(terms[mask]), np.log(covariance[mask]), 1)[0]
        print(f"  covariance matrix:          seconds ~ terms^{exponent:.2f}")
    print(f"  largest measured: {int(terms.max()):,} terms, "
          f"{plan.max():.1f}s to group, {covariance.max():.1f}s for covariance")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default=",".join(ORIGINAL + LARGER))
    ap.add_argument("--max-terms", type=int, default=MAX_TERMS)
    args = ap.parse_args()

    print("=== experiment 011 :: measurement schemes past LiH ===")
    print("Result 38 ranked five schemes on H2/H4/LiH -- at most 631 terms\n")

    path = RESULTS_DIR / "exp011_scheme_ranking_at_scale.json"
    rows = ranking(args.molecules.split(","), args.max_terms, path)
    planning_cost(rows)

    if rows:
        losses = [r for r in rows if r["qwc_ratio"] < 1.0]
        print()
        if losses:
            print("QWC beat general commuting on: " + ", ".join(r["molecule"] for r in losses))
            print("Result 38's ranking does NOT hold at these sizes.")
        else:
            print("General-commuting grouping still wins on every molecule tested,")
            print(f"now up to {max(r['terms'] for r in rows):,} terms "
                  f"({max(r['qubits'] for r in rows)} qubits).")

    _save(rows, path)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
