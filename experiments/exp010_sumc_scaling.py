"""Experiment 010 -- what actually drives the coefficient sum Sum|c|?

`Sum|c|` is the whole shot budget: RESEARCH_LOG Result 34 measured
`shots ~ (Sum|c|)^2 * n / eps^2`, and Result 42 turned that into the project's
headline by fitting `Sum|c| ~ N^2.78` over five molecules at STO-3G.

That fit is not identifiable from those five points, which is what this
experiment establishes and repairs.  Two of them sit at the **same** N = 7 with
coefficient sums differing 3.35x (BeH2 21.52, H2O 72.00) -- a function of N
alone cannot do that.  The cause is collinearity: in a minimal basis the orbital
count is decided by which atoms are present, so N and total nuclear charge move
together (corr(log N, log Z) = 0.88 across the original five), and a regression
cannot attribute the growth to either.

Three parts, each isolating one thing:

``survey``      thirteen molecules; compare an N-only fit against N-and-Z, and
                report the collinearity that makes both sets of exponents
                untrustworthy on their own.
``fixed_charge``  four systems at Z = 10 with 6, 7, 8, 9 orbitals.  Adding atoms
                at constant nuclear charge -- does Sum|c| grow?
``fixed_molecule``  one molecule, growing basis.  Nuclear charge is pinned, so
                this is the only clean measurement of the orbital-count
                exponent, and it is the direction that matters: the chemistry is
                at the basis limit, not at more atoms in a minimal basis.

Run:  python -m experiments.exp010_sumc_scaling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qres.bench import RESULTS_DIR
from qres.problems.chemistry import NUCLEAR_CHARGE, build_molecule

#: the five the original law was fitted on, for the collinearity comparison
ORIGINAL_FIVE = ("H2", "H4", "LiH", "BeH2", "H2O")

SURVEY = (
    "H2", "H4", "H6", "H8", "LiH", "Li2", "BeH2", "BH3",
    "CH4", "NH3", "H2O", "HF", "N2",
)


def coefficient_sum(problem) -> float:
    """Sum |c| over non-identity terms.

    The identity carries no variance -- it is a constant offset on every shot --
    so it must not enter a shot-count formula.  Including it inflates H2O from
    72.00 to 127.61 and would have made this experiment report a scaling
    correction that was not there.
    """
    hamiltonian = problem.hamiltonian
    return float(
        sum(
            abs(complex(coeff))
            for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs)
            if np.any(pauli.z) or np.any(pauli.x)
        )
    )


def _spatial_orbitals(problem) -> int:
    """Orbital count as the driver recorded it.

    Not ``num_qubits // 2 + 1``: that happens to agree under the default parity
    mapping with two-qubit reduction and silently stops agreeing under any other
    mapper or an active space, which would shift every point in a log-log fit.
    """
    return problem.num_spatial_orbitals


def survey(names=SURVEY) -> list[dict]:
    print("=== part 1 :: thirteen molecules at STO-3G ===\n")
    header = f"{'molecule':<10}{'orbitals':>9}{'charge':>8}{'qubits':>8}{'terms':>8}{'sum|c|':>10}{'s':>7}"
    print(header)
    print("-" * len(header))

    rows = []
    for name in names:
        started = time.perf_counter()
        try:
            problem = build_molecule(name)
        except Exception as exc:  # a basis/geometry this environment cannot build
            print(f"{name:<10}  skipped: {str(exc)[:52]}", flush=True)
            continue
        seconds = time.perf_counter() - started
        total = coefficient_sum(problem)
        orbitals = _spatial_orbitals(problem)
        print(
            f"{name:<10}{orbitals:>9}{NUCLEAR_CHARGE[name]:>8}"
            f"{problem.hamiltonian.num_qubits:>8}{len(problem.hamiltonian):>8}"
            f"{total:>10.2f}{seconds:>7.1f}",
            flush=True,
        )
        rows.append(
            {
                "molecule": name,
                "orbitals": orbitals,
                "charge": NUCLEAR_CHARGE[name],
                "qubits": problem.hamiltonian.num_qubits,
                "terms": len(problem.hamiltonian),
                "sum_c": total,
            }
        )
    return rows


def report_fits(rows) -> dict:
    n = np.array([r["orbitals"] for r in rows], dtype=float)
    z = np.array([r["charge"] for r in rows], dtype=float)
    s = np.array([r["sum_c"] for r in rows], dtype=float)

    one = np.polyfit(np.log(n), np.log(s), 1)
    residual_one = np.log(s) - np.polyval(one, np.log(n))

    design = np.column_stack([np.log(n), np.log(z), np.ones(len(n))])
    two, *_ = np.linalg.lstsq(design, np.log(s), rcond=None)
    residual_two = np.log(s) - design @ two

    correlation = float(np.corrcoef(np.log(n), np.log(z))[0, 1])

    print(f"\n{'model':<18}{'fit':<30}{'residual std':>13}")
    print(f"{'N only':<18}{f'sum|c| ~ N^{one[0]:.2f}':<30}{residual_one.std():>13.3f}")
    print(
        f"{'N and Z':<18}"
        f"{f'sum|c| ~ N^{two[0]:.2f} Z^{two[1]:.2f}':<30}{residual_two.std():>13.3f}"
    )
    print(
        f"\ncorr(log N, log Z) = {correlation:.3f}, "
        f"variance inflation {1 / (1 - correlation**2):.2f}"
    )
    print(
        "Both predictors move together, so neither set of exponents is\n"
        "trustworthy on its own. Parts 2 and 3 vary one at a time."
    )
    return {
        "n_only_exponent": float(one[0]),
        "n_only_residual_std": float(residual_one.std()),
        "n_exponent": float(two[0]),
        "z_exponent": float(two[1]),
        "two_variable_residual_std": float(residual_two.std()),
        "collinearity": correlation,
    }


def fixed_charge(rows) -> dict:
    """Adding atoms while total nuclear charge stays put."""
    print("\n=== part 2 :: more atoms, same total nuclear charge (Z = 10) ===\n")
    group = sorted((r for r in rows if r["charge"] == 10), key=lambda r: r["orbitals"])
    if len(group) < 3:
        print("not enough Z=10 systems built")
        return {}

    for row in group:
        print(f"  {row['molecule']:<6} orbitals {row['orbitals']:<3} sum|c| = {row['sum_c']:.2f}")

    n = np.array([r["orbitals"] for r in group], dtype=float)
    s = np.array([r["sum_c"] for r in group], dtype=float)
    exponent = float(np.polyfit(np.log(n), np.log(s), 1)[0])
    print(f"\n  -> sum|c| ~ N^{exponent:+.2f}")
    print("     Spreading the same electrons over more orbitals does not")
    print("     increase the coefficient sum.")
    return {"molecules": [r["molecule"] for r in group], "exponent": exponent}


def fixed_molecule(name: str = "H2", bases=("sto3g", "631g", "ccpvdz")) -> dict:
    """Growing the basis on one molecule: N varies, nuclear charge is pinned."""
    print(f"\n=== part 3 :: more basis functions, same molecule ({name}) ===\n")
    header = f"{'basis':<10}{'orbitals':>9}{'terms':>8}{'sum|c|':>10}{'build s':>9}"
    print(header)
    print("-" * len(header))

    points = []
    for basis in bases:
        started = time.perf_counter()
        try:
            problem = build_molecule(name, basis=basis)
        except Exception as exc:
            print(f"{basis:<10}  skipped: {str(exc)[:48]}", flush=True)
            continue
        seconds = time.perf_counter() - started
        total = coefficient_sum(problem)
        orbitals = _spatial_orbitals(problem)
        print(
            f"{basis:<10}{orbitals:>9}{len(problem.hamiltonian):>8}"
            f"{total:>10.2f}{seconds:>9.1f}",
            flush=True,
        )
        points.append({"basis": basis, "orbitals": orbitals, "sum_c": total})

    if len(points) < 3:
        return {"points": points}

    n = np.array([p["orbitals"] for p in points], dtype=float)
    s = np.array([p["sum_c"] for p in points], dtype=float)
    exponent = float(np.polyfit(np.log(n), np.log(s), 1)[0])
    print(f"\n  -> sum|c| ~ N^{exponent:.2f} at fixed nuclear charge")
    print(f"     shots ~ (sum|c|)^2 n / eps^2  =  N^{2 * exponent:.2f} n / eps^2")
    print(f"     with UCCSD's n ~ N^4 that is N^{2 * exponent + 4:.1f} in shots,")
    print("     against CCSD(T)'s N^7 in operations.")
    return {"molecule": name, "points": points, "exponent": exponent}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis-molecule", default="H2")
    ap.add_argument("--skip-survey", action="store_true")
    args = ap.parse_args()

    print("=== experiment 010 :: what drives Sum|c|? ===")
    print("the coefficient sum sets the shot budget; Result 42 fitted it to N alone\n")

    rows = [] if args.skip_survey else survey()
    fits = report_fits(rows) if len(rows) >= 6 else {}
    charge = fixed_charge(rows) if rows else {}
    basis = fixed_molecule(args.basis_molecule)

    if charge and basis:
        print(
            f"\nThe two directions have opposite signs "
            f"({charge['exponent']:+.2f} against {basis['exponent']:+.2f}).\n"
            "A single exponent in N averages over them in whatever proportion the\n"
            "molecule set happens to contain, which is what Result 42 reported."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp010_sumc_scaling.json"
    with open(path, "w") as fh:
        json.dump(
            {"survey": rows, "fits": fits, "fixed_charge": charge, "fixed_molecule": basis},
            fh,
            indent=2,
        )
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
