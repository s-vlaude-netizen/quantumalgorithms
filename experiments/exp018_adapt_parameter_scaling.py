"""Experiment 018 -- does the adaptive ansatz change an exponent, or a constant?

This is the one open question in this repository that could move a *complexity
exponent* rather than a constant factor, and it has been stated in the README for
several sessions without ever being measured:

    shots ~ (Sigma|c|)^2 * n / eps^2

Everything this project has produced buys a constant factor against that.  The
single exception is ``n``, the parameter count.  UCCSD has ``n ~ N^4``, which
with the measured ``Sigma|c|`` scaling gives ``N^9.7`` in shots against CCSD(T)'s
``N^7`` in operations.  An adaptive ansatz reaching the same accuracy with
``n ~ N^2`` would give ``N^7.6`` -- the same order as the classical competitor,
which is where a comparison would start being interesting.

**What existed before this experiment was three anecdotes**: 1 parameter against
3 on H2, 9 against 26 on H4, 5 against 92 on LiH.  That looks like a growing
advantage, and Result 53 is the standing reminder of what happens when a scaling
law is fitted across molecules that differ in more than one way -- there, N and
nuclear charge were 88% collinear and the exponent was assigned by default rather
than measured.

So this uses a **homologous series** -- H2, H4, H6, H8 at fixed bond length --
where the only thing that varies is N.  Same chemistry, same basis, same
geometry, one variable.

**The convergence criterion has to be identical across molecules or the
comparison is meaningless.**  Both ansaetze are asked for the same thing: energy
within chemical accuracy (1.6 mHa) of FCI.  ADAPT stops when it gets there;
UCCSD's parameter count is what its fixed construction supplies.  A run that does
*not* reach the target is reported as not reaching it rather than quietly
contributing its parameter count to the fit.

Run:  python -m experiments.exp018_adapt_parameter_scaling
      python -m experiments.exp018_adapt_parameter_scaling --molecules H2 H4 H6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qres.adapt import adapt_vqe
from qres.ansatz import uccsd
from qres.bench import RESULTS_DIR
from qres.problems.chemistry import build_molecule

CHEMICAL_ACCURACY = 1.6e-3

#: the homologous series: only N varies
DEFAULT_MOLECULES = ("H2", "H4", "H6", "H8")


def fit_exponent(sizes, counts) -> tuple[float, float]:
    """Log-log slope and its standard error.

    The standard error matters more than the slope here: four points can produce
    a confident-looking exponent that means very little, which is exactly the
    failure Result 53 recorded.  Reporting one without the other would repeat it.
    """
    sizes = np.asarray(sizes, dtype=float)
    counts = np.asarray(counts, dtype=float)
    keep = (sizes > 0) & (counts > 0)
    sizes, counts = sizes[keep], counts[keep]
    if len(sizes) < 3:
        return float("nan"), float("nan")

    x, y = np.log(sizes), np.log(counts)
    slope, intercept = np.polyfit(x, y, 1)

    residuals = y - (slope * x + intercept)
    dof = len(x) - 2
    if dof <= 0:
        return float(slope), float("nan")
    variance = float(np.sum(residuals**2) / dof)
    spread = float(np.sum((x - x.mean()) ** 2))
    return float(slope), float(np.sqrt(variance / spread)) if spread > 0 else float("nan")


def measure(name: str, budget_seconds: float, max_operators: int) -> dict:
    """One molecule: ADAPT to chemical accuracy, and UCCSD's fixed count."""
    started = time.perf_counter()
    problem = build_molecule(name)
    orbitals = problem.num_spatial_orbitals

    row = {
        "molecule": name,
        "orbitals": orbitals,
        "qubits": problem.hamiltonian.num_qubits,
        "fci_energy": float(problem.fci_energy),
        "correlation_energy": float(problem.correlation_energy),
        "build_seconds": time.perf_counter() - started,
    }

    # the fixed-ansatz reference, which is just a construction -- no optimisation
    circuit = uccsd(problem)
    row["uccsd_parameters"] = int(circuit.num_parameters)

    adapt_started = time.perf_counter()
    result = adapt_vqe(
        problem,
        energy_tolerance=CHEMICAL_ACCURACY,
        max_operators=max_operators,
    )
    error = abs(result.energy - problem.fci_energy)

    row.update({
        "adapt_parameters": int(result.num_parameters),
        "adapt_error": float(error),
        "adapt_converged": bool(result.converged),
        # the honest gate: a run that missed the target does not enter the fit
        "reached_chemical_accuracy": bool(error < CHEMICAL_ACCURACY),
        "adapt_seconds": time.perf_counter() - adapt_started,
        "hit_operator_cap": int(result.num_parameters) >= max_operators,
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="+", default=list(DEFAULT_MOLECULES))
    ap.add_argument("--max-operators", type=int, default=60)
    ap.add_argument("--budget", type=float, default=5400.0,
                    help="seconds per molecule before giving up on the series")
    args = ap.parse_args()

    print("=== experiment 018 :: does the adaptive ansatz move an exponent? ===")
    print(f"homologous series, chemical accuracy {CHEMICAL_ACCURACY:.1e} Ha vs FCI")
    print("a run that misses the target is reported, not folded into the fit\n")

    header = (
        f"{'molecule':<9}{'N':>3}{'qubits':>8}{'UCCSD n':>9}{'ADAPT n':>9}"
        f"{'ratio':>8}{'error':>11}{'reached':>9}{'seconds':>10}"
    )
    print(header)
    print("-" * len(header))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp018_adapt_parameter_scaling.json"

    rows: list[dict] = []

    def save() -> None:
        """Incremental -- a timeout on H8 must not lose H2 through H6."""
        sizes = [r["orbitals"] for r in rows if r["reached_chemical_accuracy"]]
        adapt_n = [r["adapt_parameters"] for r in rows if r["reached_chemical_accuracy"]]
        uccsd_n = [r["uccsd_parameters"] for r in rows if r["reached_chemical_accuracy"]]
        adapt_fit = fit_exponent(sizes, adapt_n)
        uccsd_fit = fit_exponent(sizes, uccsd_n)
        with open(path, "w") as fh:
            json.dump({
                "chemical_accuracy": CHEMICAL_ACCURACY,
                "rows": rows,
                "points_in_fit": len(sizes),
                "adapt_exponent": adapt_fit[0], "adapt_exponent_stderr": adapt_fit[1],
                "uccsd_exponent": uccsd_fit[0], "uccsd_exponent_stderr": uccsd_fit[1],
            }, fh, indent=2)

    for name in args.molecules:
        try:
            row = measure(name, args.budget, args.max_operators)
        except Exception as exc:  # noqa: BLE001 -- report, never skip silently
            print(f"{name:<9}{'FAILED':>20}  {type(exc).__name__}: {exc}", flush=True)
            rows.append({"molecule": name, "failed": f"{type(exc).__name__}: {exc}",
                         "reached_chemical_accuracy": False})
            save()
            continue

        ratio = (row["uccsd_parameters"] / row["adapt_parameters"]
                 if row["adapt_parameters"] else float("nan"))
        print(
            f"{row['molecule']:<9}{row['orbitals']:>3}{row['qubits']:>8}"
            f"{row['uccsd_parameters']:>9}{row['adapt_parameters']:>9}"
            f"{ratio:>8.1f}{row['adapt_error']:>11.2e}"
            f"{'yes' if row['reached_chemical_accuracy'] else 'NO':>9}"
            f"{row['adapt_seconds']:>10.1f}",
            flush=True,
        )
        if row["hit_operator_cap"]:
            print(f"{'':<9}^ hit the {args.max_operators}-operator cap; "
                  f"this count is a floor, not a measurement", flush=True)
        rows.append(row)
        save()

    # --------------------------------------------------------------- the fits
    usable = [r for r in rows if r.get("reached_chemical_accuracy")]
    print(f"\n{len(usable)} of {len(rows)} molecules reached chemical accuracy "
          f"and enter the fit.")

    if len(usable) < 3:
        print("Fewer than three points -- no exponent is reported. Three points")
        print("through two free parameters would fit anything.")
        save()
        print(f"\nsaved -> {path}")
        return 0

    sizes = [r["orbitals"] for r in usable]
    adapt_slope, adapt_err = fit_exponent(sizes, [r["adapt_parameters"] for r in usable])
    uccsd_slope, uccsd_err = fit_exponent(sizes, [r["uccsd_parameters"] for r in usable])

    print(f"\n{'ansatz':<10}{'exponent in N':>18}{'std error':>12}")
    print(f"{'UCCSD':<10}{uccsd_slope:>18.2f}{uccsd_err:>12.2f}")
    print(f"{'ADAPT':<10}{adapt_slope:>18.2f}{adapt_err:>12.2f}")

    print(f"\nUCCSD's textbook exponent is 4. Measured here: {uccsd_slope:.2f}.")
    print("That is the control -- if it does not come out near 4, the fit is")
    print("measuring something other than what it claims to.")

    # what it means for the shots exponent, using the measured Sigma|c| scaling
    # for the basis-limit direction (Result 53: N^2.86 in Sigma|c|, so N^5.7 squared)
    sigma_exponent = 5.7
    print(f"\nCarried into shots ~ (Sigma|c|)^2 * n / eps^2, with the measured")
    print(f"(Sigma|c|)^2 ~ N^{sigma_exponent} for the basis-limit direction:")
    print(f"  UCCSD:  N^{sigma_exponent + uccsd_slope:.1f}")
    print(f"  ADAPT:  N^{sigma_exponent + adapt_slope:.1f}")
    print(f"  CCSD(T) classical competitor:  N^7 in operations")

    if adapt_err == adapt_err and adapt_err > 0:  # not nan
        low, high = adapt_slope - 2 * adapt_err, adapt_slope + 2 * adapt_err
        print(f"\nADAPT's exponent at +-2 standard errors: [{low:.2f}, {high:.2f}]")
        if high >= uccsd_slope:
            print("That interval reaches UCCSD's exponent, so on this data the")
            print("two are NOT distinguishable. The advantage is not established")
            print("as a scaling effect, whatever the ratios look like.")
        else:
            print("That interval excludes UCCSD's exponent, so the reduction is")
            print("a genuine change in scaling rather than a constant factor.")

    print(f"\nPoints in fit: {len(usable)}. Read the standard error, not the slope:")
    print("four points can produce a confident-looking exponent that means little,")
    print("which is exactly what Result 53 recorded.")

    save()
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
