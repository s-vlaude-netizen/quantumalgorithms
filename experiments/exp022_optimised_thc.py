"""Experiment 022 -- does an *optimised* chi reach the linear rank a selected one misses?

Result 77 fitted the THC form by **selecting** chi from the double-factorisation
eigenbasis and solving for the coupling.  It split: the 1-norm delivered
(0.56x the Pauli norm at N=8, 6.8x below DF's) but the rank did not --
``M ~ N^2.26 +- 0.13``, where THC's whole argument is ``M ~ O(N)``.

It also named the reason.  A selection cannot beat its pool, and published THC
does the harder thing: it **optimises** chi and Z jointly by nonlinear least
squares.  So the open question was left as one sharp comparison, and this is it:

    does an optimised chi reach M ~ N, where a selected one gives N^2.26?

Everything else is held fixed against Result 77 -- the same molecules, the same
energy criterion (rebuild the Hamiltonian, compare exact ground energies, since a
tensor residual is not a physical error), and Result 77's guards -- plus a third
that this experiment turned out to need:

* **stability** -- refit from a perturbed start; a 1-norm that moves is an
  artefact of the landscape, not a property of the molecule
* **convergence in rank** -- a 1-norm that falls when the rank *rises* had not
  converged; Result 77's H8 passed the first guard at M=64 with lambda=1548
  while M=80 gave 50
* **optimiser convergence** -- new here, and it was needed: with the tolerances
  first chosen, *every* H6 rank exhausted the iteration budget. A fit that runs
  out of budget is not a measurement whatever its residual reads.

**And then the guards were still not enough.** With all three in place, H4 at
M=12 gave 3.4e-8 on one run and 9.3e-3 on the next -- five orders of magnitude,
both *converged*, in 847 and 262 iterations. The landscape has multiple local
minima and floating-point noise decides which one L-BFGS reaches, so a single
start is not a measurement of anything. Every fit below is therefore the best of
``RESTARTS`` starts, and the spread across them is reported rather than hidden:
it is the honest uncertainty on the rank, and it is large.

The gradient below is analytic and was checked against finite differences to
1.4e-8 before being used.  A first derivation was wrong by a factor on both
blocks and would have optimised a different objective than the one documented.

Run:  python -m experiments.exp022_optimised_thc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize

from qres.bench import RESULTS_DIR
from qres.factorization import molecular_integrals, pauli_one_norm
from qres.problems.chemistry import build_molecule

from experiments.exp021_tensor_hypercontraction import (
    CHEMICAL_ACCURACY,
    CONVERGENCE_FACTOR,
    STABILITY_TOLERANCE,
    energy_of,
    thc_candidates,
)

RANK_MULTIPLES = (1, 2, 3, 4, 6, 8)
DEFAULT_MOLECULES = ("H2", "H4", "H6", "H8")
MAX_ITERATIONS = 20000

#: Independent starts per rank.  One is not enough -- see the module docstring.
RESTARTS = 6


def thc_tensor(chi: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    """``T_pqrs = sum_{mn} chi_pm chi_qm Z_mn chi_rn chi_sn``."""
    outer = np.einsum("pm,qm->mpq", chi, chi)
    return np.einsum("mpq,mn,nrs->pqrs", outer, coupling, outer)


def _objective(packed, orbitals, rank, target):
    """Residual and its analytic gradient.

    The gradient was verified against central differences to 1.4e-8 (chi) and
    3.6e-6 (Z, finite-difference limited).  The `2 *` factors come from p<->q and
    r<->s symmetry of the residual; a first version folded them into a single
    `4 *` on one block and symmetrised Z's gradient as well, which was wrong on
    both counts.
    """
    split = orbitals * rank
    chi = packed[:split].reshape(orbitals, rank)
    coupling = packed[split:].reshape(rank, rank)

    outer = np.einsum("pm,qm->mpq", chi, chi)
    residual = np.einsum("mpq,mn,nrs->pqrs", outer, coupling, outer) - target

    value = 0.5 * float(np.sum(residual**2))
    grad_coupling = np.einsum("pqrs,mpq,nrs->mn", residual, outer, outer)
    forward = np.einsum("mn,nrs->mrs", coupling, outer)
    backward = np.einsum("npq,nm->mpq", outer, coupling)
    grad_chi = 2 * np.einsum("aqrs,qm,mrs->am", residual, chi, forward) + 2 * np.einsum(
        "pqas,sm,mpq->am", residual, chi, backward
    )
    return value, np.concatenate([grad_chi.ravel(), grad_coupling.ravel()])


def _single_fit(one_body, two_body, rank, chi, iterations):
    """One L-BFGS run from a given start."""
    orbitals = one_body.shape[0]
    target = two_body
    coupling = np.zeros((rank, rank))

    packed = np.concatenate([chi.ravel(), coupling.ravel()])
    result = minimize(
        _objective, packed, args=(orbitals, rank, target),
        jac=True, method="L-BFGS-B",
        # ftol/gtol tight enough for a real fit, loose enough that L-BFGS can
        # actually *reach* them.  A first version used 1e-16/1e-12 and hit the
        # iteration cap on every H6 rank -- so nothing was converged, and two
        # runs of the same H4 point disagreed (7.6e-3 vs 2.2e-4) because BLAS
        # threading noise compounds over thousands of iterations near a flat
        # minimum.  Exhausting the budget is the signal, not a detail.
        options={"maxiter": iterations, "maxfun": iterations * 2, "ftol": 1e-14,
                 "gtol": 1e-10},
    )

    split = orbitals * rank
    chi = result.x[:split].reshape(orbitals, rank)
    coupling = result.x[split:].reshape(rank, rank)
    coupling = 0.5 * (coupling + coupling.T)

    reconstructed = thc_tensor(chi, coupling)
    # chi is no longer unit-norm after optimisation, so the 1-norm has to absorb
    # the column norms -- omitting this would understate lambda by |chi|^4
    scale = np.linalg.norm(chi, axis=0) ** 2
    one_norm = float(0.5 * np.abs(coupling * np.outer(scale, scale)).sum())

    return {
        "rank": rank,
        "residual": float(result.fun),
        "two_body": reconstructed,
        "tensor_error": float(np.max(np.abs(reconstructed - target))),
        "one_norm": one_norm,
        "iterations": int(result.nit),
        # a fit that exhausted its budget has not converged, whatever its
        # residual happens to be at the cutoff
        "converged_optimiser": bool(result.success and result.nit < iterations),
        "chi": chi,
        "coupling": coupling,
    }


def optimised_thc_fit(one_body, two_body, rank, ridge=1e-8, iterations=MAX_ITERATIONS,
                      restarts=RESTARTS):
    """Best of several starts, with the spread across them reported.

    The first start is Result 77's selection so the comparison isolates the
    *fitting*; the rest are perturbations of it.  Taking the best is what a real
    THC fit does; reporting the spread is what makes the number honest.
    """
    orbitals = one_body.shape[0]
    base = thc_candidates(one_body, two_body)[:rank].T.copy()
    rng = np.random.default_rng(20260907)
    if base.shape[1] < rank:
        base = np.hstack([base, rng.normal(scale=1e-3,
                                           size=(orbitals, rank - base.shape[1]))])

    attempts = []
    for index in range(max(1, restarts)):
        start = base if index == 0 else base + rng.normal(scale=0.1 * (index),
                                                          size=base.shape)
        attempts.append(_single_fit(one_body, two_body, rank, start, iterations))

    best = min(attempts, key=lambda a: a["residual"])
    residuals = [a["residual"] for a in attempts]
    best["restart_residual_spread"] = float(max(residuals) / max(min(residuals), 1e-300))
    best["restart_one_norms"] = [a["one_norm"] for a in attempts]
    best["restarts"] = len(attempts)
    return best


def _stability(one_body, two_body, rank, tight):
    """Refit from a perturbed start; a real optimum should not move."""
    rng = np.random.default_rng(12345)
    orbitals = one_body.shape[0]
    perturbed = tight["chi"] + rng.normal(scale=1e-3, size=tight["chi"].shape)
    packed = np.concatenate([perturbed.ravel(), tight["coupling"].ravel()])
    result = minimize(
        _objective, packed, args=(orbitals, rank, two_body),
        jac=True, method="L-BFGS-B",
        # identical tolerances to the fit being checked -- comparing against a
        # refit run to a different stopping rule would measure the tolerances
        options={"maxiter": MAX_ITERATIONS, "maxfun": MAX_ITERATIONS * 2,
                 "ftol": 1e-14, "gtol": 1e-10},
    )
    split = orbitals * rank
    chi = result.x[:split].reshape(orbitals, rank)
    coupling = result.x[split:].reshape(rank, rank)
    coupling = 0.5 * (coupling + coupling.T)
    scale = np.linalg.norm(chi, axis=0) ** 2
    return float(0.5 * np.abs(coupling * np.outer(scale, scale)).sum())


def measure(name: str, multiples=RANK_MULTIPLES) -> dict:
    started = time.perf_counter()
    problem = build_molecule(name)
    one_body, two_body, _ = molecular_integrals(problem)
    orbitals = one_body.shape[0]
    reference = energy_of(problem, one_body, two_body)

    sweep, seen = [], set()
    for multiple in multiples:
        rank = multiple * orbitals
        if rank in seen:
            continue
        seen.add(rank)
        fit = optimised_thc_fit(one_body, two_body, rank)
        error = abs(energy_of(problem, one_body, fit["two_body"]) - reference)

        perturbed_norm = _stability(one_body, two_body, rank, fit)
        largest = max(fit["one_norm"], perturbed_norm, 1e-12)
        stable = abs(fit["one_norm"] - perturbed_norm) / largest < STABILITY_TOLERANCE

        row = {
            "rank": rank,
            "rank_over_orbitals": rank / orbitals,
            "energy_error": error,
            "tensor_error": fit["tensor_error"],
            "one_norm": fit["one_norm"],
            "one_norm_perturbed": perturbed_norm,
            "stable": bool(stable),
            "iterations": fit["iterations"],
            "optimiser_converged": fit["converged_optimiser"],
            # persisted, not just printed: a spread that is only in stdout
            # cannot be checked by a test later, and this one is the whole
            # justification for the multi-start machinery
            "restart_residual_spread": fit["restart_residual_spread"],
            "restart_one_norms": fit["restart_one_norms"],
            "restarts": fit["restarts"],
            "within_chemical_accuracy": bool(error < CHEMICAL_ACCURACY),
        }
        sweep.append(row)
        # Stop two ranks past the first valid one.  The larger ranks exist only
        # to check convergence in lambda, and two suffice for that -- grinding
        # through M/N=6 and 8 on H8 costs hours per rank (cost ~ M^2 N^4 per
        # objective evaluation, times restarts, times the stability refit) and
        # cannot change a threshold already found below them.
        valid = [
            index for index, entry in enumerate(sweep)
            if entry["within_chemical_accuracy"] and entry["stable"]
            and entry["optimiser_converged"]
        ]
        stop_after = valid[0] + 2 if valid else None
        print(
            f"  {name:<4} M={rank:>4} (M/N={rank / orbitals:>3.0f})  "
            f"energy err {error:>9.2e}  lambda {fit['one_norm']:>9.2f}  "
            f"spread {fit['restart_residual_spread']:>8.1e}"
            f"{'' if stable else ' UNSTABLE'}"
            f"{'' if fit['converged_optimiser'] else ' NOT-CONVERGED'}",
            flush=True,
        )
        if stop_after is not None and len(sweep) > stop_after:
            print(f"  {name:<4} stopping: two ranks past the first valid one",
                  flush=True)
            break

    smallest = None
    for index, row in enumerate(sweep):
        if not (row["within_chemical_accuracy"] and row["stable"]
                and row["optimiser_converged"]):
            continue
        larger = [r["one_norm"] for r in sweep[index + 1:] if r["stable"]]
        if larger and row["one_norm"] > CONVERGENCE_FACTOR * min(larger):
            row["converged"] = False
            continue
        row["converged"] = True
        smallest = row
        break

    return {
        "molecule": name,
        "orbitals": orbitals,
        "pauli_one_norm": pauli_one_norm(problem.hamiltonian),
        "sweep": sweep,
        "smallest_sufficient": smallest,
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="+", default=list(DEFAULT_MOLECULES))
    args = ap.parse_args()

    print("=== experiment 022 :: optimised THC, against Result 77's selected chi ===")
    print("Same molecules and energy criterion as Result 77. Only the fit changes:")
    print("chi and Z optimised jointly instead of chi selected, best of")
    print(f"{RESTARTS} restarts, with three guards (stability, rank convergence,")
    print("optimiser convergence) rather than Result 77's two.\n")

    rows = []
    for name in args.molecules:
        try:
            rows.append(measure(name))
        except Exception as exc:  # noqa: BLE001 -- report, never skip silently
            print(f"  {name:<4} FAILED  {type(exc).__name__}: {exc}", flush=True)
        print()

    usable = [r for r in rows if r["smallest_sufficient"]]
    print("--- rank needed for chemical accuracy ---")
    print(f"{'molecule':<9}{'N':>3}{'M':>6}{'M/N':>6}{'lambda':>10}{'lam/Pauli':>11}")
    print("-" * 45)
    for row in usable:
        best = row["smallest_sufficient"]
        print(f"{row['molecule']:<9}{row['orbitals']:>3}{best['rank']:>6}"
              f"{best['rank_over_orbitals']:>6.0f}{best['one_norm']:>10.2f}"
              f"{best['one_norm'] / row['pauli_one_norm']:>11.2f}")

    exponent = stderr = None
    if len(usable) >= 3:
        sizes = np.array([r["orbitals"] for r in usable], dtype=float)
        ranks = np.array([r["smallest_sufficient"]["rank"] for r in usable], dtype=float)
        fit = np.polyfit(np.log(sizes), np.log(ranks), 1)
        residual = np.log(ranks) - np.polyval(fit, np.log(sizes))
        spread = float(np.sum((np.log(sizes) - np.log(sizes).mean()) ** 2))
        stderr = float(np.sqrt(np.sum(residual**2) / max(len(sizes) - 2, 1) / spread))
        exponent = float(fit[0])

        print(f"\n**Optimised: M ~ N^{exponent:.2f} +- {stderr:.2f}**")
        print(f"  Result 77, selected chi:  M ~ N^2.26 +- 0.13")
        print(f"  THC's claim:              M ~ N^1")
        low, high = exponent - 2 * stderr, exponent + 2 * stderr
        print(f"\nAt +-2 standard errors: [{low:.2f}, {high:.2f}]")
        if low <= 1.0 <= high:
            print("-> consistent with linear. Optimising chi recovers the claim that")
            print("   selection missed, and the reason for Result 77's failure is")
            print("   confirmed to be the selection rather than the THC form.")
        elif high < 2.26 - 2 * 0.13:
            print("-> below the selected exponent but above linear: optimisation")
            print("   helps and does not fully close the gap.")
        else:
            print("-> not distinguishable from the selected fit. Optimising chi is")
            print("   NOT what separates this construction from the published one.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp022_optimised_thc.json"
    with open(path, "w") as fh:
        json.dump({
            "chemical_accuracy": CHEMICAL_ACCURACY,
            "rows": [{k: v for k, v in r.items()} for r in rows],
            "rank_exponent": exponent,
            "rank_exponent_stderr": stderr,
            "selected_exponent_result_77": 2.26,
            "selected_exponent_stderr_result_77": 0.13,
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
