"""Experiment 021 -- tensor hypercontraction: what rank does it actually need?

Result 76 measured double factorisation as a block encoding (332x at 50 orbitals)
and left a measured gap of ~200x to Lee et al.'s tensor hypercontraction.  THC is
the open item, and the question that decides it is **not** the cost formula --
it is the rank:

    g_pqrs ~ sum_{mu,nu} chi_p^mu chi_q^mu Z_{mu,nu} chi_r^nu chi_s^nu

THC's entire claim is that ``M``, the number of auxiliary vectors, grows as
**O(N)** where the Pauli representation grows as O(N^4) and double factorisation
needs O(N) factors each carrying an O(N^2) rotation network.  If M really is
O(N), the per-walk cost collapses.  If it is not, the whole approach is a
constant-factor rearrangement.

**So this measures M, and it measures it against the energy rather than a tensor
norm.**  A residual in the Coulomb tensor is not a physical error; the question
is whether the *ground-state energy* of the reconstructed Hamiltonian is within
chemical accuracy.  The rebuild path is verified exact to 3.6e-12 against
`build_molecule`'s own FCI, so any error measured here is the approximation's.

**What is measured and what is not.**  The rank M, the reconstruction energy
error, and the resulting 1-norm are measured from real integrals.  The *cost per
walk* is a model, and a rougher one than Result 76's -- THC's published
constructions use QROM-based state preparation this repository does not
implement, so the projection here is deliberately conservative and is reported
separately from the measurements.  Result 72 is the standing rule: none of this
is novel, and the published version is better than this one.

Run:  python -m experiments.exp021_tensor_hypercontraction
      python -m experiments.exp021_tensor_hypercontraction --molecules H2 H4 H6
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
from qres.factorization import double_factorize, molecular_integrals, pauli_one_norm
from qres.problems.chemistry import build_molecule, exact_ground_energy

CHEMICAL_ACCURACY = 1.6e-3

#: ranks to try, as multiples of the orbital count
RANK_MULTIPLES = (1, 2, 3, 4, 5, 6, 8, 10, 12)

DEFAULT_MOLECULES = ("H2", "H4", "H6", "H8")

#: Tikhonov parameter for the coupling solve.  Small enough not to bias an
#: honest fit, large enough to stop a degenerate one running away.
RIDGE = 1e-8

#: A fit whose 1-norm moves when the regulariser moves is not a fit, it is an
#: interpolation through a near-singular design.  ``lambda`` at ``RIDGE`` and at
#: ``100 * RIDGE`` must agree this closely for the rank to count.
STABILITY_TOLERANCE = 0.10

#: A threshold rank's 1-norm must not exceed this multiple of any larger rank's.
#: Enforces convergence rather than mere numerical stability.
CONVERGENCE_FACTOR = 2.0


def _ridge_solve(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Symmetric Tikhonov-regularised solve of ``design Z design^T = target``."""
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    filtered = singular / (singular**2 + alpha)
    inverse = (right.T * filtered) @ left.T
    coupling = inverse @ target @ inverse.T
    return 0.5 * (coupling + coupling.T)


def thc_candidates(one_body: np.ndarray, two_body: np.ndarray) -> np.ndarray:
    """Candidate THC vectors, ordered by the weight they carry.

    Double factorisation already produces an exact representation as a sum of
    squared one-body operators, each diagonalised by its own rotation.  Every
    eigenvector of every factor is therefore a candidate ``chi``, and the DF
    weights say which ones matter.  THC's compression is the claim that O(N) of
    these ~N^2 candidates suffice -- which makes the selection the measurement.
    """
    factorised = double_factorize(one_body, two_body)
    orbitals = one_body.shape[0]

    vectors, weights = [], []
    for weight, rotation, diagonal in zip(
        factorised.factor_weights,
        factorised.factor_rotations,
        factorised.factor_diagonals,
    ):
        for column in range(orbitals):
            vectors.append(rotation[:, column])
            weights.append(abs(weight) * abs(diagonal[column]))

    order = np.argsort(-np.asarray(weights))
    return np.asarray(vectors)[order]


def thc_fit(one_body: np.ndarray, two_body: np.ndarray, rank: int) -> dict:
    """Fit the THC form at a given rank and report its 1-norm and residual."""
    orbitals = one_body.shape[0]
    target = two_body.reshape(orbitals**2, orbitals**2)
    target = 0.5 * (target + target.T)

    chi = thc_candidates(one_body, two_body)[:rank]
    design = np.array([np.outer(v, v).ravel() for v in chi]).T

    # The design matrix is severely ill-conditioned -- 1e15 to 1e18 measured --
    # because outer products of near-degenerate eigenvectors are near-degenerate.
    # A plain pseudo-inverse then returns a Z with entries around 1e26 that
    # cancel to reproduce the tensor, and `lambda`, the quantity the algorithm
    # cost actually depends on, becomes meaningless.  A first version did exactly
    # that and would have reported lambda = 116109 for H4 as if it meant
    # something.  The giveaway was non-monotonicity: the energy error went
    # 3e-3 -> 5e10 -> 1.6e-12 as the rank grew, and more information cannot make
    # a fit worse.
    coupling = _ridge_solve(design, target, RIDGE)
    # Z must be symmetric for the reconstruction to carry chemists' 8-fold
    # symmetry.  The pinv solve leaves asymmetry at ~1e-16, which is enough for
    # qiskit-nature's index-order detection to reject the tensor outright -- and
    # a first version of this experiment therefore crashed at M/N >= 3 on every
    # molecule but H2, truncating the sweep before THC could show anything.
    coupling = 0.5 * (coupling + coupling.T)

    reconstructed = design @ coupling @ design.T
    reconstructed = 0.5 * (reconstructed + reconstructed.T)
    approx_two_body = reconstructed.reshape(orbitals, orbitals, orbitals, orbitals)

    # stability: does the 1-norm survive a hundredfold change in the regulariser?
    loose = _ridge_solve(design, target, RIDGE * 100)
    loose_norm = float(0.5 * np.abs(loose).sum())
    tight_norm = float(0.5 * np.abs(coupling).sum())
    largest = max(tight_norm, loose_norm, 1e-12)
    stable = abs(tight_norm - loose_norm) / largest < STABILITY_TOLERANCE

    return {
        "rank": len(chi),
        "two_body": approx_two_body,
        "one_norm_loose": loose_norm,
        "stable": bool(stable),
        "tensor_error": float(np.max(np.abs(target - reconstructed))),
        # chi vectors are unit-norm eigenvectors, so the 1-norm is Z's.
        # NOTE: this is *not* comparable to a published THC lambda.  Real fits
        # optimise chi and Z jointly and penalise the 1-norm; the least-squares
        # solve here does neither, so its Z contains large cancelling entries and
        # lambda can grow with rank.  Measured and reported, not defended.
        "one_norm": float(0.5 * np.abs(coupling).sum()),
    }


def energy_of(problem, one_body: np.ndarray, two_body: np.ndarray) -> float:
    """Exact ground energy of the Hamiltonian built from given integrals."""
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import ParityMapper

    hamiltonian = ElectronicEnergy.from_raw_integrals(one_body, two_body)
    mapper = ParityMapper(num_particles=problem.num_particles)
    operator = mapper.map(hamiltonian.second_q_op()).simplify(atol=1e-12)
    return float(exact_ground_energy(operator))


def measure(name: str, multiples=RANK_MULTIPLES) -> dict:
    started = time.perf_counter()
    problem = build_molecule(name)
    one_body, two_body, _ = molecular_integrals(problem)
    orbitals = one_body.shape[0]

    # the reference is the rebuild at full information, not build_molecule's FCI,
    # so the rebuild path itself cannot contribute to the measured error
    reference = energy_of(problem, one_body, two_body)

    sweep, smallest, seen = [], None, set()
    for multiple in multiples:
        rank = multiple * orbitals
        fit = thc_fit(one_body, two_body, rank)
        if fit["rank"] in seen:
            continue  # the rank was capped by the candidate pool; no new point
        seen.add(fit["rank"])
        error = abs(energy_of(problem, one_body, fit["two_body"]) - reference)
        row = {
            "rank": fit["rank"],
            "rank_over_orbitals": fit["rank"] / orbitals,
            "energy_error": error,
            "tensor_error": fit["tensor_error"],
            "one_norm": fit["one_norm"],
            "stable": fit["stable"],
            "within_chemical_accuracy": bool(error < CHEMICAL_ACCURACY),
        }
        sweep.append(row)
        print(
            f"  {name:<5} M={fit['rank']:>4} (M/N={fit['rank'] / orbitals:>3.0f})  "
            f"energy err {error:>9.2e}  tensor err {fit['tensor_error']:>9.2e}  "
            f"lambda {fit['one_norm']:>10.2f}{'' if fit['stable'] else ' UNSTABLE'}"
            f"{'   <-- chemical accuracy' if row['within_chemical_accuracy'] and smallest is row else ''}",
            flush=True,
        )

    # Choosing the threshold rank needs a *converged* lambda, not merely one that
    # survives the regulariser.  H8 at M=64 passed the ridge-stability check with
    # lambda = 1548 while M=80 gave 50 -- a thirtyfold drop from *more*
    # information, which means the smaller fit had not converged at all.  So a
    # rank counts only if its lambda is within CONVERGENCE_FACTOR of every larger
    # rank's, which is the property "more information does not change the answer".
    for index, row in enumerate(sweep):
        if not (row["within_chemical_accuracy"] and row["stable"]):
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
        "qubits": problem.hamiltonian.num_qubits,
        "pauli_terms": len(problem.hamiltonian),
        "pauli_one_norm": pauli_one_norm(problem.hamiltonian),
        "reference_energy": reference,
        "sweep": sweep,
        "smallest_sufficient": smallest,
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="+", default=list(DEFAULT_MOLECULES))
    args = ap.parse_args()

    print("=== experiment 021 :: tensor hypercontraction, and the rank it needs ===")
    print("Rank chosen against the ENERGY of the reconstructed Hamiltonian, not a")
    print("tensor norm. Rebuild path verified exact to 3.6e-12.\n")

    rows = []
    for name in args.molecules:
        try:
            rows.append(measure(name))
        except Exception as exc:  # noqa: BLE001 -- report, never skip silently
            print(f"  {name:<5} FAILED  {type(exc).__name__}: {exc}", flush=True)
        print()

    usable = [r for r in rows if r["smallest_sufficient"]]
    print("--- the rank needed for chemical accuracy ---")
    print(f"{'molecule':<9}{'N':>3}{'M':>6}{'M/N':>6}{'lam_THC':>10}"
          f"{'lam_Pauli':>11}{'ratio':>8}")
    print("-" * 53)
    for row in usable:
        best = row["smallest_sufficient"]
        print(f"{row['molecule']:<9}{row['orbitals']:>3}{best['rank']:>6}"
              f"{best['rank_over_orbitals']:>6.0f}{best['one_norm']:>10.2f}"
              f"{row['pauli_one_norm']:>11.2f}"
              f"{best['one_norm'] / row['pauli_one_norm']:>8.2f}")

    if len(usable) >= 3:
        sizes = np.array([r["orbitals"] for r in usable], dtype=float)
        ranks = np.array([r["smallest_sufficient"]["rank"] for r in usable], dtype=float)
        norms = np.array([r["smallest_sufficient"]["one_norm"] for r in usable], dtype=float)

        rank_fit = np.polyfit(np.log(sizes), np.log(ranks), 1)
        norm_fit = np.polyfit(np.log(sizes), np.log(norms), 1)
        residual = np.log(ranks) - np.polyval(rank_fit, np.log(sizes))
        stderr = float(np.sqrt(np.sum(residual**2) / max(len(sizes) - 2, 1)
                               / np.sum((np.log(sizes) - np.log(sizes).mean()) ** 2)))

        print(f"\n**Rank scaling: M ~ N^{rank_fit[0]:.2f} +- {stderr:.2f}** "
              f"({len(usable)} points)")
        print(f"THC 1-norm scaling: lambda ~ N^{norm_fit[0]:.2f}")
        print("\nTHC's claim is M ~ O(N), i.e. exponent 1. Against the Pauli")
        print("representation's O(N^4) that is the whole argument for it.")
        low, high = rank_fit[0] - 2 * stderr, rank_fit[0] + 2 * stderr
        print(f"Measured exponent at +-2 standard errors: [{low:.2f}, {high:.2f}]")
        if low <= 1.0 <= high:
            print("-> consistent with linear, which is what THC claims")
        elif low > 1.0:
            print("-> ABOVE linear: the claim does not hold on this data")
        else:
            print("-> below linear on this data, which would be better than claimed")
    else:
        print("\nFewer than three molecules reached chemical accuracy; no fit.")
        rank_fit = norm_fit = None
        stderr = float("nan")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp021_tensor_hypercontraction.json"
    with open(path, "w") as fh:
        json.dump({
            "chemical_accuracy": CHEMICAL_ACCURACY,
            "rank_multiples": list(RANK_MULTIPLES),
            "rows": [{k: v for k, v in r.items() if k != "sweep"} | {
                "sweep": [{kk: vv for kk, vv in s.items()} for s in r["sweep"]]
            } for r in rows],
            "rank_exponent": float(rank_fit[0]) if rank_fit is not None else None,
            "rank_exponent_stderr": stderr,
            "one_norm_exponent": float(norm_fit[0]) if norm_fit is not None else None,
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
