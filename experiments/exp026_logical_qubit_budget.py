"""Experiment 026 -- what does a 50-logical-qubit machine buy for drug metabolism?

The question this answers came in as: given a machine with ~50 logical qubits,
could you simulate just the *pocket* of a drug molecule that touches its target
-- keeping under ~50 atoms -- and learn something about pharmacokinetics?

Three translations have to happen before that is answerable, and two of them
move the answer by orders of magnitude.

**Logical qubits are not atoms.**  A qubit carries one spin orbital, so 50
logical qubits is **25 spatial orbitals**.  A 50-atom organic fragment has a few
hundred basis functions even in a minimal basis -- roughly 200 spatial orbitals,
so ~400 qubits.  The gap is about an order of magnitude, and it is not closed by
being clever about which atoms to include.  What 25 orbitals actually buys is an
**active space**: a handful of frontier orbitals treated exactly, embedded in a
classical mean-field description of everything else.  That is the standard
technique and it is the right frame for the question.

**Pharmacokinetics is mostly not an electronic-structure problem.**  Absorption,
distribution and excretion are conformational and solvation thermodynamics over
nanoseconds to microseconds and thousands of atoms -- force-field territory, and
a 25-orbital ground-state energy says nothing about them.  There is one large
exception, and it is the one that decides clearance and half-life:
**metabolism by cytochrome P450**, whose Compound I intermediate is a high-valent
iron-oxo species with genuine multireference character.  That step *is* an
electronic-structure problem at a transition-metal centre, and it is a published
quantum-advantage candidate (Goings et al., PNAS 2022, arXiv:2202.01244, which
compares DMRG against qubitized phase estimation for exactly these models and
concludes it "has the potential to be a quantum advantage problem").

So the user's instinct points at a real target.  The question is whether the
machine reaches it, and this sizes that with the repository's own measured laws
rather than with the announcement's headline numbers.

**Both criteria are applied, as everywhere else in this project:**

1. *is there a classical gap?*  Exact diagonalisation of a CAS(N, N) active space
   needs `C(N, N/2)^2` determinants, and that wall is passed well before N = 25.
   But exact diagonalisation is not the classical baseline -- DMRG and selected
   CI are, and they reach considerably further.  So the honest statement is that
   the gap **opens** near this size rather than that it is established, and the
   comparison that would settle it is against DMRG, which this repository does
   not have.
2. *does the hardware reach it?*  Sized two ways: the variational route against
   Result 66's product criterion, and the qubitized phase-estimation route
   against Results 69-71's surface-code model, both at a trapped-ion physical
   error rate of 7.9e-4.

**The extrapolation caveat, which Result 53 exists to enforce.**  The gate-count
law below is fitted on a hydrogen chain, which grows by *adding atoms* at a fixed
basis.  An active space grows in the *other* direction -- more orbitals on a
fixed molecule -- and Result 53 measured those two directions giving different
exponents for the 1-norm (`N^-0.39` against `N^2.86`).  The extrapolation to
N = 25 is therefore indicative and is reported with its standard error, not as a
precise count.

Run:  python -m experiments.exp026_logical_qubit_budget
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qres.ansatz import build_ansatz
from qres.bench import RESULTS_DIR
from qres.problems.chemistry import build_molecule

from experiments.exp016_error_correction_overhead import (
    FACTORY_LOGICAL_QUBITS,
    distillation_rounds,
    required_distance,
)
from experiments.exp020_block_encoding import (
    CHEMICAL_ACCURACY,
    double_factorised_cost,
)
from experiments.exp025_trapped_ion import (
    HELIOS_TWO_QUBIT_ERROR,
    count_across_seeds,
)

#: Helios, published: 98 physical qubits, 48 logical at a 2:1 encoding
HELIOS_PHYSICAL_QUBITS = 98
HELIOS_LOGICAL_QUBITS = 48

#: logical-qubit budgets to translate into chemistry
BUDGETS = (48, 50, 100, 400, 1000)

#: the homologous series the gate law is fitted on -- see the docstring caveat
CHAIN = ("H2", "H4", "H6")

#: exponents measured in Result 76 on the same chain, used to extrapolate the
#: double-factorised block encoding to an active space this repo cannot build
DF_ONE_NORM_EXPONENT = 1.9289057239102216
DF_FACTOR_EXPONENT = 1.6481193259074804

#: surface code: 2d^2 physical qubits per logical qubit (data plus syndrome)
PHYSICAL_PER_LOGICAL = 2


def determinants(orbitals: int, electrons: int | None = None) -> float:
    """Determinants in a CAS(electrons, orbitals) at half filling by default."""
    electrons = orbitals if electrons is None else electrons
    alpha = electrons // 2
    return float(math.comb(orbitals, alpha)) ** 2


def fit_power_law(sizes, values):
    """Log-log slope with a standard error -- never a slope on its own."""
    sizes, values = np.asarray(sizes, float), np.asarray(values, float)
    keep = (sizes > 0) & (values > 0)
    x, y = np.log(sizes[keep]), np.log(values[keep])
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    dof = len(x) - 2
    variance = float(np.sum(residual**2) / dof) if dof > 0 else float("nan")
    stderr = float(np.sqrt(variance / np.sum((x - x.mean()) ** 2))) if dof > 0 else float("nan")
    return float(slope), float(intercept), stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=",".join(CHAIN))
    ap.add_argument("--physical-error", type=float, default=HELIOS_TWO_QUBIT_ERROR)
    ap.add_argument("--target-orbitals", type=int, default=25)
    args = ap.parse_args()

    print("=== experiment 026 :: what does a 50-logical-qubit machine buy? ===")
    print("Device figures are Quantinuum's published Helios numbers, not measured")
    print("here. Resource models are this repository's (Results 66, 69-71, 76).\n")

    # ------------------------------------------------- 1. qubits into chemistry
    print("--- 1. what a logical qubit budget is, in chemistry ---")
    print(f"{'logical qubits':>15}{'spatial orbitals':>18}{'CAS determinants':>19}"
          f"{'exact diagonalisation?':>24}")
    print("-" * 76)
    translation = []
    for budget in BUDGETS:
        orbitals = budget // 2
        count = determinants(orbitals)
        # ~1e10 determinants is already a heroic exact diagonalisation
        feasible = "yes" if count < 1e10 else "no"
        translation.append({
            "logical_qubits": budget, "spatial_orbitals": orbitals,
            "determinants": count, "exact_diagonalisation": feasible,
        })
        print(f"{budget:>15}{orbitals:>18}{count:>19.2e}{feasible:>24}")
    print("\nExact diagonalisation is NOT the classical baseline -- DMRG and selected")
    print("CI reach further, and the P450 literature uses DMRG at bond dimension 1500.")
    print("So this table shows where the gap OPENS, not where it is established.")

    # --------------------------------------------- 2. the variational route
    print("\n--- 2. the variational route, against Result 66's product criterion ---")
    sizes, gates = [], []
    for name in args.chain.split(","):
        problem = build_molecule(name)
        circuit = build_ansatz("uccsd", problem)
        orbitals = problem.hamiltonian.num_qubits // 2
        counted = count_across_seeds(circuit, None)["median"]
        sizes.append(orbitals)
        gates.append(counted)
        print(f"  {name:>4}: {orbitals:>2} spatial orbitals, "
              f"{counted:>9.0f} two-qubit gates (all-to-all)")

    slope, intercept, stderr = fit_power_law(sizes, gates)
    target = args.target_orbitals
    predicted = float(np.exp(intercept) * target**slope)
    needed_error = CHEMICAL_ACCURACY / predicted
    shortfall = args.physical_error / needed_error

    print(f"\n  gates ~ N^{slope:.2f} +- {stderr:.2f}  (atom-count direction; see docstring)")
    print(f"  extrapolated to {target} orbitals: {predicted:.2e} two-qubit gates")
    print(f"  per-gate error that would need: {needed_error:.2e}")
    print(f"  best published trapped-ion rate: {args.physical_error:.2e}")
    print(f"  SHORT BY: {shortfall:.1e}x")

    # ------------------------------------------- 3. the phase-estimation route
    print("\n--- 3. the qubitized phase-estimation route, on error-corrected qubits ---")
    one_norm = float(target**DF_ONE_NORM_EXPONENT)
    factors = float(target**DF_FACTOR_EXPONENT)
    cost = double_factorised_cost(one_norm, target, factors, CHEMICAL_ACCURACY)
    t_gates = cost["t_gates"]

    # the whole computation must stay coherent, so the per-operation logical
    # error has to be below 1/(total operations)
    target_logical = 1.0 / t_gates
    distance = required_distance(target_logical, args.physical_error)
    rounds = distillation_rounds(args.physical_error, target_logical)

    logical_qubits = 2 * target                      # one per spin orbital
    print(f"  extrapolated lambda ~ {one_norm:.1f}, {factors:.0f} factors "
          f"(Result 76 exponents)")
    print(f"  T gates for one phase-estimation run: {t_gates:.2e}")
    print(f"  logical error that demands: {target_logical:.2e}")

    if distance is None:
        print(f"  surface-code distance needed: NONE below the search limit")
        physical = float("nan")
    else:
        data_physical = PHYSICAL_PER_LOGICAL * distance**2 * logical_qubits
        factory_physical = PHYSICAL_PER_LOGICAL * distance**2 * FACTORY_LOGICAL_QUBITS
        physical = float(data_physical + factory_physical)
        print(f"  surface-code distance: {distance}")
        print(f"  distillation rounds: {rounds}")
        print(f"  logical qubits: {logical_qubits} data + {FACTORY_LOGICAL_QUBITS} factory")
        print(f"  PHYSICAL QUBITS: {physical:.2e}")
        print(f"  Helios has {HELIOS_PHYSICAL_QUBITS} -- short by "
              f"{physical / HELIOS_PHYSICAL_QUBITS:.1e}x")

    # ------------------------------------------------------------- the verdict
    print("\n--- the answer ---")
    print(f"A 50-logical-qubit machine is a {args.target_orbitals}-orbital active space.")
    print("That is the right size to be interesting: it is past exact diagonalisation,")
    print("and it is the size the P450 literature actually studies.")
    print("\nBut the two routes to using it are short by:")
    print(f"  variational, on physical ion qubits: {shortfall:.1e}x in gate error")
    if distance is not None:
        print(f"  phase estimation, error-corrected:   "
              f"{physical / HELIOS_PHYSICAL_QUBITS:.1e}x in physical qubits")
    print("\nand the '48 logical qubits' figure is a 2:1 ENCODING, which is error")
    print("detection, not the deep surface code the second row assumes. The two")
    print("numbers are not the same kind of qubit and must not be compared directly.")
    print("\nOne way the second row is PESSIMISTIC for this platform: the surface")
    print("code is a 2D nearest-neighbour code, and an all-to-all machine does not")
    print("need one. High-rate qLDPC codes need far fewer physical qubits per")
    print("logical qubit, and Helios's own 2:1 encoding is evidence of exactly that.")
    print("So 4.5e4 is an upper bound on the qubit count for a trapped-ion")
    print("architecture, and the real gap is smaller than the one printed above.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp026_logical_qubit_budget.json"
    with open(path, "w") as fh:
        json.dump({
            "helios_physical_qubits": HELIOS_PHYSICAL_QUBITS,
            "helios_logical_qubits": HELIOS_LOGICAL_QUBITS,
            "physical_error": args.physical_error,
            "device_figures_are_published_not_measured": True,
            "translation": translation,
            "chain": args.chain.split(","),
            "chain_orbitals": sizes,
            "chain_two_qubit_gates": gates,
            "gate_exponent": slope,
            "gate_exponent_stderr": stderr,
            "target_orbitals": target,
            "predicted_two_qubit_gates": predicted,
            "needed_two_qubit_error": needed_error,
            "variational_shortfall": shortfall,
            "df_one_norm": one_norm,
            "df_factors": factors,
            "t_gates": t_gates,
            "required_distance": distance,
            "distillation_rounds": rounds,
            "physical_qubits": physical,
            "qubit_shortfall": (physical / HELIOS_PHYSICAL_QUBITS
                                if distance is not None else None),
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
