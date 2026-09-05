"""Experiment 020 -- the block encoding, where Result 75 said the leverage is.

Result 75 measured that this repository's naive LCU block encoding lands at
1.1e15 T gates for a drug-sized molecule -- roughly where the published field
stood in 2017, and ~5e4 behind tensor hypercontraction.  It named the block
encoding as the one place a large factor is still available, and this experiment
takes the first step: **double factorisation**, measured rather than assumed.

The interesting part is that the answer inverts a verdict this repository already
recorded.  `qres/factorization.py` was written to shrink the *measurement*
1-norm, and Result 38 ranked it **fourth of five** schemes -- a negative result.
Measuring the 1-norm directly says why:

    the double-factorised 1-norm is LARGER than the Pauli one, by 3.6-10x

So as a measurement scheme it is worse, exactly as Result 38 found.  But a block
encoding is charged `lambda x (cost per walk)`, and double factorisation trades a
*bigger* lambda for a *much* smaller per-walk cost:

* naive LCU walks over `L ~ N^4` Pauli terms, so the walk costs O(L)
* double factorisation walks over `~N` factors, each applied as a Givens
  rotation network of O(N^2) rotations, so the walk costs O(N^2)

Whether the trade pays is an empirical question about the constants, which is
what this measures.

**Provenance.** The 1-norms and factor counts are **measured** from the real
molecular integrals via `qres.factorization`.  The per-walk gate models are
**standard theory** taken at a deliberately simple level -- no QROM, no advanced
state preparation -- so both sides are charged naively and the *comparison*
survives even though neither absolute number should be quoted against the
literature.  Result 72 is the standing lesson: none of this is novel, and the
published constructions are better than both arms here.

Run:  python -m experiments.exp020_block_encoding
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
from qres.problems.chemistry import build_molecule

CHEMICAL_ACCURACY = 1.6e-3
T_PER_TOFFOLI = 4
QPE_CONSTANT = np.pi / 2
QPE_REPETITIONS = 3

#: Ross-Selinger: one arbitrary rotation to precision eps_s costs ~3 log2(1/eps_s)
#: T gates.  The synthesis precision is set well below the target so it is not
#: the limiting error.
SYNTHESIS_T_PER_ROTATION = 3 * np.log2(1 / 1e-10)

#: naive LCU: SELECT plus PREPARE over L terms, ~2 Toffolis per term
TOFFOLIS_PER_TERM = 2.0

MOLECULES = ("H2", "H4", "LiH", "H6", "H2O", "NH3", "H8", "H10")


def naive_lcu_cost(one_norm: float, terms: int, epsilon: float) -> dict:
    """T gates for qubitized QPE with a Pauli-LCU block encoding."""
    walks = QPE_CONSTANT * one_norm / epsilon
    t_per_walk = TOFFOLIS_PER_TERM * terms * T_PER_TOFFOLI
    return {
        "one_norm": one_norm,
        "walks": walks,
        "t_per_walk": t_per_walk,
        "t_gates": walks * t_per_walk * QPE_REPETITIONS,
    }


def double_factorised_cost(
    one_norm: float, orbitals: int, factors: int, epsilon: float
) -> dict:
    """T gates for qubitized QPE with a double-factorised block encoding.

    Per walk: select a factor (PREPARE over ``factors``), rotate into its
    eigenbasis with a Givens network of ``orbitals*(orbitals-1)/2`` rotations,
    apply the diagonal, rotate back.  The rotation network dominates and is
    O(N^2) rather than the LCU's O(N^4).
    """
    walks = QPE_CONSTANT * one_norm / epsilon
    givens = orbitals * (orbitals - 1) / 2
    rotation_t = 2 * givens * SYNTHESIS_T_PER_ROTATION  # in and back out
    prepare_t = TOFFOLIS_PER_TERM * factors * T_PER_TOFFOLI
    diagonal_t = orbitals * SYNTHESIS_T_PER_ROTATION
    t_per_walk = rotation_t + prepare_t + diagonal_t
    return {
        "one_norm": one_norm,
        "walks": walks,
        "t_per_walk": t_per_walk,
        "t_gates": walks * t_per_walk * QPE_REPETITIONS,
        "givens_rotations": givens,
    }


def measure(name: str, epsilon: float) -> dict:
    started = time.perf_counter()
    problem = build_molecule(name)
    one_body, two_body, _ = molecular_integrals(problem)
    factorised = double_factorize(one_body, two_body)

    orbitals = one_body.shape[0]
    pauli_norm = pauli_one_norm(problem.hamiltonian)
    df_norm = factorised.one_norm()

    naive = naive_lcu_cost(pauli_norm, len(problem.hamiltonian), epsilon)
    factored = double_factorised_cost(
        df_norm, orbitals, factorised.num_factors, epsilon
    )
    return {
        "molecule": name,
        "orbitals": orbitals,
        "qubits": problem.hamiltonian.num_qubits,
        "pauli_terms": len(problem.hamiltonian),
        "pauli_one_norm": pauli_norm,
        "df_one_norm": df_norm,
        "one_norm_ratio": df_norm / pauli_norm,
        "df_factors": factorised.num_factors,
        "naive": naive,
        "double_factorised": factored,
        "t_gate_speedup": naive["t_gates"] / factored["t_gates"],
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon", type=float, default=CHEMICAL_ACCURACY)
    ap.add_argument("--molecules", nargs="+", default=list(MOLECULES))
    args = ap.parse_args()

    print("=== experiment 020 :: does double factorisation pay as a block encoding? ===")
    print("1-norms and factor counts MEASURED from the molecular integrals;")
    print("per-walk gate models are standard theory, charged naively on both sides.")
    print(f"target accuracy {args.epsilon:.1e} Ha\n")

    header = (
        f"{'molecule':<7}{'N':>3}{'L':>7}{'lam_P':>8}{'lam_DF':>8}{'ratio':>7}"
        f"{'T/walk LCU':>12}{'T/walk DF':>11}{'total LCU':>11}{'total DF':>11}{'gain':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in args.molecules:
        try:
            row = measure(name, args.epsilon)
        except Exception as exc:  # noqa: BLE001 -- report, never skip silently
            print(f"{name:<7}  FAILED  {type(exc).__name__}: {exc}", flush=True)
            continue
        print(
            f"{row['molecule']:<7}{row['orbitals']:>3}{row['pauli_terms']:>7,d}"
            f"{row['pauli_one_norm']:>8.1f}{row['df_one_norm']:>8.1f}"
            f"{row['one_norm_ratio']:>7.2f}"
            f"{row['naive']['t_per_walk']:>12,.0f}{row['double_factorised']['t_per_walk']:>11,.0f}"
            f"{row['naive']['t_gates']:>11.2e}{row['double_factorised']['t_gates']:>11.2e}"
            f"{row['t_gate_speedup']:>8.2f}",
            flush=True,
        )
        rows.append(row)

    print("\n--- the two effects, which point in opposite directions ---")
    norms = np.array([r["one_norm_ratio"] for r in rows])
    walk = np.array([r["naive"]["t_per_walk"] / r["double_factorised"]["t_per_walk"]
                     for r in rows])
    gains = np.array([r["t_gate_speedup"] for r in rows])
    print(f"1-norm ratio (DF/Pauli):      {norms.min():.2f} - {norms.max():.2f}  "
          f"-- DF is WORSE, so it needs more walks")
    print(f"per-walk cost ratio (LCU/DF): {walk.min():.2f} - {walk.max():.2f}  "
          f"-- DF is better, but by LESS")

    # Computed from the data, not narrated: an earlier version hardcoded
    # "loses on all seven molecules" and kept saying it after H10 was added and
    # won, which is the same class of error as a stale comment.
    winners = [r for r in rows if r["t_gate_speedup"] > 1.0]
    losers = [r for r in rows if r["t_gate_speedup"] <= 1.0]
    print(f"\n**Net across {len(rows)} molecules: {gains.min():.2f} - {gains.max():.2f}.**")
    print(f"Double factorisation loses on {len(losers)} and wins on {len(winners)}"
          f"{' (' + ', '.join(r['molecule'] for r in winners) + ')' if winners else ''}.")
    if winners:
        smallest_win = min(winners, key=lambda r: r["orbitals"])
        print(f"The smallest molecule where it wins is {smallest_win['molecule']} "
              f"at N = {smallest_win['orbitals']}, gain "
              f"{smallest_win['t_gate_speedup']:.2f} -- **measured, not extrapolated.**")
    else:
        print("It wins nowhere measured, so anything positive below is extrapolation.")
    print("\nThe 1-norm half of this is consistent with Result 38, which ranked the")
    print("same factorisation fourth of five as a MEASUREMENT scheme -- measurement")
    print("is charged the 1-norm alone, so it saw only the penalty.")

    # ---------------------------------------------------------- extrapolation
    #
    # **All four quantities are fitted from the SAME series.**  A first version
    # took the Pauli side from Result 53's `N^2.86` and the DF side from these
    # measurements, which mixes two directions Result 53 had explicitly separated:
    # `N^2.86` is the *basis-limit* direction (one molecule, growing basis) while
    # this is the *atom-count* direction, where Result 53 measured `N^-0.39`.
    # The mismatch showed up as a predicted lambda of 61.5 at N=8 against a
    # measured 40.0 -- 1.5x too high, in the direction that flattered DF.
    print("\n--- extrapolation, all exponents fitted on the same H-chain series ---")
    chain = [r for r in rows if r["molecule"] in ("H2", "H4", "H6", "H8", "H10")]
    if len(chain) < 3:
        print("need at least three chain molecules to fit; skipping")
        chain = rows

    logs = np.log(np.array([r["orbitals"] for r in chain], dtype=float))
    fits = {
        key: np.polyfit(logs, np.log(np.array([r[key] for r in chain], dtype=float)), 1)
        for key in ("pauli_one_norm", "pauli_terms", "df_one_norm", "df_factors")
    }
    for key, fit in fits.items():
        print(f"  {key:<16} ~ N^{fit[0]:.2f}")

    def predict(key: str, n: float) -> float:
        return float(np.exp(np.polyval(fits[key], np.log(n))))

    def costs_at(n: float) -> tuple[float, float]:
        naive = naive_lcu_cost(predict("pauli_one_norm", n),
                               predict("pauli_terms", n), CHEMICAL_ACCURACY)
        factored = double_factorised_cost(predict("df_one_norm", n), n,
                                          predict("df_factors", n), CHEMICAL_ACCURACY)
        return naive["t_gates"], factored["t_gates"]

    print("\nThe DF 1-norm penalty SHRINKS with size -- that is the real finding:")
    print(f"  Pauli 1-norm ~ N^{fits['pauli_one_norm'][0]:.2f} against "
          f"DF ~ N^{fits['df_one_norm'][0]:.2f}")

    # scan from the smallest size, not from a hand-picked start
    crossover = next(
        (n for n in range(2, 400) if costs_at(n)[0] > costs_at(n)[1]), None
    )
    largest = int(max(r["orbitals"] for r in rows))
    print(f"\n**Crossover: N = {crossover}.** Largest molecule measured: N = {largest}.")
    if crossover is not None and crossover <= largest:
        print("That is INSIDE the measured range, so it is a measurement, not a")
        print("prediction -- check it against the table above.")
    else:
        print("That is outside the measured range, so it remains a prediction.")

    target = 50
    lam_drug = predict("df_one_norm", target)
    factors_drug = predict("df_factors", target)
    naive_total, df_total = costs_at(target)
    naive_drug = {"t_gates": naive_total}
    df_drug = {"t_gates": df_total}
    print(f"  naive LCU: {naive_drug['t_gates']:.2e} T gates")
    print(f"  double factorised: {df_drug['t_gates']:.2e} T gates  "
          f"(**{naive_drug['t_gates'] / df_drug['t_gates']:.0f}x**)")

    state_of_the_art = 5.3e9 * T_PER_TOFFOLI
    print(f"\n  published (Lee et al. 2021, THC): {state_of_the_art:.1e} T gates")
    print(f"  still behind by: {df_drug['t_gates'] / state_of_the_art:.0e}x")
    print("\nSo double factorisation recovers a large part of the gap Result 75")
    print("identified, and does not close it. The remainder is tensor")
    print("hypercontraction plus the QROM-based state preparation neither arm")
    print("here implements -- both published, neither invented here.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp020_block_encoding.json"
    with open(path, "w") as fh:
        json.dump({
            "epsilon": args.epsilon,
            "synthesis_t_per_rotation": SYNTHESIS_T_PER_ROTATION,
            "rows": rows,
            "chain_exponents": {k: float(v[0]) for k, v in fits.items()},
            "crossover_orbitals": crossover,
            "drug": {
                "orbitals": target, "df_one_norm": lam_drug,
                "df_factors": factors_drug,
                "naive_t_gates": naive_drug["t_gates"],
                "df_t_gates": df_drug["t_gates"],
                "speedup": naive_drug["t_gates"] / df_drug["t_gates"],
                "behind_state_of_the_art": df_drug["t_gates"] / state_of_the_art,
            },
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
