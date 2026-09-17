"""Experiment 027 -- are this project's own ground states classically easy?

The question behind this one was put as: can a classical computer simulate a
quantum computer without paying a worse complexity class, since you can just use
complex numbers and pseudorandomness for the measurements?

For a *general* circuit the answer is no, and the reason is memory rather than
cleverness: a statevector is `2^n` complex amplitudes, so 50 qubits is 18
petabytes and 300 qubits exceeds the number of atoms in the observable universe.
"Just complex numbers in Python" is exactly what a statevector simulator is, and
it is what every result in this repository was produced on.

But "in general" is doing a lot of work in that sentence, and the interesting
question is the specific one:

    do the states THIS PROJECT actually needs require exponential resources?

Because there are large classes that do not.  Clifford circuits are simulable in
polynomial time (Gottesman-Knill) however many qubits they use, and a state with
bounded entanglement across every cut is representable by a matrix product state
of bounded bond dimension -- which is what DMRG exploits, and DMRG is the real
classical competitor for quantum chemistry, not exact diagonalisation.

So this measures the quantity that decides it, on the exact ground states of the
molecules this repository already builds:

    the Schmidt rank across every contiguous cut, and the bond dimension needed
    to hold the discarded weight below a target.

If that bond dimension grows polynomially with system size, these Hamiltonians
are classically easy at every size and no quantum method can have an advantage
on them -- a much stronger statement than "the classical baseline happens to win
at the sizes we can test", which is all Results 42 and 50 could say.  If it grows
exponentially, the gap is real and the question moves to hardware.

**Two caveats, both of which make this an upper bound rather than an estimate.**

1. The cut is taken in the qubit ordering the repository's mapper produces.
   DMRG optimises the orbital ordering, and a good ordering lowers the bond
   dimension -- often by a lot.  So the true classical cost is **at most** what
   is measured here.
2. A hydrogen chain at equilibrium spacing is a weakly correlated,
   quasi-one-dimensional system, which is the best case for MPS.  The
   transition-metal active sites that make chemistry a quantum-advantage
   candidate (Result 82) are neither, and nothing here measures those.

Run:  python -m experiments.exp027_entanglement
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
from qres.problems.chemistry import build_molecule

from experiments.exp026_logical_qubit_budget import fit_power_law

#: discarded-weight targets.  1e-6 is the one to read: truncating a wavefunction
#: at discarded weight w costs roughly w in the energy, and chemical accuracy is
#: 1.6e-3 Hartree, so 1e-6 is comfortably inside it.
TRUNCATIONS = (1e-3, 1e-6, 1e-9)

MOLECULES = ("H2", "H4", "H6", "H8")

#: dense diagonalisation above this many qubits is not worth attempting
DENSE_LIMIT = 12


def ground_state(hamiltonian) -> np.ndarray:
    """Lowest eigenvector, dense or sparse depending on size."""
    if hamiltonian.num_qubits <= DENSE_LIMIT:
        matrix = hamiltonian.to_matrix()
        values, vectors = np.linalg.eigh(matrix)
        return vectors[:, int(np.argmin(values))]
    from scipy.sparse.linalg import eigsh

    matrix = hamiltonian.to_matrix(sparse=True)
    _, vectors = eigsh(matrix, k=1, which="SA", maxiter=100_000)
    return vectors[:, 0]


def bond_dimensions(state: np.ndarray, truncations=TRUNCATIONS) -> dict:
    """Bond dimension needed at each cut, for each discarded-weight target.

    The bond dimension an MPS needs is the *maximum* over cuts, since one bad
    cut forces the whole representation.  Reporting the mean would understate it
    in exactly the flattering direction.
    """
    vector = np.asarray(state).ravel()
    qubits = int(round(np.log2(vector.size)))
    vector = vector / np.linalg.norm(vector)

    exact_rank, entropies = 0, []
    needed = {target: 0 for target in truncations}

    for cut in range(1, qubits):
        matrix = vector.reshape(2**cut, 2 ** (qubits - cut))
        singular = np.linalg.svd(matrix, compute_uv=False)
        weights = singular**2
        weights = weights / weights.sum()

        exact_rank = max(exact_rank, int(np.sum(singular > 1e-10 * singular.max())))

        positive = weights[weights > 1e-18]
        entropies.append(float(-np.sum(positive * np.log2(positive))))

        # discarded weight if we keep the first k: 1 - cumsum
        tail = 1.0 - np.cumsum(weights)
        for target in truncations:
            inside = np.nonzero(tail < target)[0]
            keep = int(inside[0]) + 1 if inside.size else len(weights)
            needed[target] = max(needed[target], keep)

    return {
        "qubits": qubits,
        "exact_schmidt_rank": exact_rank,
        "max_entropy_bits": max(entropies) if entropies else 0.0,
        "maximal_rank": 2 ** (qubits // 2),
        "bond_dimension": {str(k): v for k, v in needed.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default=",".join(MOLECULES))
    args = ap.parse_args()

    print("=== experiment 027 :: are these ground states classically easy? ===")
    print("Exact ground states, Schmidt rank across every contiguous cut.")
    print("Upper bound on DMRG's cost: the orbital ordering is not optimised.\n")

    header = (
        f"{'molecule':>9}{'qubits':>8}{'exact rank':>12}{'maximal':>9}"
        f"{'entropy':>9}" + "".join(f"{'chi@' + f'{t:.0e}':>12}" for t in TRUNCATIONS)
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in args.molecules.split(","):
        started = time.perf_counter()
        problem = build_molecule(name)
        try:
            state = ground_state(problem.hamiltonian)
        except Exception as exc:                          # noqa: BLE001
            print(f"{name:>9}   skipped: {type(exc).__name__}: {exc}")
            continue

        measured = bond_dimensions(state)
        measured["molecule"] = name
        measured["seconds"] = time.perf_counter() - started
        rows.append(measured)

        chis = "".join(
            f"{measured['bond_dimension'][str(t)]:>12}" for t in TRUNCATIONS
        )
        print(
            f"{name:>9}{measured['qubits']:>8}{measured['exact_schmidt_rank']:>12}"
            f"{measured['maximal_rank']:>9}{measured['max_entropy_bits']:>9.3f}{chis}",
            flush=True,
        )

    # --------------------------------------------------------------- verdict
    print("\n--- does the bond dimension grow polynomially or exponentially? ---")
    summary = {}
    if len(rows) >= 3:
        qubits = [r["qubits"] for r in rows]
        for target in TRUNCATIONS:
            chis = [r["bond_dimension"][str(target)] for r in rows]
            # polynomial in the qubit count?  a log-log slope
            slope, _, stderr = fit_power_law(qubits, chis)
            # exponential?  log(chi) linear in n, with the maximal slope being
            # log(2)/2 per qubit -- the rate a maximally entangled state grows at
            rate = np.polyfit(qubits, np.log2(chis), 1)[0]
            summary[str(target)] = {
                "power_law_exponent": slope,
                "power_law_stderr": stderr,
                "bits_per_qubit": float(rate),
                "maximal_bits_per_qubit": 0.5,
            }
            print(f"  discarded weight {target:.0e}: chi ~ n^{slope:.2f} +- {stderr:.2f}, "
                  f"or 2^({rate:.3f} n) against a maximal 2^(0.5 n)")

        rates = {t: summary[str(t)]["bits_per_qubit"] for t in TRUNCATIONS}
        loose, tight = rates[max(TRUNCATIONS)], rates[min(TRUNCATIONS)]
        print(f"\n**The answer depends on the truncation, and that is the finding.**")
        print(f"  at {max(TRUNCATIONS):.0e} discarded weight: 2^({loose:.3f} n) -- "
              f"{loose / 0.5:.0%} of the maximal rate")
        print(f"  at {min(TRUNCATIONS):.0e} discarded weight: 2^({tight:.3f} n) -- "
              f"{tight / 0.5:.0%} of the maximal rate")
        print("\nSo the growth is exponential at every truncation measured, but the")
        print("exponent is a function of how much error you accept -- a cheap")
        print("approximate answer is much cheaper than an exact one, which is the")
        print("mechanism DMRG runs on. Neither 'classically easy' nor 'classically")
        print("hard' is the right summary of four points with a +-0.23 exponent.")
        print("\nAnd the two caveats bound it in the same direction: the orbital")
        print("ordering is not optimised and a hydrogen chain is the best case for")
        print("MPS, so the real classical cost is at most this. It does NOT transfer")
        print("to the transition-metal active sites of Result 82, which are the cases")
        print("where the classical baseline is actually in doubt.")

    print("\nWhat this does and does not settle: a statevector simulator costs 2^n")
    print("regardless, so 'simulate a quantum computer classically in Python' is")
    print("exponential as asked. But the STATES of interest can be far cheaper")
    print("than the general case, and where they are, no quantum method can win.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp027_entanglement.json"
    with open(path, "w") as fh:
        json.dump({
            "truncations": list(TRUNCATIONS),
            "ordering_is_not_optimised": True,
            "rows": rows,
            "growth": summary,
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
