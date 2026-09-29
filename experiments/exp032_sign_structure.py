"""Experiment 032 -- what Result 85's walker thresholds were actually measuring.

Result 85 built FCIQMC and read two things off it: that **correlation strength,
not system size, is the axis** of the sign problem ("stretched H2 costs 32x more
than equilibrium H2 at identical size"), and that the sign problem is **directly
visible as 33-38% annihilation** against 0.3-0.7% at equilibrium. It then named
H8 and H10 stretched the highest-value open measurement in the repository.

Three checks, and each one undercuts a reading rather than a number.

1. **H2 has no sign problem at all, provably.** Its sector is two determinants
   joined by one coupling. A signed graph with no cycle is always balanced, so a
   diagonal +-1 gauge makes it stoquastic, and FCIQMC spawns from a
   sign-coherent population can never carry the wrong sign: annihilation is
   **exactly** zero. Result 85's own data agree -- 0.0% at every rung. So the 32x
   that H2 pays when stretched is not the sign problem, and it is the only
   evidence for "correlation, not size" that is free of size.

2. **The thresholds trade one-for-one against run length.** Re-running the
   identical code, seeds and ladder at 4x the propagation steps cuts every
   threshold by 4-8x. A sign-problem wall does not move with run length; a
   statistical error bar does. The quantity whose exponent Result 85 fitted is
   ``walkers x steps`` to reach chemical accuracy -- sampling efficiency.

3. **The sign problem's severity, computed deterministically.** The gap between
   the ground state of ``H`` and of the same matrix with every off-diagonal
   element replaced by ``-|H_ij|`` is the rate at which the sign-incoherent
   component outgrows the physical one in a projector Monte Carlo. It is exactly
   zero if and only if the sector's sign graph is balanced, and it needs no walkers
   and no seeds. It grows with system size at **both** geometries, and the
   stretched/equilibrium ratio falls 21x -> 5x -> 1.9x from H4 to H8: by H8 size,
   not stretching, dominates.

Run:  python -m experiments.exp032_sign_structure [--skip-rerun]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from qres.bench import RESULTS_DIR
from qres.problems.chemistry import build_molecule

from experiments.exp030_stochastic_chemistry import (
    STEPS,
    prepare,
    sector,
    sector_ground_energy,
    smallest_sufficient,
)

MOLECULES = ("H2", "H4", "H6", "H8")
BOND_LENGTHS = (0.75, 2.0)

#: couplings below this are noise, whose signs are arbitrary and would fake
#: frustrated cycles. H8's sector has two bands of it: integral noise at
#: 1e-21..1e-16, and 1 920 entries at 1e-12..1e-11 -- the residue of Pauli
#: coefficients rounded to 12 decimals (``COEFFICIENT_DECIMALS``) failing to
#: cancel. The smallest coupling above both is 1.2e-9 (H6) / 1e-9 (H8).
SIGN_FLOOR = 1e-10

#: the run-length check: same code, same seeds, same ladder, this many times
#: the steps. A statistical threshold falls roughly in proportion; a sign wall
#: does not move.
STEP_FACTOR = 4
RERUN = ("H2@2.0", "H4@2.0", "H6@2.0", "H6@0.75")

#: Result 85's seed, so the 1x arm is its own published run
RESULT_85_SEED = 2026


def sector_matrix(problem, floor=SIGN_FLOOR):
    """The reference's sector, with couplings below ``floor`` dropped first.

    exp030's ``sector`` joins determinants through anything above 1e-12, which
    on H8 includes the 12-decimal rounding residue: its "sector" of 2 624 is the
    physical block of 2 468 plus eight blocks attached only through 1e-12
    couplings. Harmless for walkers (a spawn there has probability ~1e-14 per
    step) and for the ground energy (it lies in the physical block), but not for
    the stoquastic ground state, which would be minimised over blocks that are
    not physically connected.
    """
    diagonal, offdiag = prepare(problem.hamiltonian)
    offdiag = offdiag.multiply(abs(offdiag) > floor).tocsr()
    offdiag.eliminate_zeros()
    reference = int(problem.hf_bitstring, 2)
    members, _ = sector(offdiag, reference)
    block = (sparse.diags(diagonal) + offdiag).tocsr()[members][:, members]
    return block.tocsr(), members


def balanced(block, floor=SIGN_FLOOR):
    """Is the sector's signed graph balanced (sign-problem-free under some gauge)?

    Assign ``g_i = +-1`` along a breadth-first forest so that
    ``g_i H_ij g_j < 0`` on tree edges, then count the remaining edges that
    violate it. Zero violations means a gauge exists; any violation closes a
    cycle whose sign product no gauge can fix. Returns (balanced, edges,
    violated edges).

    A *forest*, one tree per component: dropping noise couplings can split the
    sector, and a first version grew a single tree from determinant 0, left the
    rest with gauge 0 -- which makes every edge they touch look satisfied -- and
    reported H8 balanced alongside a sign gap of 2.3 Hartree.
    """
    size = block.shape[0]
    offdiag = block - sparse.diags(block.diagonal())
    offdiag = offdiag.multiply(abs(offdiag) > floor).tocsr()
    offdiag.eliminate_zeros()
    gauge = np.zeros(size, dtype=int)
    for root in range(size):
        if gauge[root]:
            continue
        gauge[root] = 1
        queue = deque([root])
        while queue:
            i = queue.popleft()
            for k in range(offdiag.indptr[i], offdiag.indptr[i + 1]):
                j = offdiag.indices[k]
                if gauge[j] == 0:
                    gauge[j] = -gauge[i] * int(np.sign(offdiag.data[k]))
                    queue.append(j)
    coo = sparse.triu(offdiag, k=1).tocoo()
    violated = int(np.sum(gauge[coo.row] * coo.data * gauge[coo.col] > 0))
    return violated == 0, int(coo.nnz), violated


def sign_gap(block):
    """``E0(H) - E0(H~)`` with ``H~ = diag(H) - |offdiag(H)|``: zero iff balanced."""
    stoquastic = (sparse.diags(block.diagonal())
                  - abs(block - sparse.diags(block.diagonal()))).tocsr()
    if block.shape[0] <= 3000:
        fermionic = np.linalg.eigvalsh(block.toarray())[0]
        bosonic = np.linalg.eigvalsh(stoquastic.toarray())[0]
    else:
        fermionic = eigsh(block, k=1, which="SA")[0][0]
        bosonic = eigsh(stoquastic, k=1, which="SA")[0][0]
    return float(fermionic), float(bosonic), float(fermionic - bosonic)


def structure(molecules=MOLECULES, bond_lengths=BOND_LENGTHS):
    rows = []
    for name in molecules:
        for bond in bond_lengths:
            started = time.perf_counter()
            block, members = sector_matrix(build_molecule(name, bond_length=bond))
            is_balanced, edges, violated = balanced(block)
            fermionic, bosonic, gap = sign_gap(block)
            rows.append({
                "molecule": name, "bond_length": bond, "sector": len(members),
                "edges": edges, "violated_edges": violated, "balanced": is_balanced,
                "E0": fermionic, "E0_stoquastic": bosonic, "sign_gap": gap,
                "seconds": time.perf_counter() - started,
            })
            print(f"  {name}@{bond:<4} sector {len(members):>5}  edges {edges:>7}  "
                  f"balanced {str(is_balanced):>5}  gap {gap:10.6f} Ha", flush=True)
    return rows


def rerun_at_longer_runs(specs=RERUN, factor=STEP_FACTOR):
    rows = []
    for spec in specs:
        name, bond = spec.split("@")
        problem = build_molecule(name, bond_length=float(bond))
        diagonal, offdiag = prepare(problem.hamiltonian)
        reference = int(problem.hf_bitstring, 2)
        exact, _ = sector_ground_energy(problem.hamiltonian, offdiag, reference)
        print(f"  {spec}: {factor}x steps ({factor * STEPS})", flush=True)
        needed, trace = smallest_sufficient(diagonal, offdiag, reference, exact,
                                            RESULT_85_SEED, steps=factor * STEPS)
        rows.append({"spec": spec, "steps": factor * STEPS,
                     "walkers_needed": needed, "trace": trace})
    return rows


def result_85_rows():
    with open(RESULTS_DIR / "exp030_stochastic_chemistry.json") as fh:
        return json.load(fh)["rows"]


def annihilation_at(row, target):
    for rung in row["trace"]:
        if rung["target"] == target:
            return rung["annihilation"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rerun", action="store_true",
                    help="skip the 4x-steps FCIQMC re-run (about 10 minutes)")
    args = ap.parse_args()

    print("=== experiment 032 :: what Result 85's thresholds were measuring ===\n")

    print("--- 1. sign structure of each sector, exactly ---")
    rows = structure()

    published = {(r["molecule"], r["bond_length"]): r for r in result_85_rows()}
    h2 = [published[("H2", b)] for b in BOND_LENGTHS if ("H2", b) in published]
    h2_annihilation = sorted({rung["annihilation"] for r in h2 for rung in r["trace"]})
    print(f"\n  H2 annihilation at every rung of Result 85's own runs: {h2_annihilation}")

    print("\n--- 2. the same walker count, both geometries (Result 85's data) ---")
    equal = []
    for name, target in (("H4", 40), ("H6", 160)):
        eq = annihilation_at(published[(name, 0.75)], target)
        st = annihilation_at(published[(name, 2.0)], target)
        at_threshold = (annihilation_at(published[(name, 0.75)],
                                        published[(name, 0.75)]["walkers_needed"]),
                        annihilation_at(published[(name, 2.0)],
                                        published[(name, 2.0)]["walkers_needed"]))
        equal.append({"molecule": name, "target": target, "equilibrium": eq,
                      "stretched": st, "at_thresholds": at_threshold})
        print(f"  {name} at {target} walkers: equilibrium {eq:.2%}, stretched {st:.2%}"
              f"  ({st / eq:.1f}x)   -- at each arm's own threshold: "
              f"{at_threshold[0]:.2%} vs {at_threshold[1]:.2%}")

    reruns = []
    if not args.skip_rerun:
        print(f"\n--- 3. identical code and seeds, {STEP_FACTOR}x the run length ---")
        reruns = rerun_at_longer_runs()
        for rerun in reruns:
            name, bond = rerun["spec"].split("@")
            before = published[(name, float(bond))]["walkers_needed"]
            after = rerun["walkers_needed"]
            print(f"  {rerun['spec']}: threshold {before} -> {after} walkers "
                  f"({before / after:.0f}x fewer)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp032_sign_structure.json"
    with open(path, "w") as fh:
        json.dump({"structure": rows, "h2_annihilation": h2_annihilation,
                   "equal_walkers": equal, "step_factor": STEP_FACTOR,
                   "reruns": reruns}, fh, indent=2, default=float)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
