"""Experiment 030 -- the probabilistic classical baseline, and where it breaks.

Every classical baseline in this repository so far is **deterministic**: FCI
(exact, and exponential), CCSD(T) (polynomial, and single-reference). The one
that is missing is the *probabilistic* one, and it is the one that matters most
for the advantage question, for a reason that is precise rather than rhetorical.

**Full Configuration Interaction Quantum Monte Carlo** samples the same
imaginary-time propagation a quantum computer's phase estimation would run,
using signed random walkers on determinants instead of amplitudes on qubits. Its
cost is **polynomial** -- and that stays true no matter how large the
determinant space gets -- *unless* the **fermionic sign problem** bites, in which
case the walker population needed to resolve the ground state grows
**exponentially**.

That makes the sign problem the actual boundary of the quantum advantage
question for chemistry, sharper than anything measured here so far:

* where FCIQMC is efficient, a classical computer solves the problem in
  polynomial time and **no quantum method can have an advantage**, whatever the
  hardware does
* where the sign problem bites, the classical method fails for a reason that is
  structural rather than a matter of implementation, and that is precisely the
  window a quantum computer could occupy

So this measures one number, as a function of system size and of correlation
strength:

    how many walkers are needed to reach chemical accuracy?

If that grows polynomially, the window is closed for these systems. If it grows
exponentially, it is open, and the growth rate says how wide.

**The correlation-strength axis is why this is worth running on tiny systems.**
Result 83 measured hydrogen chains at equilibrium spacing to be substantially
compressible -- quasi-one-dimensional, weakly correlated, the *easy* case for
every classical method. Stretching the bonds breaks the molecule towards
independent radicals, which is a textbook strong-correlation regime and where the
sign problem is expected to appear. Both are measured here, because reporting
only the equilibrium geometry would test the classical method exactly where it is
strongest.

**What this is not.** This is a from-scratch FCIQMC without the refinements that
make production codes efficient -- no initiator approximation, no semi-stochastic
projection, no excitation generators smarter than uniform. So an absolute walker
count here is an **upper bound** on what a real implementation needs, and a
finding of "the sign problem bites" is only meaningful as a *scaling*, never as
an absolute cost. The implementation is validated against exact diagonalisation
on every system before any scaling is read off it.

Run:  python -m experiments.exp030_stochastic_chemistry
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import sparse

from qres.bench import RESULTS_DIR
from qres.factorization import molecular_integrals
from qres.problems.chemistry import build_molecule, exact_ground_energy

from experiments.exp021_tensor_hypercontraction import CHEMICAL_ACCURACY
from experiments.exp026_logical_qubit_budget import fit_power_law

#: imaginary-time step. Small enough that death/clone probabilities stay below
#: one for these Hamiltonians, which is what makes the integer walker update a
#: faithful sampling rather than a truncation.
TIME_STEP = 5e-3

#: Walker ladder. The smallest entry reaching chemical accuracy is the result,
#: so the ladder has to start BELOW every threshold it means to measure -- a
#: first version started at 100 and H4 passed on the first rung, which clips the
#: threshold at the ladder's own floor and would make the scaling an artefact of
#: the grid rather than a measurement. Factor-two steps from 10.
WALKER_LADDER = (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120,
                 10240, 20480, 40960, 81920)

#: propagation steps, and how many to discard before accumulating the estimator
STEPS = 12000
EQUILIBRATION = 4000

#: independent seeds per walker target; the median decides the threshold
SEEDS = 5

#: shift update: every `SHIFT_PERIOD` steps once the population target is met
SHIFT_PERIOD = 10
SHIFT_DAMPING = 0.05


def sector(offdiag, reference):
    """Indices reachable from ``reference`` through the off-diagonal couplings.

    **This is load-bearing and its absence produced a silent wrong answer.** The
    qubit Hamiltonian is block diagonal by particle number, so the computational
    basis splits into sectors that never talk to each other. A first version
    picked the reference by ``argmin(diagonal)`` and landed on a determinant in a
    *different* sector with no couplings at all -- ``|H_ref,j|`` was 7e-18 -- so
    every walker sat on it forever and the projected energy came back as exactly
    the Hartree-Fock value at every walker count. An error of 5.8e-2 with a
    perfectly stable population looks like a converged measurement.

    The exact energy has to be taken inside this sector too. ``exact_ground_energy``
    returns the lowest eigenvalue of the *whole* matrix, which can live in a
    sector with the wrong electron count and is then not the number FCIQMC is
    converging to.
    """
    from scipy.sparse.csgraph import connected_components

    adjacency = abs(offdiag) > 1e-12
    count, labels = connected_components(adjacency, directed=False)
    return np.nonzero(labels == labels[reference])[0], count


def sector_ground_energy(hamiltonian, offdiag, reference):
    """Lowest eigenvalue within the sector containing the reference."""
    members, _ = sector(offdiag, reference)
    matrix = hamiltonian.to_matrix(sparse=True).real.tocsr()
    block = matrix[members][:, members].toarray()
    return float(np.linalg.eigvalsh(block)[0]), members


def prepare(hamiltonian):
    """Diagonal and off-diagonal parts of H in the computational basis.

    The chemistry Hamiltonians here are real symmetric in this basis; the
    imaginary part is checked rather than assumed, because a silently-discarded
    imaginary part would change the walker dynamics.
    """
    matrix = hamiltonian.to_matrix(sparse=True).tocsr()
    imaginary = np.abs(matrix.imag).max() if matrix.nnz else 0.0
    if imaginary > 1e-12:
        raise ValueError(f"Hamiltonian is not real in this basis: {imaginary:.2e}")
    matrix = matrix.real.tocsr()
    diagonal = matrix.diagonal().copy()
    offdiag = (matrix - sparse.diags(diagonal)).tocsr()
    offdiag.eliminate_zeros()
    return diagonal, offdiag


def fciqmc(diagonal, offdiag, reference, target, rng, steps=STEPS,
           equilibration=EQUILIBRATION, dt=TIME_STEP):
    """Signed-walker imaginary-time propagation with annihilation.

    Vectorised over walkers rather than looping in Python: a uniform excitation
    is drawn for every walker at once by indexing into the CSR row structure, so
    a step costs one pass over the population regardless of its size.
    """
    dim = len(diagonal)
    counts = np.zeros(dim, dtype=np.int64)
    # Start AT the target population rather than growing into it. With a shift
    # at the Hartree-Fock energy the growth rate is (E_HF - E_0) -- the
    # correlation energy, ~0.02 Hartree on H2 -- so reaching 1e4 walkers from 1e2
    # would take an imaginary time of ~500, which at this step size is half a
    # million steps. A first version did try to grow and never left 123 walkers
    # on a target of 10 000, which silently made every row of the ladder the
    # same measurement.
    counts[reference] = target

    indptr, indices, data = offdiag.indptr, offdiag.indices, offdiag.data
    row_length = np.diff(indptr)
    reference_row = offdiag[reference].toarray().ravel()

    shift = diagonal[reference]
    previous_total = float(np.abs(counts).sum())

    numerator = denominator = 0.0
    gross_spawns = net_spawns = 0.0
    populations, shifts = [], []

    for step in range(steps):
        occupied = np.nonzero(counts)[0]
        if occupied.size == 0:
            break
        magnitudes = np.abs(counts[occupied])
        walkers = np.repeat(occupied, magnitudes)
        walker_signs = np.sign(counts[walkers])

        change = np.zeros(dim, dtype=np.int64)

        # ---- spawning: one uniform excitation per walker
        lengths = row_length[walkers]
        live = lengths > 0
        if np.any(live):
            parents = walkers[live]
            parent_signs = walker_signs[live]
            span = lengths[live]
            pick = indptr[parents] + (rng.random(span.size) * span).astype(np.int64)
            targets = indices[pick]
            values = data[pick]

            # p_gen = 1/span, so the unbiased spawn rate is dt*|H_ij|*span
            rate = dt * np.abs(values) * span
            whole = np.floor(rate)
            number = whole.astype(np.int64) + (rng.random(rate.size) < (rate - whole))
            # the propagator is (1 - dt*H), so a positive H_ij flips the sign
            signs = -parent_signs * np.sign(values)
            contribution = number * signs
            # Accumulated separately from the death step so the annihilation
            # measurement below compares spawns against spawns. A first version
            # compared gross spawns against the *combined* spawn-and-death array
            # and reported negative annihilation, which is not a quantity.
            spawned = np.zeros(dim, dtype=np.int64)
            np.add.at(spawned, targets, contribution)
            change += spawned
            gross_spawns += float(np.abs(contribution).sum())
            net_spawns += float(np.abs(spawned).sum())

        # ---- death and cloning on the diagonal
        rate = dt * (diagonal[walkers] - shift)
        magnitude = np.abs(rate)
        whole = np.floor(magnitude)
        events = whole.astype(np.int64) + (rng.random(magnitude.size) < (magnitude - whole))
        # a positive rate removes the walker, a negative one duplicates it
        direction = np.where(rate > 0, -1, 1)
        np.add.at(change, walkers, events * direction * walker_signs)

        counts = counts + change

        total = float(np.abs(counts).sum())
        if total == 0.0:
            break

        # ---- population control, active from the first step
        if step % SHIFT_PERIOD == 0:
            shift -= (SHIFT_DAMPING / (SHIFT_PERIOD * dt)) * np.log(
                total / max(previous_total, 1.0)
            )
            previous_total = total

        # ---- projected energy estimator, accumulated after equilibration
        if step >= equilibration and counts[reference] != 0:
            numerator += float(
                diagonal[reference] * counts[reference] + reference_row @ counts
            )
            denominator += float(counts[reference])
            populations.append(total)
            shifts.append(shift)

    energy = numerator / denominator if denominator else float("nan")
    return {
        "energy": energy,
        "walkers": float(np.mean(populations)) if populations else 0.0,
        "shift": float(np.mean(shifts)) if shifts else float("nan"),
        "target": target,
        # how much of the spawned population cancelled against opposite signs --
        # the direct signature of the sign problem
        "gross_spawns": gross_spawns,
        # None, not 0.0: no spawns at all and no cancellation are different
        # failures and reporting both as "0%" hid the sector bug for a full run
        "annihilation": (float(1.0 - net_spawns / gross_spawns)
                         if gross_spawns else None),
        "samples": len(populations),
    }


def smallest_sufficient(diagonal, offdiag, reference, exact, seed,
                        ladder=WALKER_LADDER, steps=STEPS, seeds=SEEDS):
    """Climb the walker ladder until the MEDIAN error reaches chemical accuracy.

    Several independent seeds per rung, and the median decides. A single run is
    not a threshold: the errors here are non-monotonic in the walker count
    (1.6e-3 at 80 walkers against 1.6e-1 at 40 on stretched H4), so the first
    rung a single realisation happens to pass is partly luck. This repository's
    rule 3 exists for exactly this and was written after a 4-seed result showing
    a 3.8x effect became 1.3x and p = 0.54 at 24 seeds.
    """
    trace = []
    for target in ladder:
        started = time.perf_counter()
        runs = []
        for index in range(seeds):
            rng = np.random.default_rng(seed + 1009 * index)
            run = fciqmc(diagonal, offdiag, reference, target, rng, steps=steps)
            run["error"] = abs(run["energy"] - exact)
            runs.append(run)

        errors = [run["error"] for run in runs]
        annihilations = [r["annihilation"] for r in runs if r["annihilation"] is not None]
        row = {
            "target": target,
            "median_error": float(np.median(errors)),
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "errors": [float(e) for e in errors],
            "median_energy": float(np.median([r["energy"] for r in runs])),
            "walkers": float(np.median([r["walkers"] for r in runs])),
            "annihilation": float(np.median(annihilations)) if annihilations else None,
            "seeds": seeds,
            "seconds": time.perf_counter() - started,
        }
        row["inside"] = bool(row["median_error"] < CHEMICAL_ACCURACY)
        trace.append(row)

        annihilation = (f"annih {row['annihilation']:>5.1%}"
                        if row["annihilation"] is not None
                        else "NO SPAWNS -- walkers cannot move")
        print(f"      target {target:>7}  walkers {row['walkers']:>9.0f}  "
              f"E {row['median_energy']:>12.6f}  med err {row['median_error']:>9.2e}  "
              f"[{row['min_error']:.1e},{row['max_error']:.1e}]  "
              f"{annihilation}  ({row['seconds']:.0f}s)", flush=True)
        if row["inside"]:
            return target, trace
    return None, trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default="H2,H4,H6")
    ap.add_argument("--bond-lengths", default="0.75,2.0",
                    help="equilibrium and stretched, in angstrom")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    print("=== experiment 030 :: the probabilistic classical baseline ===")
    print("FCIQMC: signed walkers on determinants, the classical method that")
    print("samples the same propagation phase estimation would run.")
    print("The question is whether the walker count grows polynomially.\n")

    rows = []
    for bond in [float(b) for b in args.bond_lengths.split(",")]:
        label = "equilibrium" if bond < 1.0 else f"stretched r={bond}"
        print(f"--- {label} (r = {bond} A) ---")
        for name in args.molecules.split(","):
            problem = build_molecule(name, bond_length=bond)
            diagonal, offdiag = prepare(problem.hamiltonian)
            # The Hartree-Fock determinant, not the lowest diagonal element:
            # argmin(diagonal) can land in a different particle-number sector.
            reference = int(problem.hf_bitstring, 2)
            exact, members = sector_ground_energy(
                problem.hamiltonian, offdiag, reference)
            global_exact = exact_ground_energy(problem.hamiltonian)
            coupling = float(np.abs(offdiag[reference].toarray()).max())
            if coupling < 1e-10:
                raise ValueError(
                    f"{name}: reference determinant has no couplings "
                    f"({coupling:.1e}); FCIQMC cannot move")
            # Spatial orbitals from the integrals, not from the qubit count: the
            # parity mapper's two-qubit reduction makes num_qubits//2 wrong (it
            # called H2 a one-orbital molecule).
            orbitals = molecular_integrals(problem)[0].shape[0]

            print(f"   {name}  {orbitals} orbitals, dim {len(diagonal)}, "
                  f"sector {len(members)}  exact {exact:.6f}"
                  f"  (global {global_exact:.6f})")
            needed, trace = smallest_sufficient(
                diagonal, offdiag, reference, exact, 2026,
                steps=args.steps, seeds=args.seeds
            )
            if needed is None:
                print(f"   {name}  never reached chemical accuracy on this ladder")
            rows.append({
                "molecule": name, "bond_length": bond, "regime": label,
                "dimension": len(diagonal), "orbitals": orbitals,
                "exact_energy": exact, "global_exact_energy": global_exact,
                "sector_size": int(len(members)),
                "reference_determinant": reference,
                "reference_max_coupling": coupling,
                "walkers_needed": needed,
                "trace": [{k: v for k, v in run.items()} for run in trace],
            })

    # ------------------------------------------------------------- the scaling
    print("\n--- how does the walker requirement scale? ---")
    summary = {}
    for label in sorted({row["regime"] for row in rows}):
        group = [r for r in rows if r["regime"] == label and r["walkers_needed"]]
        print(f"\n  {label}:")
        for row in group:
            print(f"    {row['molecule']:>4}  {row['orbitals']:>2} orbitals  "
                  f"dim {row['dimension']:>6}  walkers {row['walkers_needed']:>7}")
        if len(group) >= 3:
            slope, _, stderr = fit_power_law(
                [r["orbitals"] for r in group], [r["walkers_needed"] for r in group]
            )
            rate = np.polyfit([r["orbitals"] for r in group],
                              np.log2([r["walkers_needed"] for r in group]), 1)[0]
            summary[label] = {"exponent": slope, "stderr": stderr,
                              "bits_per_orbital": float(rate)}
            print(f"    walkers ~ N^{slope:.2f} +- {stderr:.2f}, "
                  f"or 2^({rate:.2f} N)")
        else:
            print("    fewer than three points that reached accuracy -- no fit")

    print("\n--- what this says about the advantage question ---")
    print("Where the walker count grows POLYNOMIALLY, a classical computer solves")
    print("the problem in polynomial time and no quantum method can have an")
    print("advantage there, whatever the hardware does. Where it grows")
    print("EXPONENTIALLY, the sign problem is biting and that is the window.")
    print("\nThe absolute counts are an upper bound: no initiator approximation,")
    print("no semi-stochastic projection, uniform excitation generation. Only the")
    print("SCALING is a claim, and only against these systems.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp030_stochastic_chemistry.json"
    with open(path, "w") as fh:
        json.dump({
            "time_step": TIME_STEP, "steps": args.steps,
            "equilibration": EQUILIBRATION, "ladder": list(WALKER_LADDER),
            "seeds": args.seeds,
            "chemical_accuracy": CHEMICAL_ACCURACY,
            "rows": rows, "scaling": summary,
        }, fh, indent=2, default=float)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
