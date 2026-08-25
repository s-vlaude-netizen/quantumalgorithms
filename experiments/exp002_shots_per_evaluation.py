"""Experiment 002 -- how finely should a fixed shot budget be sliced?

At a fixed total budget ``B``, spending ``s`` shots per energy evaluation buys
``B / (2s)`` optimiser iterations.  Both extremes are bad and the interesting
question is where the optimum sits and what moves it.

There is a reason to expect an interior optimum rather than a monotone trend.
For SPSA the per-step parameter noise scales as ``a_k * sigma_E / (c sqrt(s))``
with the gain ``a_k ~ a / k^alpha``, ``alpha ~ 0.6``.  Accumulated over ``K``
steps at fixed budget ``B = 2 K s``:

    noise variance  ~  K a_k^2 sigma^2 / (c^2 s)
                    ~  K^2 a_k^2 / B       ~  K^(2 - 2 alpha) / B  =  K^0.8 / B

so *more* iterations means *more* accumulated parameter noise, while too few
means the optimiser has not converged.  The two effects cross somewhere in the
middle.

Run:  python -m experiments.exp002_shots_per_evaluation --molecule H2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qres.bench import Study, run_over_seeds
from qres.problems.chemistry import CHEMICAL_ACCURACY_HA, build_molecule
from qres.vqe import run_vqe

# Controlled-rotation entanglers, so theta=0 IS the Hartree-Fock reference.
# With fixed CX entanglers the zero-parameter state is orthogonal to it and
# every run below starts 4 Ha from the answer -- see RESEARCH_LOG Result 5.
ANSATZ = "hea:2:linear:cry"

MOLECULE_KWARGS = {"H2": {}, "H4": {}, "LiH": {}, "BeH2": dict(active_electrons=4, active_orbitals=6)}


def _run(seed: int, molecule: str, molecule_kwargs: dict, **kwargs):
    problem = build_molecule(molecule, **molecule_kwargs)
    return run_vqe(problem, seed=seed, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="H2")
    ap.add_argument("--env", default="ideal")
    ap.add_argument("--budget", type=int, default=120_000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--optimizers", default="cobyla,spsa")
    ap.add_argument(
        "--shots",
        default="256,512,1024,2048,4096,8192,16384",
        help="comma-separated shots-per-evaluation values to sweep",
    )
    args = ap.parse_args()

    shots_values = [int(s) for s in args.shots.split(",")]
    optimizers = args.optimizers.split(",")
    seeds = list(range(args.seeds))

    problem = build_molecule(args.molecule, **MOLECULE_KWARGS[args.molecule])
    name = f"exp002_{args.molecule}_{args.env.replace('@', '')}_{args.budget}"
    study = Study(name=name, seeds=seeds)

    print(f"=== experiment 002 :: {args.molecule} ({problem.num_qubits}q, "
          f"{len(problem.hamiltonian)} terms) env={args.env} ===")
    print(f"budget={args.budget:,}  seeds={len(seeds)}  "
          f"chemical accuracy={CHEMICAL_ACCURACY_HA:.2e} Ha\n")

    header = f"{'shots/eval':>11}{'~iters':>8}" + "".join(f"{o:>13}" for o in optimizers)
    print(header)
    print("-" * len(header))

    for shots in shots_values:
        cells = []
        for optimizer in optimizers:
            label = f"{args.molecule}|{optimizer}|s{shots}"
            t0 = time.perf_counter()
            runs = run_over_seeds(
                _run,
                seeds,
                molecule=args.molecule,
                molecule_kwargs=MOLECULE_KWARGS[args.molecule],
                ansatz=ANSATZ,
                environment=args.env,
                grouping="commuting",
                allocation="adaptive",
                optimizer=optimizer,
                shots=shots,
                shot_budget=args.budget,
                maxiter=100_000,
                label=label,
            )
            study.add(label, runs)
            med = float(np.median([r.noiseless_error for r in runs]))
            cells.append(f"{med:>13.3e}")
            del t0
        print(f"{shots:>11}{args.budget // (2 * shots):>8}" + "".join(cells), flush=True)

    print()
    best = min(
        study.aggregates().items(),
        key=lambda kv: kv[1].stat("noiseless_error").get("median", np.inf),
    )
    print(f"best configuration: {best[0]}  "
          f"median error {best[1].stat('noiseless_error')['median']:.3e}")
    path = study.save()
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
