"""Experiment 001 -- where does a VQE shot budget actually go?

Every configuration is given the *same* total number of circuit executions and
we ask what it achieved with them.  Three axes:

  grouping    none | qwc | commuting     -- how many circuits per energy
  allocation  uniform | coefficient | adaptive  -- how shots split across groups
  optimizer   cobyla | spsa | icans      -- how shots split across iterations

The headline metric is ``noiseless_error``: the exact energy at the parameters
the optimiser ended on.  That isolates "did we find the right state" from "can
we read the energy off precisely", which are different problems with different
fixes.

Run:  python -m experiments.exp001_measurement_budget --size small
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qres.bench import Study, run_over_seeds
from qres.problems.chemistry import CHEMICAL_ACCURACY_HA, build_molecule
from qres.vqe import run_vqe


def _run(seed: int, molecule: str, molecule_kwargs: dict, **kwargs):
    """Worker entry point -- builds the problem in-process (it is cached)."""
    problem = build_molecule(molecule, **molecule_kwargs)
    return run_vqe(problem, seed=seed, **kwargs)


# Controlled-rotation entanglers, so theta=0 IS the Hartree-Fock reference.
# With fixed CX entanglers the zero-parameter state is orthogonal to it and
# every run below starts 4 Ha from the answer -- see RESEARCH_LOG Result 5.
ANSATZ = "hea:2:linear:cry"

SIZES = {
    "small": dict(seeds=range(4), budget=120_000, shots=2048, molecules=["H2"]),
    "medium": dict(seeds=range(8), budget=300_000, shots=3072, molecules=["H2", "H4"]),
    "large": dict(seeds=range(12), budget=600_000, shots=4096, molecules=["H2", "H4", "LiH"]),
}

MOLECULE_KWARGS = {
    "H2": {},
    "H4": {},
    "LiH": {},
    "BeH2": dict(active_electrons=4, active_orbitals=6),
}


def stage_grouping_allocation(study, molecule, cfg, env):
    """Axis 1+2: how many circuits, and how shots split across them."""
    for grouping, allocation in itertools.product(
        ["none", "qwc", "commuting"], ["uniform", "coefficient", "adaptive"]
    ):
        label = f"{molecule}|cobyla|{grouping}|{allocation}"
        t0 = time.perf_counter()
        runs = run_over_seeds(
            _run,
            cfg["seeds"],
            molecule=molecule,
            molecule_kwargs=MOLECULE_KWARGS[molecule],
            ansatz=ANSATZ,
            environment=env,
            grouping=grouping,
            allocation=allocation,
            optimizer="cobyla",
            shots=cfg["shots"],
            shot_budget=cfg["budget"],
            maxiter=10_000,
            label=label,
        )
        study.add(label, runs)
        study.save()  # save after every config -- a slow tail config must not
                      # hold the completed ones hostage
        print(f"  {label:<44} {time.perf_counter() - t0:6.1f}s", flush=True)


def stage_optimizers(
    study, molecule, cfg, env, grouping="commuting", allocation="adaptive", optimizers=None
):
    """Axis 3: how shots split across optimiser iterations."""
    for optimizer in optimizers or ["cobyla", "spsa"]:
        label = f"{molecule}|{optimizer}|{grouping}|{allocation}"
        if label in study.results:
            continue
        t0 = time.perf_counter()
        extra = {}
        if optimizer == "icans":
            # icans picks its own per-parameter shot counts; the `shots`
            # argument only sets the audit evaluation size
            extra = dict(optimizer_kwargs=dict(max_shots=cfg["shots"], min_shots=8))
        runs = run_over_seeds(
            _run,
            cfg["seeds"],
            molecule=molecule,
            molecule_kwargs=MOLECULE_KWARGS[molecule],
            ansatz=ANSATZ,
            environment=env,
            grouping=grouping,
            allocation=allocation,
            optimizer=optimizer,
            shots=cfg["shots"],
            shot_budget=cfg["budget"],
            maxiter=10_000,
            label=label,
            **extra,
        )
        study.add(label, runs)
        study.save()
        print(f"  {label:<44} {time.perf_counter() - t0:6.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(SIZES), default="small")
    ap.add_argument("--env", default="ideal", help="noise env spec, e.g. ideal / small / medium")
    ap.add_argument("--name", default=None)
    ap.add_argument("--molecules", default=None,
                    help="comma-separated override of the size preset's molecule list")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--stages", default="grouping,optimizers",
                    help="comma-separated subset of: grouping, optimizers")
    ap.add_argument("--optimizers", default="cobyla,spsa",
                    help="optimisers for the optimizer stage. icans needs 2n circuit "
                         "evaluations per step and is impractical above ~20 parameters: "
                         "measured ~3.5 h for 8 seeds of H4 (46 params).")
    args = ap.parse_args()

    cfg = dict(SIZES[args.size])
    cfg["seeds"] = list(range(args.seeds)) if args.seeds else list(cfg["seeds"])
    if args.molecules:
        cfg["molecules"] = args.molecules.split(",")
    if args.budget:
        cfg["budget"] = args.budget
    name = args.name or (
        f"exp001_{'-'.join(cfg['molecules'])}_{args.env.replace('@', '')}_{cfg['budget']}"
    )
    study = Study(name=name, seeds=cfg["seeds"])

    print(f"=== experiment 001 [{args.size}] env={args.env} ===")
    print(f"budget={cfg['budget']:,} shots/eval={cfg['shots']} seeds={len(cfg['seeds'])}")

    for molecule in cfg["molecules"]:
        problem = build_molecule(molecule, **MOLECULE_KWARGS[molecule])
        print(f"\n{molecule}: {problem.num_qubits}q, {len(problem.hamiltonian)} terms, "
              f"FCI={problem.fci_energy:.6f}")
        stages = args.stages.split(",")
        if "grouping" in stages:
            stage_grouping_allocation(study, molecule, cfg, args.env)
        if "optimizers" in stages:
            stage_optimizers(study, molecule, cfg, args.env, optimizers=args.optimizers.split(","))

    print("\n" + study.table())
    print()
    for molecule in cfg["molecules"]:
        base = f"{molecule}|cobyla|qwc|uniform"
        if base in study.results:
            print(study.compare(base))
            print()

    path = study.save()
    print(f"chemical accuracy = {CHEMICAL_ACCURACY_HA:.2e} Ha")
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
