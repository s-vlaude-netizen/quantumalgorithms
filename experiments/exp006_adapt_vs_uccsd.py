"""Experiment 006 -- does ADAPT-VQE's parameter reduction actually save cost?

ADAPT reaches chemical accuracy with far fewer parameters than a fixed UCCSD
(RESEARCH_LOG Result 44): 1 against 3 on H2, 9 against 26 on H4, 5 against 92 on
LiH.  That is not the same as costing less, and the difference is the point of
this experiment.

Three costs have to be charged, and only the first is what a parameter count
measures:

1. **Optimiser evaluations.** ADAPT re-optimises after *every* growth step, so
   its cost tracks the number of steps rather than the final parameter count.
2. **Gradient sweeps.** Each step measures every pool operator's gradient. The
   commutators ``[H, A_k]`` barely share measurement groups with ``H`` -- on LiH
   their union is 29 855 terms in 1 154 groups against H's 631 in 41.
3. **The precision each needs.** A gradient only has to *rank* operators, and
   ranks correctly at sigma = 0.01 where an energy needs 1.6e-3. Shots go as
   1/sigma^2, so a sweep is ~39x cheaper than its group count implies.

Gradient-free optimisation throughout, because that is the setting that matters:
under shot noise there are no cheap analytic gradients.

Run:  python -m experiments.exp006_adapt_vs_uccsd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from qres.adapt import _evolution, excitation_pool, operator_gradients
from qres.ansatz import build_ansatz, hartree_fock_state
from qres.bench import RESULTS_DIR
from qres.problems.chemistry import CHEMICAL_ACCURACY_HA, build_molecule

#: Cost of one gradient sweep in energy-evaluation equivalents, after the
#: 39x discount for the looser precision ranking needs (Result 44).
SWEEP_COST = {"H2": 2.0 / 39, "H4": 10.1 / 39, "LiH": 28.1 / 39}


def uccsd_cost(problem, maxiter: int = 20_000) -> tuple[int, int, float]:
    """Evaluations for a fixed UCCSD to first reach chemical accuracy."""
    ansatz = build_ansatz("uccsd", problem)
    seen: list[float] = []

    def energy(x):
        value = float(
            np.real(Statevector(ansatz.assign_parameters(x)).expectation_value(problem.hamiltonian))
        )
        seen.append(value)
        return value

    minimize(energy, np.zeros(ansatz.num_parameters), method="COBYLA",
             options={"maxiter": maxiter})
    values = np.array(seen)
    crossed = np.flatnonzero(np.abs(values - problem.fci_energy) < CHEMICAL_ACCURACY_HA)
    reached = int(crossed[0]) + 1 if len(crossed) else -1
    best = float(np.abs(values - problem.fci_energy).min())
    return ansatz.num_parameters, reached, best


def adapt_cost(problem, max_steps: int = 15, gradient_tolerance: float = 1e-3):
    """Evaluations and sweeps for ADAPT to first reach chemical accuracy."""
    pool = excitation_pool(problem, "sd")
    reference = Statevector(hartree_fock_state(problem.hf_bitstring))

    def build(params, operators):
        state = reference
        for theta, index in zip(params, operators):
            state = state.evolve(_evolution(pool[index], theta))
        return state

    def energy(params, operators):
        return float(np.real(build(params, operators).expectation_value(problem.hamiltonian)))

    chosen: list[int] = []
    params = np.zeros(0)
    evaluations = 0
    sweeps = 0
    best = np.inf

    for _ in range(max_steps):
        gradients = operator_gradients(problem.hamiltonian, build(params, chosen), pool)
        sweeps += 1
        top = int(np.argmax(gradients))
        if gradients[top] < gradient_tolerance:
            break

        chosen.append(top)
        params = np.concatenate([params, [0.0]])
        counter = [0]

        def wrapped(x):
            counter[0] += 1
            return energy(x, chosen)

        result = minimize(wrapped, params, method="COBYLA", options={"maxiter": 20_000})
        params = np.asarray(result.x)
        evaluations += counter[0]
        best = abs(float(result.fun) - problem.fci_energy)
        if best < CHEMICAL_ACCURACY_HA:
            break

    return len(chosen), evaluations, sweeps, best, len(pool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default="H2,H4,LiH")
    args = ap.parse_args()

    print("=== experiment 006 :: ADAPT-VQE vs fixed UCCSD ===")
    print("gradient-free (COBYLA), exact energies, evaluations to chemical accuracy\n")
    header = (
        f"{'molecule':<8}{'method':<10}{'params':>8}{'evals':>9}"
        f"{'sweeps':>8}{'total':>9}{'error':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in args.molecules.split(","):
        problem = build_molecule(name)

        t0 = time.perf_counter()
        u_params, u_evals, u_best = uccsd_cost(problem)
        u_seconds = time.perf_counter() - t0
        print(
            f"{name:<8}{'UCCSD':<10}{u_params:>8}{u_evals:>9}{'-':>8}{u_evals:>9}"
            f"{u_best:>11.2e}   ({u_seconds:.0f}s)",
            flush=True,
        )

        t0 = time.perf_counter()
        a_params, a_evals, a_sweeps, a_best, pool_size = adapt_cost(problem)
        a_seconds = time.perf_counter() - t0
        sweep_cost = SWEEP_COST.get(name, 1.0)
        a_total = a_evals + a_sweeps * sweep_cost
        print(
            f"{name:<8}{'ADAPT':<10}{a_params:>8}{a_evals:>9}{a_sweeps:>8}{a_total:>9.0f}"
            f"{a_best:>11.2e}   ({a_seconds:.0f}s, pool {pool_size})",
            flush=True,
        )

        ratio = u_evals / a_total if a_total > 0 and u_evals > 0 else float("nan")
        verdict = "cheaper" if ratio > 1 else "MORE EXPENSIVE"
        print(f"{'':<8}{'-> ADAPT':<10}{ratio:>8.2f}x {verdict}\n", flush=True)

        rows.append(
            {
                "molecule": name,
                "uccsd_parameters": u_params,
                "uccsd_evaluations": u_evals,
                "adapt_parameters": a_params,
                "adapt_evaluations": a_evals,
                "adapt_sweeps": a_sweeps,
                "adapt_total": a_total,
                "ratio": ratio,
                "pool_size": pool_size,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp006_adapt_vs_uccsd.json"
    with open(path, "w") as fh:
        json.dump({"rows": rows}, fh, indent=2)
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
