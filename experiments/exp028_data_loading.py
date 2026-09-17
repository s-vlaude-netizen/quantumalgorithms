"""Experiment 028 -- can a quantum computer help machine learning, and LLMs?

The question came in in two halves.  The first is whether learning could happen
*directly* on probabilistic hardware, using the fact that measurement is
stochastic as the resource rather than as the noise -- possibly with no trained
quantum parameters at all.  The second is whether any of this reaches large
language models.

**The first half is a real research programme and the intuition behind it is
sound.**  Two published lines are exactly the thing described:

* **Quantum reservoir computing / quantum extreme learning machines.**  The
  quantum dynamics are fixed and *untrained*; only a classical linear readout is
  fitted.  "No trainable quantum parameters, the dynamics are the model" is the
  defining feature, not a workaround.
* **Quantum circuit Born machines.**  The model *is* the measurement
  distribution -- sampling is the forward pass.  And this is where the strongest
  known separation lives: sampling from IQP and QAOA circuit families is
  classically intractable up to multiplicative error under standard complexity
  assumptions (Coyle et al., *The Born supremacy*, npj Quantum Information 2020).

So "stochasticity as the advantage" is aligned with the one part of quantum
computing where hardness is best established.  **But hardness of sampling is not
usefulness of learning**, and the open problem in that literature is precisely
generalisation -- whether a Born machine produces valid novel samples or
memorises its training set.

**The second half has a decisive, measurable obstacle, and this experiment
measures it.**  To compute on classical data, the data has to get in.  Encoding a
general vector of dimension `d` into amplitudes is a state-preparation circuit,
and if that costs `Theta(d)` gates then **no algorithm that must read its whole
input can be exponentially faster than reading it** -- the speedup is spent
before the algorithm starts.  This is the "input problem", and it is the same
barrier that dequantization results attack from the other side: when a quantum
ML speedup assumes QRAM state preparation, matching the classical algorithm with
l2-norm sampling access removes the exponential gap (Tang and successors;
*Dequantizing algorithms to understand quantum advantage in machine learning*,
Nature Reviews Physics 2022).

So this measures one number:

    how many two-qubit gates does it cost to load a general classical vector of
    dimension d into a quantum state?

and reports what it implies at the dimensions machine learning actually uses.

**The loophole is measured too**, because reporting only the barrier would be as
selective as reporting only the speedup: a *sparse* vector with `s` nonzeros
loads in `O(s)`, not `O(d)`.  Structure in the input is the escape route, and the
experiment measures how much of one is needed.

Run:  python -m experiments.exp028_data_loading
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

from qres.bench import RESULTS_DIR

from experiments.exp025_trapped_ion import BASIS
from experiments.exp026_logical_qubit_budget import fit_power_law

#: qubit counts to sweep; d = 2^n, so 12 qubits is already a 4096-dim vector
QUBITS = (2, 3, 4, 5, 6, 7, 8, 9, 10)

#: sparsity levels, as a fraction of d, for the structured-input arm
SPARSITIES = (1.0, 0.5, 0.25, 0.1, 0.05)

#: dimensions that matter in practice, for the closing table
ML_DIMENSIONS = {
    "a 768-dim BERT embedding": 768,
    "a 4096-dim LLaMA hidden state": 4096,
    "one 4096x4096 weight matrix": 4096 * 4096,
    "a 70B-parameter model": 70_000_000_000,
}


def loading_cost(vector: np.ndarray, seed: int = 0) -> int:
    """Two-qubit gates to prepare ``vector`` as amplitudes."""
    vector = vector / np.linalg.norm(vector)
    qubits = int(round(np.log2(vector.size)))
    circuit = QuantumCircuit(qubits)
    circuit.append(StatePreparation(vector), range(qubits))
    isa = transpile(circuit, basis_gates=list(BASIS), optimization_level=3,
                    seed_transpiler=seed)
    return sum(
        1 for inst in isa.data
        if len(inst.qubits) == 2 and inst.operation.name != "barrier"
    )


def compressed_loading_cost(nonzeros: int, rng, seed: int = 0) -> int:
    """Gates to load a vector whose support is ``nonzeros`` known basis states.

    Qiskit's ``StatePreparation`` is a *dense* decomposition: handed a vector
    with 51 nonzeros out of 1024 it emits the same 1013 gates as for a full one,
    because it never inspects the zeros.  A first version of this experiment
    measured exactly that and would have reported "sparsity buys nothing", which
    is a fact about the library and not about the problem.

    The honest version prepares the `s` amplitudes on ``ceil(log2 s)`` qubits and
    leaves the rest in |0>, so the cost tracks the number of nonzeros rather than
    the dimension.  That is a real construction, and its caveat is explicit: it
    assumes the support is the first `s` basis states.  An arbitrary support
    needs an index permutation on top, which is a known additional cost and is
    not measured here.
    """
    qubits = max(1, int(np.ceil(np.log2(max(nonzeros, 2)))))
    return loading_cost(rng.normal(size=2**qubits), seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-qubits", type=int, default=max(QUBITS))
    args = ap.parse_args()

    print("=== experiment 028 :: what does it cost to get classical data in? ===")
    print("Amplitude encoding of a general vector, transpiled to a real basis.\n")

    rng = np.random.default_rng(7)

    # ------------------------------------------------- 1. dense: the barrier
    print("--- 1. a general (dense) vector ---")
    print(f"{'qubits':>8}{'dimension d':>14}{'2q gates':>11}{'gates/d':>10}")
    print("-" * 43)
    dimensions, gates = [], []
    for qubits in QUBITS:
        if qubits > args.max_qubits:
            break
        dimension = 2**qubits
        started = time.perf_counter()
        counted = loading_cost(rng.normal(size=dimension))
        dimensions.append(dimension)
        gates.append(counted)
        print(f"{qubits:>8}{dimension:>14}{counted:>11}{counted / dimension:>10.2f}"
              f"   ({time.perf_counter() - started:.1f}s)", flush=True)

    # The ratio, not the fitted exponent, is the honest headline here. A log-log
    # fit over the whole range reads high because the small-d points curve --
    # 1 gate at d=4 is 0.25 per dimension against 0.99 at d=1024 -- so the fit is
    # reported on the upper half, where the asymptotics have set in, and the
    # ratio is printed beside it as the thing that actually converges.
    half = len(dimensions) // 2
    slope, intercept, stderr = fit_power_law(dimensions[half:], gates[half:])
    ratio = gates[-1] / dimensions[-1]
    print(f"\n  gates/d converges to {ratio:.3f}  (gates = d - {dimensions[-1] - gates[-1]} "
          f"at d = {dimensions[-1]})")
    print(f"  fitted on the upper half: gates ~ d^{slope:.3f} +- {stderr:.3f}")
    print("  Loading is LINEAR in the data size. Reading the vector classically is")
    print("  also linear, so the loading step alone costs what the classical")
    print("  algorithm costs in total -- before the quantum algorithm does anything.")

    # ------------------------------------------------- 2. sparse: the loophole
    print("\n--- 2. the loophole: input with structure ---")
    print("Cost tracks the NONZEROS, not the dimension -- support assumed known.")
    print(f"{'sparsity':>10}{'nonzeros':>11}{'2q gates':>11}{'vs dense':>10}")
    print("-" * 42)
    dimension = 2 ** min(10, args.max_qubits)
    dense_reference = loading_cost(rng.normal(size=dimension))
    sparse_rows = []
    for fraction in SPARSITIES:
        nonzeros = max(1, int(round(fraction * dimension)))
        counted = compressed_loading_cost(nonzeros, rng)
        sparse_rows.append({
            "sparsity": fraction,
            "nonzeros": nonzeros,
            "two_qubit_gates": counted,
            "vs_dense": counted / max(dense_reference, 1),
        })
        print(f"{fraction:>10.2f}{nonzeros:>11}{counted:>11}"
              f"{counted / max(dense_reference, 1):>10.2f}", flush=True)

    # ---------------------------------------------------- 3. what it means
    print("\n--- 3. what that is at the dimensions machine learning uses ---")
    print(f"{'object':>32}{'dimension':>16}{'2q gates to load':>19}")
    print("-" * 67)
    # Extrapolated with the measured RATIO rather than the fitted exponent: the
    # ratio is the quantity that converged, and using a slope of 1.2 here would
    # inflate a 70B model by a further 50x on nothing but small-d curvature.
    implications = []
    for label, dimension in ML_DIMENSIONS.items():
        predicted = float(dimension * ratio)
        implications.append({"object": label, "dimension": dimension,
                             "predicted_gates": predicted})
        print(f"{label:>32}{dimension:>16.3e}{predicted:>19.3e}")

    print("\n--- the answer ---")
    print("For LARGE LANGUAGE MODELS specifically: there is no route through this.")
    print("Loading one weight matrix costs more two-qubit gates than every gate")
    print("this repository has ever counted, and a 70B model is ~1e11 of them at")
    print("error rates that need to be ~1e-15 per gate to survive the depth.")
    print("The data-loading step is linear, which is what the classical method")
    print("already pays in total. That is not a hardware-generation problem.")
    print("\nFor MACHINE LEARNING more broadly the honest answer is 'not settled',")
    print("and the reason is row 2: input that is SPARSE or GENERATED rather than")
    print("read does not pay this cost. That is exactly where the live proposals")
    print("sit -- Born machines, whose input is a circuit rather than a dataset,")
    print("and quantum reservoir computing, whose quantum part is never trained.")
    print("Both have a real complexity-theoretic basis for sampling hardness.")
    print("Neither has a demonstrated learning advantage on data anyone cares about,")
    print("and this repository already measured its own QML negative (Result 68).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp028_data_loading.json"
    with open(path, "w") as fh:
        json.dump({
            "basis": list(BASIS),
            "dimensions": dimensions,
            "two_qubit_gates": gates,
            "exponent": slope,
            "exponent_stderr": stderr,
            "sparse_reference_dimension": dimension,
            "sparse": sparse_rows,
            "implications": implications,
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
