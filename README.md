# qres — Quantum Algorithm Research

An experimental workbench for finding quantum algorithms that are actually
cheaper to run, measured under **real device noise** rather than in the
noiseless idealisation where most published speedups live.

## The premise

For near-term quantum algorithms, the thing that costs money and time is
**circuit executions**, not floating-point operations. A VQE run on hardware
spends its life doing: prepare a state, rotate into a measurement basis, sample,
repeat a few million times. So this package optimises against

```
hardware_seconds =  n_shots   × (circuit_duration + reset_delay)
                  + n_circuits × per_circuit_overhead
```

with `circuit_duration` taken from the transpiled circuit under the target
device's *real* gate timings. Simulator wall-clock is recorded but never
optimised against — it measures the wrong machine.

Every algorithm reports into a shared `ResourceLedger`, so any two are charged
for the same things and the comparison between them means something.

## Layout

| module | what it does |
|---|---|
| `qres/resources.py` | the cost model and per-run resource ledger |
| `qres/noise.py` | noise environments from IBM device calibration snapshots |
| `qres/problems/chemistry.py` | molecular Hamiltonians (PySCF → qubit operators) with exact references |
| `qres/problems/optimization.py` | MaxCut, portfolio selection as Ising Hamiltonians |
| `qres/measurement.py` | Pauli grouping (QWC / general commuting) and shot allocation |
| `qres/covariance.py` | covariance-aware grouping and local partition refinement |
| `qres/estimator.py` | shot-based expectation values with resource accounting |
| `qres/ansatz.py` | hardware-efficient and coupled-cluster trial states |
| `qres/fermionic.py` | excitation gates compiled directly as Givens rotations |
| `qres/optimizers.py` | COBYLA/SPSA baselines plus shot-adaptive optimisers |
| `qres/vqe.py`, `qres/qaoa.py` | end-to-end drivers |
| `qres/bench.py` | multi-seed studies, paired comparisons, sign tests |

## What is known so far

Full detail with numbers in `RESEARCH_LOG.md`. The short version:

**The binding constraint is the ansatz, not measurement.** A hardware-efficient
ansatz has *exactly zero* gradient at the Hartree-Fock determinant — 0 of 46
parameters at 2 reps, 0 of 114 at 6 — because its first-order response reaches
only single excitations and Brillouin's theorem decouples those. Correlation
energy is in the doubles. So it sits at Hartree-Fock at any depth and any
budget, and every measurement-side improvement has nothing to act on.

**The gap is ~32×.** Reaching chemical accuracy on H₄ needs ~1300 two-qubit
gates; surviving a Heron-class device at fidelity > 0.5 allows ~40.

**What does work, where a run is actually shot-noise-limited:**

- general-commuting grouping: 15–20× fewer circuits than ungrouped
- covariance-aware grouping with local refinement: 0.63 variance ratio on LiH,
  validated against 300 sampled estimates at a fixed state
- direct Givens compilation of double excitations: 2.43× fewer two-qubit gates
  than the Trotterised build (1.31× after device routing), verified to be the
  same operator as `qiskit_nature`'s to 8e-13
- variance-adaptive (Neyman) shot allocation: median error ratio ~0.78 on H₂,
  ≈ 1.7× fewer shots — **suggestive, not significant** (14/10 seeds, p = 0.54).
  An earlier 4-seed run of this reported 3.8×; it did not replicate.
- **SPSA over COBYLA: 2× lower error on H₂, 15 wins of 16, p = 0.001** — the
  one statistically significant positive result here, and it only appears once
  shot noise is genuinely resampled between evaluations (see below)

**Sizing rule for any budget-matched comparison** — below this it measures the
optimiser's simplex construction, not the algorithm:

```
usable budget  ≳  10 × n_parameters × shots_per_evaluation
```

## Discipline

Three rules this repository tries to hold itself to, because they are where
quantum-algorithm benchmarks usually go wrong:

1. **Correctness is verified algebraically, not by plausibility.** The
   measurement grouping is checked against exact statevector expectation
   values to machine precision (`tests/test_measurement.py`); a basis change
   that is subtly wrong produces energies that look fine and quietly corrupt
   every downstream result.

2. **Comparisons are budget-matched and seed-paired.** Configurations are
   stopped at exactly the same shot count, run over many seeds, and compared
   with a sign test — the spread across random initialisations is routinely
   larger than the effect being measured.

3. **Negative results and retractions are recorded.** `RESEARCH_LOG.md` keeps
   what did not work alongside what did, and corrections stay next to the
   claims they correct rather than replacing them. Two headline results from
   session 1 were overturned by later measurements in the same session; both
   are still in the log.

4. **Enough seeds to tell an effect from nothing.** Measured, not assumed: at
   4 seeds a 4× apparent effect on H₂ shrank to 1.3× and lost significance at
   24. Four seeds is not enough for VQE error comparisons.

5. **The simulator has to actually be stochastic.** A fixed `seed_simulator`
   makes Aer return identical counts for identical circuits, so the optimiser
   faces a frozen rough landscape it can fit itself to rather than noise it
   must average over. This was live for all of session 1 and reversed the
   COBYLA-vs-SPSA verdict once fixed. `tests/test_noise.py` now checks it.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q

# where does a shot budget go?  grouping x allocation x optimiser
.venv/bin/python -m experiments.exp001_measurement_budget --size small --env ideal
# how finely should a fixed budget be sliced?
.venv/bin/python -m experiments.exp002_shots_per_evaluation --molecule H2
# does covariance-aware grouping beat count-minimising?
.venv/bin/python -m experiments.exp003_covariance_grouping
# how many parameters can a budget afford?
.venv/bin/python -m experiments.exp004_ansatz_size --molecule H4
# which ansatz clears gradient, accuracy AND device fidelity?  (so far: none)
.venv/bin/python -m experiments.exp005_excitation_ansatz --molecule H4
```

Noise environment specs: `ideal`, `small` (fake_manila, 5q), `medium`
(fake_kolkata, 27q), `large` (fake_brisbane, 127q Eagle), `heron`
(fake_torino), and any of those with an error scale — `medium@0.5x` halves all
gate error rates, to ask how good hardware would have to get.

See `RESEARCH_LOG.md` for findings and `NEXT_STEPS.md` for the open queue.
