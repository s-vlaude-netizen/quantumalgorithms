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
| `qres/estimator.py` | shot-based expectation values with resource accounting |
| `qres/ansatz.py` | hardware-efficient and chemistry-inspired trial states |
| `qres/optimizers.py` | COBYLA/SPSA baselines plus shot-adaptive optimisers |
| `qres/vqe.py` | end-to-end VQE driver |
| `qres/bench.py` | multi-seed studies, paired comparisons, sign tests |

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

3. **Negative results are recorded.** `RESEARCH_LOG.md` keeps what did not work
   alongside what did. An idea that fails under realistic noise is a result.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m experiments.exp001_measurement_budget --size small --env ideal
```

Noise environment specs: `ideal`, `small` (fake_manila, 5q), `medium`
(fake_kolkata, 27q), `large` (fake_brisbane, 127q Eagle), `heron`
(fake_torino), and any of those with an error scale — `medium@0.5x` halves all
gate error rates, to ask how good hardware would have to get.

See `RESEARCH_LOG.md` for findings and `NEXT_STEPS.md` for the open queue.
