# qres — Quantum Algorithm Research

An experimental workbench for finding quantum algorithms that are actually
cheaper to run, measured under **real device noise** rather than in the
noiseless idealisation where most published speedups live.

## The answer, since 66 results is a lot to read

The task was: find quantum algorithms that are useful for real problems, with a
measurable reduction in runtime or resources at equal or better quality.

**The measurements say the constant factors are not the problem.** This
repository produced real reductions — 15–20× from Pauli grouping, 2.43× from
Givens compilation, 4.6× from batched ADAPT, 16× from readout mitigation, 3.2×
from matching an extrapolation to the physics — and every one sits in front of a
requirement between 10³ and 10⁸ away.

The binding quantity is the **median two-qubit gate error**, and the cost of a
gate simply *is* that number (measured across seven device generations, ratio
0.64–1.09). So the threshold for any molecule is `chemical accuracy / gates`,
and gates scale as `N^5.12` in spatial orbitals:

| molecule | 2-qubit gates | needed 2q error | **factor from the best device today** |
|---|---|---|---|
| H₂ | 4 | 4.0e-4 | **3×** |
| H₄ | 1 471 | 1.1e-6 | 1 168× |
| LiH | 9 103 | 1.8e-7 | 7 226× |
| a 20-orbital fragment | 5.2M | 3.1e-10 | 4 157 332× |
| a 50-orbital drug molecule | 572M | 2.8e-12 | **453 677 973×** |

H₂ needs 3× — genuinely reachable. H₂ at STO-3G is also a 2×2 diagonalisation
that exact classical methods solve to twelve decimals in 35 ms. **The molecules
that fit the hardware are exactly the ones with no reason to be computed this
way**, and the gap opens at N^5.12 the moment they stop being trivial.

The same shape holds elsewhere, for different reasons. MaxCut *fits* the gate
budget and still loses to a fifty-line hill-climber running for one second.
Protein folding fails on both the classical baseline and the encoding. And a
quantum kernel — the only thing here that fits the hardware comfortably, six
gates per entry, unaffected by device noise — is at chance on periodic data, on
data built from exactly the pairwise-difference structure its entangling layer
encodes, and on parity, the one dataset where classical methods are weak. It
wins only on labels its own circuit produced.

**Four problem classes, four different failure points, all measured**: cost, a
strong competitor, an intractable encoding, and a method whose only success is
on data it wrote itself.

**One important bound on all of the above.** Every measurement here is
*bare-metal* — physical gates, no error correction — and in that regime the
verdict is exactly as measured. But sizing the error-corrected case with the
same measured inputs changes the shape: the code distance grows only
logarithmically in the required fidelity, so 4.5 × 10⁸ in gate error becomes
**~1.3 × 10⁵ physical qubits** for a drug-sized molecule, and 4 356 for H₄ —
magic-state distillation included, which costs only 1.25× at that size because
the factory is a fixed footprint while the data register grows. That is an
engineering target where the bare-metal number is not one. *"NISQ chemistry is
hopeless" and "quantum chemistry is hopeless" are different claims, and this
repository measured the first.*

**The unmeasured dimension is time, not qubits.** That drug molecule needs
**1.5 × 10⁸ T gates**, and a distillation factory emits magic states at a finite
rate — one factory feeding them sequentially is months of wall-clock. Parallel
factories trade that back into qubits at roughly linear cost, and this model does
not attempt it. The qubit figure is the one to stand behind; the time figure is
the one to ask about next.

Everything below is how that was established, and it is worth reading mainly for
the method: every claim here is a measurement, several of them corrections to
earlier claims in this same repository.

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
| `qres/problems/folding.py` | HP-lattice protein folding, exact and heuristic solvers |
| `qres/measurement.py` | Pauli grouping (QWC / general commuting) and shot allocation |
| `qres/covariance.py` | covariance-aware grouping and local partition refinement |
| `qres/estimator.py` | shot-based expectation values with resource accounting |
| `qres/ansatz.py` | hardware-efficient and coupled-cluster trial states |
| `qres/fermionic.py` | excitation gates compiled directly as Givens rotations |
| `qres/optimizers.py` | COBYLA/SPSA baselines plus shot-adaptive optimisers |
| `qres/classical_optimization.py` | greedy / local search / Goemans-Williamson / iterated local search |
| `qres/classical.py` | HF, MP2, CCSD, CCSD(T), FCI reference energies |
| `qres/lightcone.py` | exact QAOA energies on graphs too large to simulate |
| `qres/mitigation.py` | zero-noise extrapolation and readout correction, budget-matched |
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

**The gap is ~1000×, and it is now priced per gate.** A two-qubit gate costs
about **4.5 mHa of bias** on H₄ under a Heron-class model — measured directly by
padding the Hartree-Fock reference with cancelling `CX CX` pairs, so the ideal
state is unchanged and only the gate count moves. Chemical accuracy is 1.6 mHa,
so **the budget is less than one two-qubit gate**. Reaching chemical accuracy on
H₄ needs ~1300 with UCCSD, or 665 with the batched ADAPT ansatz.

Even at *zero* entangling gates, the best mitigation here lands at 4.0e-3 — 2.5×
outside chemical accuracy, on a state that requires no computation. And
mitigation's value decays with depth: 11.2× at 0 gates, 1.6× at 64, 1.12× at
256, because it subtracts a fixed readout bias from a growing gate bias.

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
- **SPSA over COBYLA: 2× lower error on H₂, 15 wins of 16, p = 0.001** — and it
  only appears once shot noise is genuinely resampled between evaluations
- **the shot ladder** — restart a model-based optimiser at escalating precision
  — halves SPSA's error (p = 0.006) and is the first thing here to reach
  chemical accuracy with any reliability: 7/16 seeds on H₂ at 12.8M shots

**That qualifier is load-bearing, and it was measured rather than assumed.** On
an ideal simulator, 16× the shots buys 3.0× the accuracy — textbook `1/√n`. On a
Heron-class noise model at the same depths it buys **4.3%**, because the error
there equals the device bias to three digits and no shot allocation reaches
bias. The interquartile range still falls as √n exactly as designed; the total
error simply contains almost no statistics to improve.

**And once the bias was decomposed, it turned out to be almost entirely
readout.** Switching the noise channels off one at a time: 97% of the H₄ bias is
readout error and thermal relaxation contributes nothing measurable. That
predicts zero-noise extrapolation will fail here — folding amplifies state
preparation, which for a Hartree-Fock reference is two X gates — and it does, at
1.04–1.06× *worse* than doing nothing. Readout correction, charged the same
total shots with calibration paid out of that budget, gives **16×**: 5.263e-2 →
3.245e-3 Hartree, the first genuine reduction in device error in this project.

The cheap calibration beats the exact one. Assuming independent per-qubit
readout replaces 2ⁿ calibration circuits with 2, and at a fixed budget those two
get 2ⁿ⁻¹ times more shots each — worth 1.7× over exact calibration despite being
an approximation, and it removes the 2ⁿ wall entirely. At 2% of the shot budget
it is within 3% of its own best. The residual bias (2.2e-3) sits at the
readout-free floor (1.7e-3), so readout mitigation on this system is finished:
the 2× still separating it from chemical accuracy is gate error.

**With a real ansatz the verdict inverts.** Swapping the HF reference for a
four-parameter paired-doubles ansatz (317 two-qubit gates) moves the gate share
of the bias from 4.7% to 95.4%. Readout mitigation then buys 4%; ZNE buys 3.4×,
but only with the right extrapolation — a depolarising channel *saturates*, so
`E(s) = a + b r^s` is the correct form and beats a straight line by **3.2× on
identical measured data**. The usable rule is to decompose the bias first and
then pick the method; both times, choosing by prominence picked the wrong one.

The sobering number: that best mitigated result is still 0.408 Hartree, 255×
chemical accuracy. Those 317 gates cost 26× more error than all the mitigation
recovers, which is the ~32× depth gap above, measured from the other direction.

## Which problems fit the device at all

The per-gate cost turns out to be a **device constant**: ~0.1% of the
observable's scale per two-qubit gate, measured at 0.089–0.113% across chemistry
and MaxCut, two problem classes with different circuits, Hamiltonians and units.
So `max two-qubit gates ≈ required relative accuracy / 0.001`, and that decides
which problems are even candidates:

| | accuracy requirement | classical baseline | encoding cost |
|---|---|---|---|
| **chemistry** | **fails** — 0.019% gives a 0.2-gate budget; an ansatz needs 665–1300 | fails — FCI exact in 100 ms | — |
| **MaxCut** | passes — 5%, budget 50, QAOA p=1 uses 48 | **fails** — a 1 ms hill-climb beats it | passes |
| **HP folding** | passes — 7%, budget ~70 | **fails** — 3 of 4 literature optima in under a minute | **fails** — 1 538 gates at N = 6 |
| **quantum ML** | passes — percent-level | **fails** — at chance (0.51–0.55) on periodic, interaction and parity data; wins only on data its own feature map generated | **passes** — 6 gates per kernel entry |

Three classes, three different failure points, each measured rather than argued.

**And the wall is this hardware generation, sized.** Every noise result above is
on one device model, so the per-gate cost was re-measured across seven
calibration snapshots spanning generations (median two-qubit error 0.0013 to
0.0077). The cost per gate simply **is** the device's own two-qubit error rate —
ratio 0.64–1.09, fitted as `(2q error)^0.93` with log-log correlation 0.952.

| device | median 2q error | cost per gate | budget in gates |
|---|---|---|---|
| fake_boston | 0.00127 | 1.219e-3 | **1.31** |
| fake_torino | 0.00419 | 4.473e-3 | 0.36 |
| fake_brisbane | 0.00772 | 8.391e-3 | 0.19 |

Batched ADAPT on H₄ needs 665 gates, which requires a two-qubit error rate of
**~1.6e-6**. The best device here is 0.00127 — **a factor of ~800**. Recent
generations delivered 6×.

So the single number that decides whether any of this becomes useful is the
median two-qubit gate error. Everything else in this repository moves the answer
by a constant factor; that one moves it linearly.
Folding fails on the encoding: self-avoidance is a global constraint on the turn
variables, so the natural Hamiltonian is fully dense — 2ⁿ Pauli terms at maximum
weight — and six residues already costs 1 538 two-qubit gates per QAOA layer, at
a size exact enumeration solves in microseconds. Published low-locality
encodings spend ancilla qubits to avoid this and have not been measured here;
that is the one open candidate.

Worth knowing separately: the transpiler is already doing 8× of readout
mitigation for free. At optimisation level 2 it placed a six-qubit problem on
physical qubits [66, 5, 87, 81, 60, 51], whose readout fidelity is 0.945–0.976
against 0.684–0.747 on qubits 0–5.

So the ranking above **inverts** under device noise. General-commuting grouping
beats QWC by 1.7× on LiH in the ideal case (predicted 0.56 from the variance
ratio, measured 0.59) and *loses* by 1.37× on `heron` — because its Clifford
basis changes cost 125 two-qubit gates against QWC's zero. Both then land at
5–9e-2 Hartree against chemical accuracy at 1.6e-3, from the Hartree-Fock state
alone, so the difference between them is a rounding error on a number already
30–55× out of range.

## What was measured and closed

Two whole lines are closed by measurement rather than by opinion. Both are kept
as regression tests, so either reopens if the numbers ever invert.

**Measurement schemes.** Five, ranked on one metric (total estimator variance
under Neyman allocation) at the same reference state:
general commuting **1.00** · QWC 1.0–3.2 · derandomised shadows 3.0–4.8 ·
double factorisation 4.0–10.1 · random-Pauli shadows 38–369.
What this package already defaults to wins on every molecule, and none of the
alternatives changes the scaling.

**Optimisers.** Three approaches, none beats the shot ladder: multi-start is
worse at small budgets (2.44×, p = 0.004), automatic trust-radius estimation
fails three different ways because the signal sits below the shot noise, and a
STORM-style stochastic trust region never gets closer than 3.3× (p ≤ 0.001)
across four orders of magnitude in its accuracy constant.

## The classical baseline

Stated first, because without it none of the numbers below mean anything. Same
molecules, same Hamiltonians, one CPU core:

| molecule | CCSD(T) | **FCI (exact)** | best VQE here |
|---|---|---|---|
| H₂ | 1.6e-7 / 95 ms | **2.7e-14 / 35 ms** | 1.69e-3 @ 12.8M shots |
| H₄ | 3.9e-6 / 242 ms | **3.6e-12 / 132 ms** | 3.66e-2 @ 256M shots |
| LiH | 2.1e-6 / 217 ms | **2.2e-12 / 141 ms** | — |

Exact diagonalisation solves all three to twelve decimals in about a tenth of a
second. The VQE here reaches chemical accuracy on none of them at any budget
tested. These molecules are *instruments* — chosen because the exact answer is
available to check against — not targets.

**The same holds on the optimisation side, and it should not have.** MaxCut is
NP-hard, so the classical competitor is only an *approximation* algorithm —
Goemans-Williamson, proven ratio 0.878. That predicts an opening. There isn't
one at any size that can be scored: across five instance families and 20 seeds
each, GW returns the **exact** optimum on 100% of instances up to n = 20, in
66–210 ms.

| n | greedy | local search | Goemans-Williamson | QAOA p=3 |
|---|---|---|---|---|
| 10 | 1.000 | **1.000** (0.8 ms) | 1.000 (93 ms) | 1.000 (178k shots) |
| 14 | 0.842 | **1.000** (1.3 ms) | 1.000 (210 ms) | 1.000 (129k shots) |
| 18 | 0.840 | **1.000** (0.9 ms) | 1.000 (141 ms) | 1.000 (127k shots) |

So any QAOA benchmark at n ≤ 20 reporting an approximation ratio below 1.0 is
reporting it on instances a 100 ms classical algorithm solves exactly.

**Where that stops** was then measured directly, by agreement between the SDP and
an iterated local search at *matched wall-clock* — because past n ≈ 24 there is
no brute force left to check against:

| n | 1 ms hill-climb is optimal | strong methods agree | who wins when they differ |
|---|---|---|---|
| ≤ 20 | 12/12 | 12/12 | — |
| 24–40 | 11/12 → 4/12 | 12/12 | — |
| 60 | 0/12 | 10/12 | local search |
| 100 | 0/12 | **0/12** | local search, by 2.55% |

Classical certainty ends between **n = 40 and n = 60**, and Goemans-Williamson
never wins once at a matched budget. So a QAOA result that means anything needs
n ≥ 60 — and the target there is not the 0.878 guarantee, it is a fifty-line
hill-climber finding a better cut in about a second.

**So that is where QAOA was measured.** 2⁶⁰ amplitudes do not exist, but at depth
`p` a QAOA edge expectation depends only on vertices within `p` hops, so a
1 000-vertex energy is a sum of independent 14-qubit simulations — exact, and
checked against full statevector simulation to 7e-15 and against Farhi's
analytic 0.6924 for p=1 on 3-regular graphs:

| n | QAOA p=1 | QAOA p=2 | **1 ms hill-climb** |
|---|---|---|---|
| 60 | 0.767 | 0.839 | **0.988** |
| 200 | 0.758 | 0.826 | **0.949** |
| 1 000 | 0.800 | 0.873 | **0.997** |

QAOA loses at every size, and the accounting is as generous as it can be made:
that is the *exact expected* cut, with optimal angles, no noise and infinite
shots — none of which is free on hardware. Angles transfer across instance size
essentially perfectly (0.018% loss), which confirms the fixed-angle literature
has a real mechanism and simultaneously closes it: the instance-specific outer
loop was never the bottleneck.

Two areas, two routes, one conclusion: the sizes that can be verified have no
hard part left. Knowing exactly where that ceases is the prerequisite for every
other question here, and on the optimisation side it is now known.

## The wall

Measured, not assumed: the shots to reach a target accuracy scale as

```
shots  ~  (Σ|c|)² · n_parameters / ε²
```

which is ~3.1M for H₂ (12 parameters, Σ|c| = 2.04) and ~170M for H₄ with UCCSD
(26 parameters, Σ|c| = 10.94) — 55× for one more pair of hydrogens, against the
64× the formula predicts. And that is a *lower bound on the estimator side*: H₄
at 256M shots, above the prediction, still reaches 0/8 chemical accuracy,
because a precise evaluation is only useful if the optimiser can exploit it.

Everything in this repository buys a constant factor against that — grouping
~15×, covariance refinement ~1.6×, the shot ladder ~2×. None of them touches
the exponent.

**What `Σ|c|` actually depends on** took two attempts to get right. A fit over
five minimal-basis molecules gave `Σ|c| ~ N^2.78` in the orbital count — but two
of those five sit at the *same* N with sums differing 3.35×, and their N and
nuclear charge are 88% collinear, so the exponent was assigned by default rather
than measured. Varying one thing at a time over thirteen molecules:

| direction | measurement | result |
|---|---|---|
| more atoms, same nuclear charge | HF → H₂O → NH₃ → CH₄ (N = 6→9, Z = 10) | `N^-0.39` — **flat** |
| more basis functions, same molecule | H₂ at STO-3G → 6-31G → cc-pVDZ | `N^2.86` |

The two directions have opposite signs, so one exponent in N averages over
whichever mix the molecule set happens to contain. **Qubit count is the wrong
figure of merit**: at six orbitals, Σ|c| ranges 12.34 (LiH) to 78.32 (HF), a
6.3× spread in a quantity the shot count depends on *squared*.

For the direction that matters — approaching the basis limit, where the
chemistry actually is — `shots ~ N^5.7 · n / ε²`, and with UCCSD's `n ~ N⁴` that
is **N^9.7 in shots** against CCSD(T)'s **N⁷ in operations**. A shot is
microseconds where an operation is nanoseconds, so the unit conversion moves it
further the wrong way. On this evidence VQE as built here has no route to
beating CCSD(T) on molecular ground states.

The one lever that changes the exponent rather than the constant is `n`. An
adaptive ansatz reaching a given accuracy with `n ~ N²` would give `N^7.6` —
the same order as the classical competitor, which is where a real comparison
would start. That is the only direction the measurements support, and it is the
open item.

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
