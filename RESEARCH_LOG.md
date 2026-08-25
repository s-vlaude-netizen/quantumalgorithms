# Research Log

Chronological, append-only. Negative results stay in — an idea that fails under
realistic noise is a result, and re-deriving that it fails costs more than
writing it down.

Every entry records: what was asked, what was measured, what the number was,
and what it means. Claims without a measurement behind them are marked
`[unverified]`.

---

## Session 1 — 2026-08-25

### Setup

Environment: Qiskit 2.5.2, Aer 0.17.2, qiskit-ibm-runtime 0.49.0 (68 fake
backends carrying real IBM calibration snapshots), qiskit-nature 0.8.0,
PySCF 2.14.0. 4 CPU cores, 15 GB RAM.

### Framing decision: what to optimise

Simulator wall-clock is the wrong objective. On hardware the cost of a
variational algorithm is circuit executions, so the ledger optimises

```
hardware_seconds = n_shots × (circuit_duration + reset_delay)
                 + n_circuits × per_circuit_overhead
```

with `circuit_duration` computed as the critical path through the *transpiled*
circuit using the target's real per-gate durations, `reset_delay = 250 µs`
(representative of reset-by-delay on IBM Eagle/Heron), and a 1 ms per-circuit
load overhead. Simulator seconds are recorded separately for transparency and
never used to compare algorithms.

Consequence for interpreting everything below: **grouping does not reduce
shots, it reduces circuits and variance-per-shot.** At a fixed shot budget,
better grouping means each group gets more shots and the energy estimate is
tighter; it also cuts the per-circuit overhead term.

### Verified: problem encodings

Checked independently rather than trusted:

| check | result |
|---|---|
| H₂/STO-3G total FCI (electronic + nuclear) | −1.1374 Ha — matches literature |
| LiH/STO-3G total FCI | −7.882 Ha — matches literature |
| MaxCut on triangle / C₅ / K₄ | 2 / 4 / 4 — all correct |
| Portfolio QUBO → Ising vs. direct objective enumeration | exact match to 1e-8 |
| Portfolio optimum respects cardinality constraint | yes (|x| = B) |

### Verified: measurement grouping

The dangerous failure mode is a basis change that is *almost* right — energies
stay plausible and every downstream benchmark is silently wrong. So the
grouping is checked algebraically, not by sampling: for random 4-qubit states
and random 20-term Hamiltonians, the expectation values reconstructed through
each group's basis change match exact statevector expectation values to 1e-10,
for all three methods (`none`, `qwc`, `commuting`). 21 tests pass.

General-commuting groups use a Clifford obtained by inverting
`synth_circuit_from_stabilizers` on a GF(2)-independent generating subset. When
that Clifford fails to diagonalise every member (it should not, but a silent
wrong answer is unacceptable), the code falls back to term-by-term measurement
rather than returning a wrong energy.

### Result 1 — grouping effectiveness on molecular Hamiltonians

Greedy clique cover, terms ordered by |coefficient| descending:

| molecule | qubits | terms | ungrouped | QWC | commuting | commuting vs. naive |
|---|---|---|---|---|---|---|
| H₂ | 2 | 5 | 4 | 2 | 2 | 2.0× |
| H₄ | 6 | 165 | 164 | 37 | **10** | **16.4×** |
| LiH | 10 | 631 | 630 | 172 | **41** | **15.4×** |
| BeH₂ (4e,6o) | 10 | 327 | 326 | 91 | **16** | **20.4×** |

(Re-measured after the reproducibility fix in Result 8. The commuting counts —
the ones that matter — are unchanged; the QWC counts moved by one each, which is
the tie-break change and not a physical difference.)

General commuting beats QWC by 4.2–5.6× on these Hamiltonians. Grouping time is
negligible (< 1 s for LiH). This is the cheapest large win available and it is
now the package default.

### Result 2 — the hardware-efficient ansatz is not the bottleneck

Before blaming noise for VQE error, checked whether the ansatz can represent
the answer at all. Exact (statevector, BFGS, 6 restarts) optimisation of the
2-qubit H₂ Hamiltonian:

| ansatz | parameters | best exact energy error |
|---|---|---|
| hea:1 | 8 | 6.0e-12 |
| hea:2 | 12 | 1.0e-11 |
| hea:3 | 16 | 4.3e-13 |

So for H₂ the ansatz is fully expressive. A first end-to-end run
(COBYLA, 4096 shots/eval, 80 iterations) reached only 2.6e-2 Ha error on the
*noiseless* simulator — ~16× worse than chemical accuracy (1.594e-3 Ha).

**The error is the optimiser under shot noise, not the ansatz and not the
device.** That reframes the research target: the lever is how the shot budget
is spent, across measurement groups and across optimiser iterations.

### Design decision: budget parity

Comparing optimisers by tuning `maxiter` per optimiser is not a comparison. The
shot budget is now enforced inside the energy oracle (`BudgetExceeded`), so
every configuration is stopped at exactly the same number of circuit
executions and the question becomes only "how well did it spend them".

Headline metric is `noiseless_error`: the exact energy at the parameters the
optimiser ended on. This separates *did we find the right state* from *can we
read the energy off precisely* — different problems with different fixes.

### Variance model

Shot allocation across groups uses Neyman allocation, `n_g ∝ sqrt(Var_g)`, on a
running variance estimate. Two choices that differ from the literature:

1. **Variance is measured empirically from the shots already taken** — the
   per-shot group value automatically contains every intra-group covariance,
   which a `(Σ|c_i|)²` proxy gets badly wrong for large commuting groups. Costs
   no extra circuits.
2. **The estimate forgets** (`decay = 0.85` per batch). Group variances are
   *not* stationary: they change as the ansatz state moves during optimisation.
   An estimator that never forgets is biased toward the variance of a state the
   optimiser has already left. `[unverified]` — the decay value is a guess and
   needs an ablation.

Shrinkage toward the `(Σ|c_i|)²` prior with pseudo-count 64 keeps early
allocations from chasing noise in a variance estimated from a handful of shots.

### Result 3 — variance-adaptive shot allocation  ⚠️ **SUPERSEDED — see Result 18**

*Original entry, kept because the correction is the point.* Experiment 001, H₂,
ideal simulator, 120k shot budget, **4 seeds**. Paired against
`cobyla|qwc|uniform` on `noiseless_error`:

| configuration | ratio | W/L/T |
|---|---|---|
| `qwc \| coefficient` | 0.628 | 4/0/0 |
| **`qwc \| adaptive`** | **0.264** | **4/0/0** |
| `spsa \| commuting \| adaptive` | 0.487 | 4/0/0 |

At the time this was read as "Neyman allocation gives a 3.8× error reduction at
equal shot count, worth ~14× fewer shots". **That reading does not survive more
seeds.** See Result 18.

### Result 4 — a fixed shot budget has an interior optimum in shots-per-evaluation

At fixed budget `B`, spending `s` shots per evaluation buys `B/2s` iterations.
Sweep on H₂, 120k budget, 8 seeds (experiment 002):

| shots/eval | ~iters | COBYLA | SPSA |
|---|---|---|---|
| 256 | 234 | 1.44e-2 | 1.05e-2 |
| 1024 | 58 | 7.66e-3 | 7.41e-3 |
| **2048** | **29** | **3.89e-3** | 6.10e-3 |
| **4096** | **14** | 1.42e-2 | **4.90e-3** |
| 8192 | 7 | 1.91e-2 | 7.44e-3 |

Both are U-shaped with a clear interior optimum, ~5× between best and worst.
There is an analytic reason: SPSA's accumulated parameter noise over `K` steps
at fixed budget goes as `K^(2-2α)/B ≈ K^0.8/B` with `α ≈ 0.6`, so *more*
iterations means *more* accumulated noise, while too few means no convergence.

### Negative result — adaptive shot escalation across iterations does not pay

Two schemes tried, both worse than fixed-shot baselines on H₂:

* **iCANS** (per-parameter shots from gradient variance). Needs `2n` circuit
  evaluations per step — 24 for a 12-parameter ansatz — and lost badly
  (2.2e-1 vs COBYLA's 2.3e-3).
* **Adaptive SPSA controlled on gradient SNR.** The rule
  `s* = τ²σ²/(2c²g²)` diverges: the gradient genuinely goes to zero at the
  minimum, so no finite shot count meets a fixed SNR target. It saturated at
  `max_shots` within a few iterations. Measured 1.1e-2 vs plain SPSA's 6.5e-3.
* **Adaptive SPSA controlled on resolvable energy change**, `s* = (σ_E /
  (κ·|ΔE|))²`, which cannot diverge because both sides shrink together. Still
  worse: 1.99e-2 vs SPSA's 6.10e-3, over 8 seeds.

Consistent with Result 4: many cheap noisy iterations lose to few precise ones
in this regime. The shot-frugal literature's gains may need larger parameter
counts or looser budgets than tested here. **Not pursuing further until the
`ansatz` and `budget` regimes below are settled.**

### Two bugs found in my own implementations (recorded so they stay fixed)

1. **iCANS self-limiting shot rule.** I had written the denominator as
   `(χ² + b + ξ/s)`. Because `s` appears on both sides, it pins itself to the
   floor: measured 535 iterations *all* at the 8-shot minimum, energy bouncing
   over a 0.5 Ha range, pure random walk. The correct rule has only `(χ² + b)`.

2. **Gain schedules anchored to a sentinel `maxiter`.** Under a shot budget the
   real stopping point is `budget/(2·shots)`, but SPSA's `A = 0.1·maxiter` was
   being anchored to the `maxiter=10000` sentinel the budget-limited driver
   passes. That freezes the step size at a near-constant tiny value for the
   whole run. This is a trap for *any* budget-matched comparison of optimisers
   with decaying schedules, and it looks exactly like a shot-allocation result.
   Fixed via `estimate_horizon()`.

### Result 5 — the standard hardware-efficient ansatz destroys the reference state

The single largest effect found this session, and it is a correctness bug in
standard practice rather than a tuning question.

A textbook hardware-efficient ansatz applies **fixed** CX entangler layers. They
act regardless of the parameters, so at `θ = 0` the circuit is *not* the initial
state. Measured on H₄:

| ansatz | overlap with Hartree-Fock at θ=0 | energy at θ=0 |
|---|---|---|
| `hea:2` (fixed CX) | **0.0000** | −1.1462 Ha |
| `hea:2:linear:cry` | **1.0000** | −5.1608 Ha (= HF exactly) |

So the standard advice — "initialise near zero so you start at Hartree-Fock" —
is **false** for the standard ansatz. The optimiser starts on a state orthogonal
to the reference, 4 Ha from the answer, and never recovers within a realistic
budget. Replacing CX with controlled rotations (identity at θ=0) fixes it:

| ansatz | env | median error (4 seeds) |
|---|---|---|
| `hea:2` | ideal | 1.609 Ha |
| `hea:2` | fake_kolkata | 2.039 Ha |
| **`hea:2:linear:cry`** | ideal | **0.0497 Ha** |
| **`hea:2:linear:cry`** | fake_kolkata | **0.0481 Ha** |

**32× error reduction**, for 10 extra parameters and no extra circuit depth
beyond the CX pair each controlled rotation compiles to.

Secondary but important: once the start is correct, **device noise costs almost
nothing here** (0.0481 under fake_kolkata vs 0.0497 ideal). The failure that
looked like a noise problem was never a noise problem. That is a reminder to
always check the ideal-simulator result before attributing an error to noise.

Caveat that keeps this honest: 0.048 Ha is still 30× worse than chemical
accuracy, and H₄'s correlation energy is only 0.0418 Ha — so at this budget we
have recovered essentially *none* of the correlation energy and are sitting at
roughly the Hartree-Fock level. The ansatz is no longer the bottleneck; the
budget and the optimiser are.

### Result 6 — Aer's cost is per-circuit, not per-shot (research-loop speedup)

A 6-qubit noisy circuit costs ~0.45 s per `run()` call at *both* 1k and 4k
shots. Adaptive allocation gives every group a distinct shot count, so naive
batching means one call per group — a ~10× simulation slowdown for exactly the
configuration most worth studying.

Fix: run all circuits once in a single batch at `max(plan)` shots and draw each
group's allotment from the returned counts without replacement. A subset of iid
multinomial draws is distributed exactly as that many iid draws, so this is
statistically identical; only surplus *simulator* effort is spent, and the
ledger still charges the planned shots.

Validated against direct sampling over 4000 trials: mean difference 6.2e-4
against a 2σ band of 1.9e-3, standard-deviation ratio 0.9959. Measured effect on
an H₄ device-noise run: **>600 s → 29.8 s (>20×)**.

### Result 7 — covariance-aware grouping, and a reproducibility failure that had to be fixed first

New idea, implemented this session. Standard grouping minimises the *number* of
groups. Under Neyman allocation the quantity that actually sets the shot cost is

```
total variance = (Σ_g √Var_g)²,   Var_g = Σ_{i,j∈g} c_i c_j Cov(P_i, P_j)
```

The cross terms mean count and variance can disagree: two **anticorrelated**
observables in one group make it cheaper to estimate than either term alone
suggests. So group to minimise variance, not count.

**Everything below is the second set of numbers.** The first set was not
reproducible, and the story of why is in the next section — it is the more
useful result of the two.

Scored against the covariance of the exact ground state (where a converged VQE
sits). Ratio < 1 is better; `control` is the group-count control:

| molecule | terms | grouping | groups | variance | ratio | control |
|---|---|---|---|---|---|---|
| H₂ | 5 | count-minimising | 2 | 0.1245 | 1.000 | — |
| H₂ | | cov-aware / HF | 2 | 0.1245 | 1.000 | 1.000 |
| H₂ | | cov-aware / oracle | 2 | 0.1245 | 1.000 | 1.000 |
| H₄ | 165 | count-minimising | 10 | 1.1680 | 1.000 | — |
| H₄ | | cov-aware / HF | 11 | 1.1293 | 0.967 | 1.089 |
| H₄ | | cov-aware / oracle | 11 | 1.1293 | 0.967 | 1.089 |
| LiH | 631 | count-minimising | 41 | 0.6909 | 1.000 | — |
| LiH | | **cov-aware / HF** | 44 | 0.5378 | **0.778** | 1.238 |
| LiH | | cov-aware / oracle | 45 | 0.5830 | 0.844 | 1.389 |

**On LiH — the largest system tested — a 22% variance reduction, worth 1.29×
fewer shots for equal accuracy, using only a free Hartree-Fock reference.**
Marginal on H₄ (3%), nothing on H₂. The benefit grows with term count, which is
the right direction: the method needs enough terms per group for the cross terms
to matter.

Two things make this trustworthy rather than suggestive:

* **The group-count control.** Covariance-aware grouping produces *more* groups
  than count-minimising, and more groups also means finer-grained Neyman
  allocation — so the win could have been an artefact of count alone. Control:
  split the count-minimising grouping at random to the same group count. It
  scores 1.089–1.389 — *worse* than baseline in every case, as it should be,
  since splitting destroys the beneficial anticorrelations. The effect is the
  covariance structure, not the count.
* **The variance model predicts measured sampling variance.** At the exact H₄
  ground state, 150 independent 20k-shot estimates: predicted σ 0.00715 vs
  measured 0.00705 for cov-aware; predicted 0.00764 vs measured 0.00794 for the
  baseline. Both estimators unbiased to within their standard error.

**The greedy search, not the covariance quality, is the bottleneck.** The
Hartree-Fock reference (0.778) reproducibly *beats* the exact-ground-state
oracle (0.844) on LiH. Perfect covariance information does not give the best
partition, so the limiting factor is the greedy placement, not what it is fed.
That redirects the next step: improve the search (local refinement, annealing on
the partition), not the reference state.

### Result 8 — the results above were not reproducible, three times over

Worth its own entry because it invalidated a full round of measurements and the
cause is invisible in any single run.

The same computation, in three separate processes, returned **11, 11 and 13
groups with variance ratios of 0.809, 0.927 and 0.795**. Within a process it was
perfectly deterministic, which is what made it hard to see.

Root cause, confirmed by hashing the coefficient array across processes:
**PySCF returns different Hamiltonian coefficients every run.** Three identical
`build_molecule("H4")` calls gave three different SHA-256 hashes, with
coefficients differing in the 15th significant digit — multithreaded BLAS in the
integral transform, whose summation order varies between runs.

That difference is ~12 orders of magnitude below chemical accuracy and
physically meaningless. It is not meaningless to a greedy algorithm: 2 of 164
placements in the H₄ grouping had candidate costs agreeing to within 1e-12, and
those flipped, changing the whole downstream partition.

Fixed in three places:

1. Molecular Hamiltonians are canonicalised on construction — coefficients
   rounded to 12 decimals, terms sorted by Pauli label.
2. The greedy ordering rounds |coefficient| and breaks ties on the label.
3. The covariance greedy's cost comparison rounds the delta and breaks ties on
   the group index; a raw float comparison there was the second flip point.

Verified: identical coefficient hash, identical group counts and identical
variance ratio to 6 decimals across three fresh processes.

**Generalised lesson, now a standing rule:** any greedy or sorting-based step
in this repository must have an explicit tie-break, and any result that depends
on one must be checked across processes, not just across calls. Single-process
determinism proves nothing.

### Result 9 — local refinement of the partition is the bigger lever

Direct follow-up on Result 7's diagnosis that the greedy placement, not the
covariance quality, was the bottleneck. Added a steepest-descent local search:
repeatedly move the single term whose relocation most reduces total √variance,
until no move helps. Compatibility is enforced, so every group stays
simultaneously measurable.

Final reproducible numbers (experiment 003, commuting grouping, scored against
the exact-ground-state covariance; `control` = random split to the same group
count):

| molecule | terms | grouping | groups | variance | ratio | control |
|---|---|---|---|---|---|---|
| H₂ | 5 | count-minimising | 2 | 0.1245 | 1.000 | — |
| H₂ | | all cov-aware variants | 2 | 0.1245 | 1.000 | 1.000 |
| H₄ | 165 | count-minimising | 10 | 1.1680 | 1.000 | — |
| H₄ | | cov-aware / HF | 11 | 1.1293 | 0.967 | 1.089 |
| H₄ | | cov-aware / oracle | 11 | 1.1293 | 0.967 | 1.089 |
| H₄ | | **cov-aware / HF + refined** | 11 | 0.8269 | **0.708** | 1.089 |
| LiH | 631 | count-minimising | 41 | 0.6909 | 1.000 | — |
| LiH | | cov-aware / HF | 44 | 0.5378 | 0.778 | 1.238 |
| LiH | | cov-aware / oracle | 45 | 0.5830 | 0.844 | 1.389 |
| LiH | | **cov-aware / HF + refined** | 44 | 0.4366 | **0.632** | 1.238 |

**29% (H₄) and 37% (LiH) variance reduction, worth 1.41× and 1.58× fewer shots
for equal accuracy** — using only a free Hartree-Fock covariance reference. Both
converge to a local optimum of single-term moves (19 moves / 2 s on H₄, 82 moves
/ 85 s on LiH), which is negligible against the shots saved.

Refinement matters more than the covariance-greedy itself: refining the *plain
count-minimising* partition already reaches 0.858 (H₄) and 0.700 (LiH). The
greedy start still helps, but the search is where the value is — exactly what
Result 7's "oracle loses to Hartree-Fock" signal implied.

**Validated empirically, not just predicted.** 300 independent 20k-shot
estimates at the exact H₄ ground state:

| grouping | predicted σ | measured σ | bias (sem 4e-4) |
|---|---|---|---|
| count-minimising | 0.007642 | 0.007858 | −0.000054 |
| cov-aware + refined | 0.006430 | 0.006241 | −0.000073 |

Model accurate to ~3%, both estimators unbiased, and the *measured* variance
ratio (0.631) slightly beats the predicted one (0.708).

Caveats kept in view: H₂ shows nothing (2 groups either way), the benefit grows
with term count, and this is still a fixed-state measurement — the grouping is
**not yet wired into `ShotEstimator`**, so no end-to-end VQE run has used it.

### Result 10 — the hardware-efficient ansatz is stuck at Hartree-Fock *by construction*

This supersedes the optimistic reading of Result 5 and explains every flat H₄
measurement in this session.

Experiment 001 repeated on H₄ (8 seeds, 300k budget, ideal simulator,
`hea:2:linear:cry`) showed the allocation effect **vanishing**: every
configuration landed at 5.0–5.9e-2 Ha, ratios 0.90–1.01 against the baseline,
nothing significant (p ≥ 0.29). On H₂ the same comparison gave 0.264.

Chasing that down:

1. **Budget is not the limit.** 150k and 600k shot budgets give bit-identical
   errors (5.1476e-2). COBYLA spends the whole budget (49 and 196 evaluations)
   and gets nowhere.
2. **Shot noise is not the limit.** With exact noiseless energies and up to
   10 000 evaluations, COBYLA converges to **4.182e-2 Ha** — and H₄'s
   correlation energy is **0.04182 Ha**. The optimiser lands on Hartree-Fock and
   recovers *exactly zero* correlation energy.
3. **The start is an exact stationary point.** Finite-difference gradient of
   `hea:2:linear:cry` at θ = 0: `|∇| = 0.00e+00`, with **0 of 46** parameters
   having a non-zero derivative.
4. **Depth does not fix it.** `hea:6:linear:cry`, 114 parameters, 30 two-qubit
   gates: `|∇| = 0.00e+00`, 0 of 114, BFGS from zero converges to 4.182e-2 —
   Hartree-Fock again.
5. **Larger initial perturbations make it worse**, not better: median error over
   5 seeds goes 4.182e-2 (scale 0) → 4.225e-2 (0.2) → 4.993e-2 (1.0).

**Why**, and this is the useful part: the first-order response of a
real-amplitude ansatz at the Hartree-Fock determinant connects only to *single*
excitations, and Brillouin's theorem says those have zero coupling to HF.
Correlation energy lives in *double* excitations, which such an ansatz reaches
only at second order. So the gradient vanishes identically, at any depth.

The contrast makes it unambiguous:

| ansatz | params | 2q gates | \|∇\| at θ=0 | non-zero grads | best exact error |
|---|---|---|---|---|---|
| **UCCSD** | 26 | **1096** | **0.551** | 14/26 | **9.7e-06** ✓ chemical accuracy |
| hea:2:linear:cry | 46 | 10 | 0.000 | 0/46 | 4.182e-2 (= HF) |
| hea:6:linear:cry | 114 | 30 | 0.000 | 0/114 | 4.182e-2 (= HF) |

UCCSD's generators *are* double excitations, so it has gradient at HF and
reaches chemical accuracy. It also has **110× more two-qubit gates**, which at
realistic error rates is fatal.

Even given a good start, the hardware-efficient ansatz is under-expressive here:
BFGS from large random starts reaches only 2.155e-2 on H₄ — still 13× worse than
chemical accuracy.

**Consequences for this project, which are large:**

* Result 5's "32× improvement" from reference-preserving entanglers is real
  *relative to the fixed-CX ansatz*, but it buys a graceful failure, not a
  solution: the run now lands exactly on Hartree-Fock instead of on garbage.
  Zero correlation energy either way.
* Every H₄ measurement of allocation, grouping and optimiser in this session was
  taken in a regime where the optimiser could not move. Those H₄ numbers say
  nothing about the methods, only about the ansatz. The H₂ results (Results 3
  and 4) stand — H₂ converges.
* The bottleneck is not measurement. **It is the ansatz.** Measurement-side
  work — which is most of this session — only pays off once a run is actually
  shot-noise-limited, and on H₄ nothing was.

The real gap this opens: UCCSD-like coupling to double excitations at
hardware-efficient depth. That is the next thing to build (candidates in
NEXT_STEPS: quantum-number-preserving gate fabrics, Givens-rotation networks,
k-UpCCGSD).

### Result 11 — grouping still pays for wall-clock and hardware time

Independent of the above, from the same H₄ run (identical shot counts by
construction):

| grouping | groups | hardware seconds | simulator wall seconds |
|---|---|---|---|
| none | 164 | 92.6 | 63.4 |
| qwc | 37 | 80.1 | 17.9 |
| commuting | 10 | 78.0 | 6.5 |

**10× faster to simulate and 16% less hardware time**, at identical shots. The
hardware-time gain is modest because the 250 µs reset delay dominates the cost
model; the simulation gain is what makes the research loop usable.

### Result 12 — the ansatz gap, quantified on real device calibration

Given Result 10 (hardware-efficient ansätze cannot leave Hartree-Fock), the
question is what the alternatives actually cost on hardware. Transpiled to real
device targets, with fidelity estimated as `(1 - e_2q)^n_2q` times a T1 decay
term over the circuit's real duration:

**fake_kolkata** (Eagle era, median 2q error 0.75%, median T1 111 µs):

| ansatz | 2q gates | duration | total fidelity |
|---|---|---|---|
| hea:2:linear:cry | 20 | 6.4 µs | **0.61** |
| puccd | 315 | 101.7 µs | 0.0004 |
| succd | 861 | 277.9 µs | 0.0000 |
| uccsd | 1342 | 430.8 µs | 0.0000 |

**fake_torino** (Heron, median 2q error 0.42%, median T1 185 µs):

| ansatz | 2q gates | duration | total fidelity |
|---|---|---|---|
| hea:2:linear:cry | 20 | 1.9 µs | **0.87** |
| puccd | 317 | 30.3 µs | **0.099** |
| succd | 867 | 82.2 µs | 0.0018 |
| uccsd | 1350 | 127.6 µs | 0.0001 |

So the situation is a clean scissor:

* The only ansatz that survives the device (HEA, fidelity 0.61–0.87) **cannot
  move off Hartree-Fock** — zero gradient, at any depth.
* The cheapest ansatz that *can* move (PUCCD) retains **10% signal** on the best
  simulated device and is still under-expressive: 1.6e-2 Ha, 10× worse than
  chemical accuracy.
* Everything expressive enough to reach chemical accuracy (UCCSD, 9.7e-6)
  retains ~0.01% — nothing at all.

**Target, now quantified:** an ansatz that couples to double excitations within
**≲ 50 two-qubit gates**. At Heron's 0.42% that is fidelity ≈ 0.81, comparable
to the hardware-efficient ansatz.

### Result 13 — Qiskit's UCC compilation is ~6× off the achievable gate count

The target above looks less remote after counting what PUCCD actually needs.

On H4 (2 occupied, 2 virtual spatial orbitals), PUCCD has **4 parameters** — four
paired double excitations. A double-excitation rotation (the 4-qubit Givens
rotation mixing |1100⟩ and |0011⟩) has a known decomposition in **~13 CNOTs**.
So 4 × 13 ≈ **52 two-qubit gates** should suffice.

Qiskit's `UCC` produces **317**.

The reason is the compilation strategy: qiskit-nature builds each excitation as
a Trotterised product of Pauli-string exponentials, and each of the eight Pauli
strings making up one double excitation is compiled to its own CNOT ladder. The
ladders share structure that the generic transpiler does not exploit.

**Predicted payoff of compiling double excitations directly as Givens
rotations:** 317 → ~52 two-qubit gates on H4, which moves Heron fidelity from
0.099 to roughly 0.80 — from unusable to comparable with the hardware-efficient
ansatz, while *keeping* the non-zero gradient at Hartree-Fock that the
hardware-efficient ansatz lacks.

`[unverified]` — this is an estimate from gate counting, not a measurement. It is
the single most promising concrete lead out of this session and the first thing
to build next.

### Result 14 — the double-excitation gate, built and verified: 26 gates against ~79

Acting on Result 13's lead. `qres/fermionic.py` compiles the double excitation
directly as a two-level Givens rotation instead of as eight Pauli-string
exponentials.

Construction: three CNOTs relabel the basis so that |0011> and |1100> differ in
a single qubit — `CX(0→1), CX(2→3), CX(0→2)` sends them to |0101> and |0100> —
a multi-controlled RY acts there, and the CNOTs are undone. The three controls
are necessary and sufficient: solving `d^c=0, c^a=1, b^a=0` gives exactly those
two states and no others, so the gate cannot leak outside the intended subspace.

| construction | two-qubit gates | correct? |
|---|---|---|
| qiskit-nature (Trotterised Paulis) | ~79 per excitation | yes |
| two `mcx` around half-angle RYs | 34 | yes |
| **Qiskit `mcry` synthesis** | **26** | **yes** |
| relative-phase `rccx` chain | 14 | **no** — error 0.15–1.0 |

**3× fewer two-qubit gates than the Trotterised compilation**, verified as an
operator against the exact target matrix to 1e-12 over seven angles. Not the
~13 the literature's hand-optimised decomposition reaches, but most of the way
there and it is exact.

The `rccx` variant is instructive and is kept in the test suite as a regression
guard. It is 14 gates — better than the target — and wrong: the relative phases
do not cancel between the two occurrences. The error scales with the angle
(0.15 at θ=0.3, 1.0 at θ=π), so at small angles it looks nearly right, which is
exactly how a wrong excitation gate would get adopted.

Properties verified, not assumed: exact operator match, particle-number
preservation (no amplitude leaks between Hamming-weight sectors), action
confined to the two-dimensional subspace, the group law `G(a)G(b) = G(a+b)`, and
**non-zero derivative at the reference state** — the property the
hardware-efficient ansatz lacks and the whole reason for the module.

Still to do before this pays off: assembling these gates into a full ansatz
needs the Jordan-Wigner Z-strings between non-adjacent orbitals, which the bare
4-qubit gate does not carry. That is the next step, and until it is done the 3×
is a per-gate figure, not an end-to-end one.

### Result 15 — a shallow PUCCD that is exactly the same operator, and where its advantage goes

`qres/fermionic.py` now assembles the verified Givens gates into a full PUCCD.
It reproduces `qiskit_nature`'s `PUCCD` **as a unitary** at random parameters,
to 8.4e-13 on H₄ and 4.9e-14 on H₂ — the same operator, not merely a similar
ansatz.

Getting there required a correction worth recording. A first version handled no
Jordan-Wigner Z-strings and **matched H₂ exactly** — orbitals 0 and 1 are
adjacent, so the string is empty by accident — while being off by 1.38 on H₄,
where α₁ lies between the occupied and virtual indices. Testing on the smaller
molecule alone would have shipped a wrong ansatz with a passing test.

The string is cheap to handle: the excitation operator is `G ⊗ Z_string`, so the
rotation runs backwards on the odd-parity branch, and conjugating the rotation
target by a CNOT from each string qubit does exactly that (`X RY(t) X = RY(-t)`).
Two CNOTs per string qubit, no extra controls.

**Where the advantage goes**, on H₄:

| circuit | abstract 2q | routed on Heron | routing penalty |
|---|---|---|---|
| qiskit-nature PUCCD | 277 | 315 | 1.14× |
| **this module** | **114** | 240 | **2.11×** |
| advantage | **2.43×** | 1.31× | |

The compilation is 2.43× cheaper in gate count and keeps only **1.31×** after
routing. The reason is locality, not compilation: qiskit-nature's Pauli ladders
are CNOT chains between *adjacent* qubits in the Jordan-Wigner ordering, which
map onto a heavy-hex device almost for free, while the 4-qubit Givens gates act
on qubits `(i, M+i, a, M+a)` — scattered across the blocked-spin ordering — and
the transpiler pays in SWAPs.

**The fix is the spin-orbital ordering, and it is not free.** Interleaving
(`α₀ β₀ α₁ β₁ …`) would put each excitation's four qubits in two adjacent pairs.
But interleaving changes the Jordan-Wigner *algebra*, not just the layout: the
spin partner then lies *inside* the parity string, so the strings must be
re-derived and re-verified against a reference that uses the same ordering.
Attempted and abandoned this session — a naive relabelling produced duplicate
qubit indices, which is the symptom of exactly that problem. Next session's
work.

Honest summary of the lead: **2.43× in gate count, 1.31× on hardware today**, and
a plausible route to recovering most of the gap through qubit ordering. Not the
6× the gate-counting estimate in Result 13 suggested; that estimate ignored
routing entirely.

### Result 16 — the gap, measured across the whole excitation family: ~32×

Experiment 005 screens every candidate on the three things that must hold at
once. H4, `fake_torino` (Heron), exact optimisation for the accuracy column:

| ansatz | params | 2q | \|grad@0\| | exact error | chem? | fidelity | verdict |
|---|---|---|---|---|---|---|---|
| hea:2:linear:cry | 46 | 20 | **0.000** | 3.67e-2 | no | 0.865 | stuck |
| hea:4:linear:cry | 80 | 40 | **0.000** | 1.72e-2 | no | 0.767 | stuck |
| puccd | 4 | 317 | 0.446 | 1.63e-2 | no | 0.099 | inexact |
| puccd:2 | 8 | 638 | 0.630 | **1.63e-2** | no | 0.0095 | inexact |
| puccd:3 | 12 | 959 | 0.772 | **1.63e-2** | no | 0.0009 | inexact |
| puccsd | 12 | 460 | 0.446 | 1.59e-2 | no | 0.038 | inexact |
| succd | 10 | 867 | 0.489 | 9.02e-3 | no | 0.0018 | inexact |
| succd:2 | 20 | 1738 | 0.691 | 2.23e-3 | no | 0.0000 | inexact |
| **ucc-d:1** | 18 | 1308 | 0.551 | **1.08e-4** | **YES** | 0.0001 | noisy |

**Nothing passes.** The requirement is a scissor with a measured width:

* reaching chemical accuracy needs **~1300 two-qubit gates**
* surviving the device (fidelity > 0.5) allows **~40**
* **the gap is ~32×**

Two sub-results that sharpen it:

**Repeating paired doubles is useless.** `puccd`, `puccd:2` and `puccd:3` give
*identical* error — 1.631e-2 at k = 1, 2 and 3 — while tripling the gate count
and dropping fidelity from 0.099 to 0.0009. The paired-doubles manifold is
closed under composition, so extra repetitions add parameters that reach no new
states. This kills "add k-fold repetitions" as a route to expressibility for
paired doubles specifically.

**Singles contribute essentially nothing**, as Brillouin's theorem predicts:
`puccsd` (12 parameters, singles + paired doubles) reaches 1.589e-2 against
`puccd`'s (4 parameters, doubles only) 1.631e-2 — a 2.6% improvement for three
times the parameters. And `ucc-d:1`, doubles only, reaches chemical accuracy
with 18 parameters where full UCCSD needs 26 for 9.7e-6. **Doubles are where
essentially all of the correlation energy is**, which is exactly why the
hardware-efficient ansatz — whose first-order response reaches only singles —
cannot move.

Against that 32× gap, this session's compilation work (Result 15) buys 2.43×
abstract, 1.31× routed. Useful, and nowhere near sufficient on its own.

### Result 17 — evaluations-per-parameter is the binding constraint everywhere

Repeating the end-to-end measurement-grouping test with UCCSD — chosen because
it *does* have gradient at Hartree-Fock and therefore converges — at a 200k shot
budget gave **1.39e-1 Ha**, worse than the stuck hardware-efficient ansatz's
5e-2.

Not a contradiction: 200 000 shots at 4096 per evaluation is **48 evaluations
for 26 parameters**, and COBYLA spends 27 of them constructing its initial
simplex. The optimiser barely starts, and with UCCSD's large gradient it
overshoots when it does.

This is the same constraint behind every flat result in this session, stated
once properly:

    usable budget  ≳  10 × n_parameters × shots_per_evaluation

For H4/UCCSD at 4096 shots that is ≈ 1.07M shots minimum; 200k is 5× short. Any
future budget-matched comparison has to be sized against the parameter count, or
it measures the simplex construction rather than the algorithm.

### Result 18 — the headline allocation result does not replicate at 24 seeds

Re-ran experiment 001 on H₂ with **24 seeds** instead of 4, same budget, same
everything else. Adaptive Neyman allocation against uniform:

| | 4 seeds (Result 3) | 24 seeds |
|---|---|---|
| median ratio | **0.264** | **0.776** |
| wins / losses | 4 / 0 | 14 / 10 |
| sign-test p | 0.125 (the floor at n=4) | **0.541** |

The effect is real in the median — 4.19e-3 against 8.25e-3, a ~1.7× shot saving
— but it is **not statistically significant**, and the interquartile ranges
overlap heavily ([1.9e-3, 7.0e-3] against [2.9e-3, 1.7e-2]). The 3.8× was a
small-sample artefact: with 4 seeds, 4/0 wins is the best obtainable outcome and
carries almost no evidence.

**Honest restatement:** variance-adaptive allocation gives a median error ratio
of ~0.78 on H₂ (≈ 1.7× fewer shots for equal accuracy), suggestive but not
significant at 24 seeds. It is not the 14× the first entry claimed.

One result *is* significant, and it is negative: ungrouped measurement with
coefficient-weighted allocation is reliably **worse** than the baseline —
ratio 1.271, 5/19, **p = 0.007**.

I set "multiple seeds, median + IQR, paired sign test" as a standing rule at the
start of this session and then reported a 4-seed result as a headline anyway.
The rule now has a number attached: **for VQE error comparisons, 4 seeds is not
enough to distinguish a 4× effect from nothing.**

### Result 19 — reference preservation and non-zero gradient are in direct conflict

Chasing Result 18 turned up why the same experiment gave 2.3e-3 in the first
session and 2.1e-2 later: **the ansatz had changed**, and the "fixed" version is
worse on H₂.

Gradient at θ = 0 on H₂, where both ansätze start *exactly* on the Hartree-Fock
determinant (overlap 1.0000):

| ansatz | overlap with HF | \|∇\| at θ=0 | non-zero | converged error |
|---|---|---|---|---|
| `hea:2` (fixed CX) | 1.0000 | **0.181** | 1/12 | **4.2e-3** |
| `hea:2:linear:cry` | 1.0000 | **0.000** | 0/14 | 2.1e-2 (= E_corr, stuck) |

**The mechanism**: fixed entangling gates act whatever the parameters are, so
the later rotation layers see an *entangled* state and acquire a non-zero
derivative. Controlled-rotation entanglers are the identity at θ=0 — which is
exactly how they preserve the reference — so every layer sees a computational
basis state and Brillouin's theorem zeroes every gradient.

**The property that preserves the reference is the property that kills the
gradient.** They cannot both come from the entangler.

This corrects Result 5, which reported the controlled-rotation ansatz as a "32×
improvement". That is true on H₄, where fixed CX *destroys* the reference
(overlap 0.0000) and lands on garbage. It is false on H₂, where fixed CX
preserves the reference *and* has gradient, and the controlled version is 5×
worse. Neither ansatz is right; the requirement is both properties at once, and
neither construction delivers them.

Plausible resolution, untested: initialise the *entangler* parameters away from
zero while leaving the rotation parameters at zero — entanglement enough for a
gradient, while staying near the reference. `[unverified]`

### Result 20 — the stationarity fix works, and does not help where it was needed

Result 19 left a candidate fix: offset only the *entangler* parameters from
zero, buying a gradient while staying near the reference. Tested.

**It works, under noiseless gradient-based optimisation.** BFGS from the offset
start:

| entangler offset | H₂ overlap w/ HF | \|∇\| at start | H₂ error | H₄ error |
|---|---|---|---|---|
| 0.0 | 1.0000 | **0.000** | 2.03e-2 (= E_corr) | 4.18e-2 (= E_corr) |
| **0.1** | 0.9988 | 0.092 | **9.1e-11** | 4.18e-2 |
| **0.5** | 0.9689 | 0.435 | **2.5e-9** | **2.01e-2** |
| π/2 | 0.7071 | 0.865 | 2.03e-2 | 3.52e-2 |
| π | 0.0000 | **0.000** | 7.9e-1 | 3.27 |

H₂ goes from stuck at Hartree-Fock to **chemical accuracy by seven orders of
magnitude**, from an offset of 0.1. H₄ halves its error. The diagnosis is
confirmed exactly: 0 *and* π are both stationary (a controlled rotation is
identity at 0 and a Clifford at π), and anything interior is not.

**It does not help under shot noise with a gradient-free optimiser.** Same
ansätze through the actual VQE driver, COBYLA, 200k budget, 6 seeds, ideal
simulator:

| ansatz | H₂ median | H₂ ≤ chem acc | H₄ median |
|---|---|---|---|
| `hea:2` (fixed CX, no offset) | **3.34e-3** | 1/6 | 1.53 |
| `hea:2:linear:cry` (offset 0.5) | 2.36e-2 | 0/6 | 0.248 |

The offset *hurts* on H₂ — 2.4e-2 against 3.3e-3. COBYLA cannot exploit the
gradient the offset buys; it only pays for starting further from the reference
(overlap 0.97 rather than 1.00) at a worse energy.

**So the conflict is relocated, not solved.** The ingredients now exist —
a start that both preserves the reference and has a gradient — but extracting
value from a gradient needs an optimiser that uses gradients, and every
gradient-based method tested this session (iCANS, parameter-shift) is far too
expensive in circuit evaluations to be affordable. Default left at 0.0;
the parameter is there for gradient-based use.

That is a fair summary of the whole session's arc: each fix exposes that the
next thing along was the real constraint.

### Result 21 — SPSA uses the offset, but not enough to win

The cheapest candidate from Result 20's open item, tested: SPSA already
estimates a gradient in two evaluations per step, so it should be able to use
the offset start that COBYLA cannot. H₂, 200k budget, **12 seeds**, ideal
simulator:

| ansatz | offset | optimiser | median error | best | ≤ chem acc |
|---|---|---|---|---|---|
| `hea:2` (fixed CX) | 0.0 | COBYLA | 7.21e-3 | 1.47e-3 | 1/12 |
| **`hea:2` (fixed CX)** | **0.0** | **SPSA** | **4.46e-3** | 2.61e-3 | 0/12 |
| `hea:2:linear:cry` | 0.0 | COBYLA | 2.17e-2 | 2.04e-2 | 0/12 |
| `hea:2:linear:cry` | 0.0 | SPSA | 2.10e-2 | 3.43e-3 | 0/12 |
| `hea:2:linear:cry` | 0.3 | SPSA | 1.43e-2 | 9.57e-3 | 0/12 |
| `hea:2:linear:cry` | 0.5 | SPSA | 1.36e-2 | 4.23e-3 | 0/12 |

**Half-confirmed.** SPSA *does* extract value from the offset where COBYLA could
not: 2.10e-2 → 1.36e-2, a 1.5× improvement, and monotone in the offset. So the
mechanism is right.

But it does not clear the bar. The best configuration remains the plain
fixed-CX ansatz with SPSA at 4.46e-3, and the acceptance test set in NEXT_STEPS
(beat 3.34e-3 median, chemical accuracy in more than 1/6 of runs) is **not
met** by any combination tested. Nothing reaches chemical accuracy reliably on
H₂ — the best single run out of 96 was 1.47e-3, and the best rate 1/12.

The remaining candidates from that item — Rosalin's random operator sampling
and quantum natural gradient — are untested.

---

## Session 1 — closing state

**What is solid and reproducible:**

* general-commuting grouping: 15–20× fewer circuits than ungrouped (Result 1)
* covariance-aware grouping with local refinement: 0.63 variance ratio on LiH,
  predicted variance confirmed against 300 sampled estimates, group-count
  confound controlled for (Results 7, 9)
* direct Givens compilation of double excitations: 2.43× fewer two-qubit gates,
  verified to be the *same operator* as qiskit-nature's to 8e-13 (Results 14, 15)
* an interior optimum in shots-per-evaluation, with an analytic reason (Result 4)
* 81 tests, every piece of measurement machinery checked algebraically

**What was retracted within the session:**

* the 3.8× allocation win → 1.3× and not significant at 24 seeds (Results 3, 18)
* "32× from reference-preserving entanglers" → true on H₄, false on H₂, and
  neither ansatz is right (Results 5, 19)

**The one thing that would unblock the rest:** an optimiser that can use a
gradient at a cost the shot budget can bear. Every measurement-side result in
this repository is worth 1.3–1.6×; the ansatz/optimiser gap is 32×.

### Open question carried forward

Scaling `fake_kolkata`'s gate errors to 0.5× / 0.25× / 0.1× left the energy
error at the reference state essentially unchanged (0.089 / 0.084 / 0.089 Ha),
i.e. it saturates. Disabling thermal relaxation moved it only slightly
(−1.2570 → −1.2473), so T1/T2 is *not* the explanation. Unresolved — most
likely readout error, or the scaling not reaching the channel it should.
`[unverified]` Needs a controlled ablation before any claim is made about where
the noise floor comes from.

---

---

## Session 2 — 2026-08-25 (10:24 UTC)

### Result 22 — shot noise was never being resampled, and it changed the answers

Found while trying to separate the shot-noise floor from the optimiser limit:
`NoiseEnvironment.run` passed a **fixed** `seed_simulator` on every call, so Aer
returned the *identical* counts for the same circuit every time.

```
raw env.run twice gives identical counts: True
```

Repeated estimates at the same parameters differed only because the adaptive
allocation drifted, not because any noise was redrawn. The optimiser was
therefore facing a **frozen rough landscape**, not a stochastic one — and could
fit itself to one particular noise realisation instead of averaging over it.
This looks entirely normal from the outside and silently distorted every
optimisation result in session 1.

Fixed by advancing a per-environment run counter, which keeps runs reproducible
(the seed sequence is fixed by the base seed) while making each draw
independent. Verified: the measured sampling error is now ~6× larger than the
apparent variation before, and scales correctly as 1/√N —

| shots | measured σ | bias / sem |
|---|---|---|
| 512 | 0.0177 | 0.50 |
| 2 048 | 0.0076 | 2.23 |
| 20 000 | 0.0024 | −1.88 |
| 200 000 | 0.00076 | −1.83 |

— unbiased, and 4× the shots gives 2× the precision as it must. Regression-tested
on both ideal and noisy backends.

### Result 23 — with real shot noise, SPSA beats COBYLA by 2× (p = 0.001)

Re-ran the optimiser comparison with the fix. H₂, ideal simulator, 200k budget,
**16 seeds**, paired against `hea:2` + COBYLA:

| ansatz | optimiser | median | IQR | ratio | W/L | p |
|---|---|---|---|---|---|---|
| `hea:2` | COBYLA | 1.84e-2 | [9.0e-3, 2.3e-2] | 1.000 | — | — |
| **`hea:2`** | **SPSA** | **5.80e-3** | [4.5e-3, 8.9e-3] | **0.485** | **15/1** | **0.001** |
| `hea:2` | SPS k=1, 512 shots | 1.04e-2 | [5.5e-3, 1.7e-2] | 0.642 | 10/6 | 0.454 |
| `hea:2` | SPS k=2, 512 shots | 8.74e-3 | [5.0e-3, 1.9e-2] | 0.771 | 9/7 | 0.804 |
| `hea:2:linear:cry` | COBYLA | 2.84e-2 | [2.4e-2, 3.3e-2] | 1.685 | 1/15 | **0.001** |
| `hea:2:linear:cry` | SPSA | 1.07e-2 | [9.0e-3, 1.2e-2] | 0.641 | 12/4 | 0.077 |

**The first statistically significant positive result in this repository:
SPSA halves COBYLA's error, 15 wins out of 16, p = 0.001.**

And it only appears once the noise is real. In session 1 COBYLA measured 7.21e-3
against SPSA's 4.46e-3 — nearly competitive. With resampled noise COBYLA
degrades to 1.84e-2 while SPSA barely moves (5.80e-3). **COBYLA's apparent
competence was an artefact of the frozen landscape**: it builds a local
quadratic model, which is exactly the thing that works on a deterministic rough
function and fails on a genuinely stochastic one. SPSA is designed for the
stochastic case and is unaffected.

Also now significant, and negative: the reference-preserving `cry` ansatz is
**worse** than fixed CX under COBYLA (ratio 1.685, 1/15, p = 0.001), consistent
with Result 19's stationary-start diagnosis.

### Result 24 — stochastic parameter-shift solves the step-count problem and does not help

Built to address the arithmetic that killed gradient methods in session 1: a
full parameter-shift gradient costs `2n` evaluations, so H₂ at 2048 shots and a
200k budget affords **four** gradient steps. `stochastic_parameter_shift`
updates `k` randomly chosen coordinates per step at `2k` evaluations, with Adam
moments kept per-coordinate and bias-corrected by each coordinate's own visit
count.

It does what it was designed to do — **195 to 781 gradient steps** instead of
four — and it does not help: 8.74e-3 at best against SPSA's 5.80e-3, and no
configuration significant against the COBYLA baseline (p ≥ 0.45).

So the step count was never the binding constraint either. Sweeping `k ∈ {1,2}`
and shots ∈ {128, 512} moved the median by less than 1.4× across the whole grid.
Nothing tested reaches chemical accuracy on H₂ at 200k shots: **0 of 16** for
every configuration.

### Result 25 — the gap is shot noise, not the optimiser and not the step count

The decisive control from NEXT_STEPS, run first because it says *which* problem
to work on. Same optimisers, **exact** energies, matched evaluation counts —
98 evaluations is what a 200k budget at 2048 shots affords:

| evaluations | COBYLA | SPSA | SPS k=1 |
|---|---|---|---|
| **98** | **4.96e-05** | 2.00e-03 | 4.06e-03 |
| 390 | 2.24e-09 | 1.30e-03 | 1.66e-03 |
| 1 562 | 2.24e-09 | 8.06e-04 | 8.52e-04 |
| 10 000 | 2.24e-09 | 5.98e-04 | 5.47e-05 |

**COBYLA reaches chemical accuracy on H₂ in 98 noiseless evaluations** — the
exact count the shot-noise run gets, where it manages only 1.84e-2. A 370× gap
with the evaluation count held fixed. The optimiser has plenty of steps; each
one is just too noisy.

This reverses session 1's conclusion that measurement-side work "has nothing to
act on". That was true of H₄, where the ansatz was stuck at Hartree-Fock. H₂ is
genuinely shot-noise-limited, and it is the measurement side that binds.

Second thing the table shows: **SPSA is a slow optimiser even without noise**,
plateauing near 6e-4 at 10 000 evaluations where COBYLA hits 2.2e-9 at 390.
So neither baseline has both properties — COBYLA converges fast and collapses
under noise; SPSA tolerates noise and converges slowly.

### Result 26 — precision alone does not rescue COBYLA; it has to arrive on a schedule

If COBYLA's failure were simple noise sensitivity, precise evaluations would fix
it. They do not. H₂, 12 seeds, 48 evaluations each, varying shots per evaluation:

| σ per evaluation | COBYLA | SPSA |
|---|---|---|
| 7.60e-3 | 1.93e-2 | 3.53e-3 |
| 1.90e-3 | 1.19e-2 | 2.80e-3 |
| 9.50e-4 | 4.57e-3 | 2.95e-3 |
| 4.75e-4 (51.2M shots) | 2.68e-3 | — |

At σ = 4.75e-4, already **below** chemical accuracy, COBYLA still manages only
2.68e-3 — 54× worse than its own noiseless 4.96e-5.

The reason is structural: near the optimum the energy *differences* a
model-based method interpolates shrink quadratically in the distance to the
minimum, while shot noise stays flat. Whatever fixed precision is chosen, the
model eventually fits noise. Precision has to escalate as the run converges.

A second measurement pins the mechanism. Restarting COBYLA from its own
converged point with more shots:

| restart `rhobeg` | error before | error after |
|---|---|---|
| **1.0** | 2.008e-2 | **2.223e-3** |
| 0.3 | 2.008e-2 | 1.895e-2 |
| 0.1 | 2.008e-2 | 1.744e-2 |
| 0.03 | 2.008e-2 | 2.044e-2 |

A *large* trust region on restart is 9× better than a small one — the opposite
of the natural guess. A small region re-converges inside the same
noise-induced basin; a large one escapes it and re-explores with the newly
precise evaluations.

### Result 27 — the shot ladder: 2.1× better than SPSA, p = 0.006, and it needs headroom

`shot_ladder` runs a model-based optimiser to convergence at one shot count,
then restarts it from that point with more shots, repeating. H₂, ideal, paired
against SPSA:

| budget | levels | SPSA median | ladder median | ratio | W/L | p | ladder ≤ chem acc |
|---|---|---|---|---|---|---|---|
| 200 000 | 2 | 4.10e-3 | 6.83e-3 | 1.710 | 3/13 | **0.021** | 0/16 |
| 800 000 | 2 | 4.56e-3 | 3.11e-3 | 0.825 | 10/6 | 0.454 | 2/16 |
| 3 200 000 | 2 | 4.20e-3 | 2.90e-3 | 0.594 | 11/5 | 0.210 | 4/16 |
| 12 800 000 | 3 | 4.54e-3 | **2.50e-3** | **0.465** | 12/4 | 0.077 | **5/16** |

and at 12 seeds with hand-set levels, the best configuration was significant:
**L=3, growth 6 at 12.8M — ratio 0.467, 12 wins / 0 losses, p = 0.000.**

Chemical-accuracy hit rate goes from SPSA's 1/16 to **5/16**. Nothing in this
repository had reached it reliably before.

Two engineering points that make the difference between working and not:

* **The schedule must be derived from the budget.** Splitting a budget
  geometrically across levels whose shot counts *also* grow geometrically
  starves the expensive rungs: 4 levels growing 8× on 200k hands the top rung
  262 144 shots per evaluation and a 175k allowance — not one evaluation. Fixing
  the evaluations per rung at `E = 2(n+1)` and solving
  `s₀·E·(gᴸ−1)/(g−1) = budget` spends the whole budget with every rung fed.
* **Over-laddering is worse than not laddering.** At 800k, four levels measured
  2.48× *worse* than SPSA (0/12 wins, p = 0.000). The automatic level rule keeps
  the bottom rung at ≥ 4096 shots, which reproduces the measured best count at
  every budget tested (800k → 2, 3.2M → 2, 12.8M → 3).

**Honest limit:** below roughly `8 · E · min_useful_shots` (≈ 800k for
12-parameter H₂) there is no room for two well-fed rungs and the ladder is
significantly *worse* than SPSA. It is a budget-rich method, and the crossover
is now measured rather than guessed.

### Result 28 — an exact fast path for noiseless runs, 14x

Every group circuit is `ansatz + basis_change`, so the expensive prefix is
shared. Profiling an H4/UCCSD evaluation showed where the time actually went:

| step | share |
|---|---|
| `assign_parameters` on the ten transpiled group circuits | **65%** (5.07 s of 7.76 s) |
| Aer runs | 19% |
| `depth()` recomputed inside the loop | 11% |

So the fix is not a faster simulator. Without noise, simulating the ansatz once
and applying each group's shallow Clifford basis change to that state is
*exactly* equivalent, and only the ansatz needs its parameters bound. Depths and
two-qubit counts are structural and are now computed once at construction.

**2.31 s → 0.164 s per evaluation, 14×**, verified unbiased against the exact
energy (bias/sem = −0.65 over 10 samples) and still stochastic. Gated on
`backend is None`: under a noise model the ansatz produces a mixed state and the
shortcut would silently discard the noise, so it must not be reachable by a
flag.

### Result 29 — deeper QAOA widens the spread faster than it raises the median

Fell out of a test that broke once shot noise was resampled. 8-node 3-regular
MaxCut, 60k budget, 8 seeds:

| depth | median AR | range |
|---|---|---|
| p = 1 | 0.492 | [0.474, 0.506] |
| p = 2 | 0.513 | [0.188, 0.599] |
| p = 3 | 0.584 | [0.010, **0.706**] |

Depth raises the *ceiling* — 0.706 against 0.506 — and destroys reliability:
**three of eight seeds at p = 3 fail almost completely** (0.010–0.025). Over
seeds 0–5 the p = 3 median (0.343) is *below* p = 1's (0.492); the 8-seed median
only looks better because two good seeds happened to land in it.

At a fixed shot budget the optimiser's failure rate grows with depth faster than
the reachable approximation ratio improves. Reporting a median over a handful of
seeds — which is what QAOA depth studies usually show — hides a 40% failure rate.

The test that caught this had asserted single-seed monotonicity and passed only
because the noise was frozen. Its replacement asserts the spread claim, which is
the part that survives.

### Result 30 — the measurement work does pay off, once the regime is shot-limited

The question session 1 could not answer, now answerable because Result 25 showed
where the binding constraint actually is.

H₂ is shot-noise-limited but has only 2 measurement groups, so grouping cannot
matter there. H₄ has 10 commuting groups but its hardware-efficient ansatz is
stuck at Hartree-Fock. **H₄ + UCCSD** has both: it converges (1.08e-4 under exact
optimisation) and has groups worth arranging. Run with `shot_ladder`, ideal
simulator, 12 seeds, paired:

| budget | grouping | groups | median error | ratio | W/L | p |
|---|---|---|---|---|---|---|
| 4 000 000 | commuting | 10 | 1.131e-1 | 1.000 | — | — |
| 4 000 000 | **covariance+refine** | 19 | **8.269e-2** | **0.798** | **10/2** | **0.039** |
| 16 000 000 | commuting | 10 | 7.549e-2 | 1.000 | — | — |
| 16 000 000 | covariance+refine | 19 | 7.831e-2 | 0.884 | 7/5 | 0.774 |

**At 4M the covariance-aware grouping is significantly better end to end**
(ratio 0.798, p = 0.039) — the first time any of session 1's measurement work has
shown a significant benefit in a full optimisation rather than at a fixed state.
Not significant at 16M.

Three caveats that keep this honest, and the third is serious:

* This compares two *badly converged* runs. Both errors (0.08–0.11 Ha) exceed
  H₄'s correlation energy of 0.042 Ha, so both are worse than simply stopping at
  Hartree-Fock. A ratio between two failures is weak evidence.
* The covariance grouping produced **19** groups here against the **11** measured
  in session 1's experiment 003 for the same molecule and reference. That
  discrepancy is unexplained and needs chasing before the number is trusted.
* The likely cause of the poor convergence is that `shot_ladder`'s `rhobeg = 1.0`
  was tuned on H₂'s hardware-efficient ansatz. UCCSD's meaningful parameter scale
  is ~0.1, so a 1-radian initial trust region throws the optimiser away from
  Hartree-Fock in a 26-dimensional space. `[unverified]` — the probe testing this
  was lost to a container restart and needs re-running.

### Result 31 — the trust region has to match the ansatz, and H4/UCCSD needs more budget than 4M

Chasing Result 30's third caveat. `shot_ladder`'s `rhobeg = 1.0` was tuned on
H₂'s hardware-efficient ansatz; UCCSD's meaningful parameter scale is ~0.1.
H₄ + UCCSD, 4M budget, 8 seeds (stopping at Hartree-Fock would score 0.0418):

| `rhobeg` | median error | best seed |
|---|---|---|
| 1.00 | 1.002e-1 | 7.39e-2 |
| 0.30 | 1.485e-1 | 8.67e-2 |
| **0.10** | **4.856e-2** | 4.15e-2 |
| 0.03 | 5.201e-2 | **3.86e-2** |

The trust region matters a lot — 0.1 halves the error against the default — and
it is **the opposite direction from H₂**, where a *large* `rhobeg` was 9× better
on restart (Result 26). There is no universal value: it has to scale with the
ansatz's parameter scale, which for a coupled-cluster ansatz is set by the
amplitudes and for a hardware-efficient one by nothing in particular.

**But even at the best setting, H₄ + UCCSD at 4M shots barely beats Hartree-Fock**
(4.86e-2 against 4.18e-2; only the best of 8 seeds, 3.86e-2, is below it at all).
So Result 30's significant covariance-grouping win (ratio 0.798, p = 0.039) is
confirmed as a comparison between two runs that both fail. It is real — the
pairing is seed-matched and the sign test is sound — but it is a 20% improvement
on a method that is not working, and it should not be quoted as a validated
end-to-end benefit until the underlying run converges.

What that needs: enough budget that UCCSD's 26 parameters get the evaluations
the exact-optimisation control (Result 25) shows are required. That is the next
measurement, and it is expensive — the 14× fast path from Result 28 is what
makes it affordable at all.

---

## Session 3 — 2026-08-25 (15:25 UTC)

### Result 32 — a second numerical-stability hole, in the covariances this time

Chasing Result 30's unexplained group count. The Hartree-Fock determinant built
as a bare circuit and the UCCSD ansatz evaluated at `θ = 0` are **the same
state** — overlap 1.000000, maximum amplitude difference **2.6e-12** — and gave
**11 groups and 19 groups** on H₄.

Same failure mode as Result 8, one level down. Result 8 fixed the *coefficient*
tie-break by rounding to 12 decimals; the covariance matrix carries the same ulp
noise and nothing rounded it, so near-ties in the greedy flipped on differences
12 orders of magnitude below anything physical.

Fixed by rounding the covariance matrix to 9 decimals before any greedy step
consumes it — covariances here are O(1e-2) to O(1), so that discards nothing
real and leaves 1000× margin over the noise. Both reference paths now give
identical partitions (11 groups, 19 moves, ratio 0.708), and experiment 003's
numbers are unchanged, meaning the fix brought the estimator path into agreement
with the correct one rather than moving it.

**Consequence: Result 30's significant end-to-end win was measured on the
19-group accident, not the intended partition.** It had to be re-run.

### Result 33 — re-run: the covariance grouping's end-to-end benefit is not robust

Same experiment with the corrected partition and `rhobeg` matched to UCCSD's
parameter scale. H₄ + UCCSD, ideal, 12 seeds, paired:

| budget | commuting | covariance+refine | ratio | W/L | p |
|---|---|---|---|---|---|
| 4 000 000 | 4.847e-2 | 4.553e-2 | 0.780 | 8/4 | 0.388 |
| 16 000 000 | 4.062e-2 | **4.302e-2** | **1.085** | 5/7 | 0.774 |
| 64 000 000 | 3.913e-2 | 3.854e-2 | 0.864 | 8/4 | 0.388 |

Three budgets, three different answers — 0.780, 1.085, 0.864 — **none
significant**. The p = 0.039 of Result 30 does not survive the fix.

**Honest statement, replacing Result 30's:** covariance-aware grouping's
predicted variance benefit (0.708 on H₄, 0.632 on LiH) is real and validated
against direct sampling *at a fixed state*. It does **not** translate into a
reliable end-to-end shot saving in this setup. Averaged over the three budgets
the ratio is ~0.91, which is the size of effect that 12 seeds cannot resolve.

### Result 34 — why H₄ never converges: the shot cost, quantified

16× more budget (4M → 64M) moved the error only from 4.85e-2 to 3.91e-2, against
Hartree-Fock's 4.18e-2 — a 6% improvement on doing nothing, after 64 million
shots. Meanwhile the exact-energy control at the *same* evaluation count (108,
which is what 4M buys) reaches **9.636e-3**, four times better than Hartree-Fock.

So it is shot noise again, and now it can be costed. Measuring the estimator's
single-evaluation σ and asking what it takes to push σ below chemical accuracy:

| system | params | Σ\|c\| | σ at 4096 shots | shots for σ = 1.594 mHa | × 10(n+1) evaluations |
|---|---|---|---|---|---|
| H₂ / hea:2 | 12 | 2.04 | 3.82e-3 | 23 556 | **~3.1M** |
| H₄ / UCCSD | 26 | 10.94 | 1.98e-2 | 631 527 | **~170M** |

The H₂ figure checks out against measurement: the shot ladder first reached
chemical accuracy reliably at 12.8M (5/16 seeds), comfortably above the 3.1M
floor. **The H₄ figure says every budget tested was too small** — 64M is 2.7×
short of the requirement.

The scaling is the useful part. Per-shot variance goes as `(Σ|c|)²`, and the
evaluation count as `n`, so the shot cost to chemical accuracy scales as

    shots  ~  (Σ|c|)² · n / ε²

H₄'s Σ|c| is 5.4× H₂'s and it has 2.2× the parameters: 5.4² × 2.2 ≈ 64×, against
the measured 55× (3.1M → 170M). **This is the real barrier to VQE on molecules,
and it is not the hardware** — Σ|c| grows with system size faster than anything
in this repository reduces it.

Against that scaling, every measurement-side improvement here buys a constant
factor: general-commuting grouping ~15×, covariance-aware refinement ~1.6× in
predicted variance, the shot ladder ~2× in error. Useful, and none of them
changes the exponent.

Also confirmed: `rhobeg = 0.1` beats 1.0 on UCCSD **even with exact energies**
(9.636e-3 against 2.872e-2 at 108 evaluations), so Result 31's trust-region
finding was genuine tuning and not a noise artefact.

### Result 35 — double factorisation is a qubitisation technique, not a VQE one

Acting on Result 34's conclusion that only `Σ|c|` matters, and on NEXT_STEPS
naming Hamiltonian factorisation "the single most promising direction". It is
not, and the measurement was cheap enough that it should have come first.

Implemented double factorisation properly — eigendecompose the reshaped
two-electron tensor, diagonalise each symmetric factor — and verified it
reconstructs the two-body integrals to **1.9e-15**.

**The 1-norm goes the wrong way**, 5–10× *up*:

| molecule | Pauli Σ\|c\| | DF factors | DF 1-norm | ratio |
|---|---|---|---|---|
| H₂ | 0.988 | 3 | 10.401 | 10.53 |
| H₄ | 8.466 | 10 | 41.149 | 4.86 |
| LiH | 12.342 | 21 | 60.482 | 4.90 |
| BeH₂ (4e,6o) | 14.763 | 28 | 114.535 | 7.76 |

That comparison is asymmetric and I do not rest anything on it: the qubit
Hamiltonian's identity term needs no measurement and is excluded, while the
corresponding constant is still inside the factorised form. The 1-norm is only a
proxy anyway.

**The symmetric comparison, on the quantity that actually sets shot cost** —
total estimator variance under Neyman allocation, both schemes scored at the
Hartree-Fock reference:

| molecule | Pauli groups | Pauli variance | DF factors | DF variance | ratio |
|---|---|---|---|---|---|
| H₂ | 2 | 0.0327 | 3 | 0.1309 | **4.00** |
| H₄ | 10 | 0.5811 | 10 | 5.8409 | **10.05** |
| LiH | 41 | 0.4146 | 21 | 2.8281 | **6.82** |

**Double factorisation is 4–10× worse for VQE sampling**, even though it needs
*fewer* measurement settings (21 factors against 41 groups on LiH).

The reason is structural and should have been predictable: each factor is a
**squared** one-body operator, and squaring inflates the variance. Double
factorisation's 1-norm advantage is real for **qubitisation and QPE**, where λ
enters the query complexity of a block encoding. VQE pays a *sampling variance*,
which is a different quantity, and the transformation that helps one hurts the
other.

So the direction NEXT_STEPS ranked first is closed. Recorded as a regression
test rather than just prose: if the ratio ever inverts, this finding is wrong
and the direction reopens.

**What this leaves.** Of the three levers named for attacking the scaling wall,
one is now measured and dead. Active-space reduction trades accuracy for cost
and does not change the exponent either. Classical shadows remain untested and
are the only candidate left that changes *how* observables are estimated rather
than how the same estimator is arranged.

### Result 36 — classical shadows are worse too, and get worse with size

The third and last lever named against the scaling wall, checked the same cheap
way. The random-Pauli shadow estimator's variance carries a `3^k` factor for a
Pauli of weight `k`, so the question is whether molecular Hamiltonians'
high-weight terms kill it. Scored against general-commuting grouping at the
Hartree-Fock reference:

| molecule | terms | max weight | grouped variance | shadow variance | ratio |
|---|---|---|---|---|---|
| H₂ | 5 | 2 | 0.0327 | 1.246 | **38×** |
| H₄ | 165 | 6 | 0.5811 | 25.86 | **44×** |
| LiH | 631 | 10 | 0.4146 | 152.77 | **369×** |

The weight distribution shows exactly where it goes. On LiH the weight-9 and
weight-10 terms are **33 of 631** and carry **67% of the shadow variance**,
because `3^10 = 59 049`:

```
LiH   w=1:12t/5%   w=2:51t/3%   w=3:98t/4%   w=4:124t/6%   w=5:118t/1%
      w=6:88t/3%   w=7:72t/4%   w=8:34t/7%   w=9:20t/27%   w=10:13t/40%
```

And the penalty **grows** with system size — 38× → 44× → 369× — which is the
opposite of what a scaling fix has to do. Fermionic Hamiltonians produce
high-weight Paulis by construction (the Jordan-Wigner strings), so this is
structural rather than an artefact of these particular molecules.

**Caveat, and it is a real one:** this is the *random-Pauli* shadow estimator.
Derandomised shadows (Huang et al. 2021) choose measurement settings to target
the specific observables at hand and are substantially better. That variant is
**untested here**, and this result does not speak to it.

### Where that leaves the three levers

All three candidates named in Result 34 for attacking `shots ~ (Σ|c|)² n / ε²`
are now measured:

| lever | result |
|---|---|
| Double factorisation | **4–10× worse** for sampling variance (Result 35) |
| Active-space reduction | trades accuracy for cost; does not change the exponent |
| Classical shadows (random Pauli) | **38–369× worse**, worsening with size |

**General-commuting grouping with Neyman allocation is the best measurement
scheme available here**, and nothing tested changes the scaling. That is the
honest state of the question: the constant factors this repository has found are
real (grouping ~15×, refinement ~1.6×, the shot ladder ~2×), and the exponent is
untouched.

The one candidate left standing is derandomised shadows. If that also fails, the
conclusion is that VQE's measurement cost on molecules is not fixable at the
estimator level, and the remaining routes are algorithmic — fewer parameters, or
a different algorithm entirely.

### Result 37 — the 170M prediction is falsified: the cost model is a lower bound, not a budget

Result 34 predicted H₄ + UCCSD needs ~170M shots from the estimator's noise
scaling. Tested at 256M, 8 seeds, above the predicted requirement:

| budget | median | best seed | beats Hartree-Fock | ≤ chemical accuracy |
|---|---|---|---|---|
| 64 000 000 | 3.799e-2 | 3.246e-2 | 5/8 | **0/8** |
| 256 000 000 | 3.656e-2 | **1.766e-2** | 5/8 | **0/8** |

**4× the budget moves the median 4%** (3.80e-2 → 3.66e-2), and nothing reaches
chemical accuracy at either. The prediction is wrong.

What the model got right is the estimator: σ does fall as 1/√shots, and the H₂
figure (~3.1M) checked out against measurement. What it omits is that a precise
evaluation is only useful if the optimiser can exploit it — and Result 26 had
already shown it cannot: COBYLA at σ = 4.75e-4, *below* chemical accuracy,
still managed only 2.68e-3 against its noiseless 4.96e-5.

So the correct statement is that

    shots  ≳  (Σ|c|)² n / ε²

is a **lower bound on the estimator side**, not a sufficient budget. The
optimiser adds a second requirement that the model does not capture, and on H₄
that one binds first.

One thing does improve with budget: the *best* seed goes from 3.25e-2 to
1.77e-2, so the tail is moving even where the median is not. That points at
optimiser variance rather than estimator noise as what is left.

### Result 38 — the complete measurement-scheme ranking

Derandomised shadows implemented (greedy over product bases, minimising
`Σ_i c_i² exp(-decay·hits_i)`, with coverage enforced afterwards — a first
version left terms at zero hits, which makes the estimator undefined rather
than imprecise and showed up as an infinite variance).

All five schemes scored on the same metric — total estimator variance under
Neyman allocation, at the Hartree-Fock reference:

| scheme | H₂ | H₄ | LiH | settings on LiH |
|---|---|---|---|---|
| **general commuting + Neyman** | **1.00** | **1.00** | **1.00** | **41** |
| QWC + Neyman | 1.00 | 1.67 | 3.24 | 172 |
| derandomised shadows | 3.00 | 3.20 | 4.79 | 351 |
| double factorisation | 4.00 | 10.05 | 6.82 | 21 |
| random-Pauli shadows | 38.1 | 44.5 | 368.5 | — |

**General-commuting grouping with Neyman allocation wins on every molecule.**

Derandomisation is a large improvement on random shadows (369× → 4.8× on LiH)
but still loses, and the reason is structural rather than a tuning failure: its
settings are **product bases**, so it can only ever measure qubit-wise-commuting
sets together, while commuting grouping uses a Clifford. It also needs *more*
settings than a QWC cover (351 against 172 on LiH) because coverage forces
redundancy. Its better allocation does not make up for its weaker grouping.

Double factorisation is the interesting outlier: it needs the **fewest settings
of all** (21 against 41) and is still 6.8× worse, because each of those settings
measures a *squared* operator.

Caveat: this derandomisation is a simplified greedy, not the paper's exact
algorithm, so the 3–5× is an upper bound on how well it does. The structural
argument — product bases cannot beat Clifford grouping on setting count — does
not depend on the implementation.

**Conclusion for the measurement side.** Five schemes, one metric, three
molecules: the scheme this repository already defaults to is the best of them,
and none of the alternatives changes the `(Σ|c|)²` scaling. The measurement
question is, as far as these tests reach, settled.

---

## Session 4 — 2026-08-25 (20:24 UTC)

### Result 39 — multi-start does not beat one long run; selection eats the gain

The clue from Result 37 was that at 256M on H₄ the *best* seed reached 1.77e-2
while the median stayed at 3.66e-2 — a 2× gap that is optimiser variance, not
estimator noise. Splitting the budget across starts is the obvious way to buy
it, so it was implemented, with the selection charged honestly: a fraction of
the budget is reserved to re-evaluate every candidate at equal precision and
pick the lowest.

That honesty matters. Ranking candidates by each run's own best *observed* value
would be free and wrong — it is the minimum of many noisy draws, biased low by
roughly the run's own noise, so it would systematically favour the **noisiest**
run rather than the best one.

H₂, `hea:2`, 16 seeds, paired against the single-start ladder:

| budget | K=2 | K=4 | K=8 |
|---|---|---|---|
| 3 200 000 | 1.93 (3/13, **p=0.021**) | 2.05 (3/13, **p=0.021**) | 2.44 (2/14, **p=0.004**) |
| 12 800 000 | 1.02 (8/8) | 1.32 (7/9) | 1.30 (5/11) |

**Significantly worse at the smaller budget**, and no better at the larger one.
The mechanism is the one Result 27 already established: the shot ladder needs
budget headroom, and eight starts of 400k each all sit below its crossover.

H₄ + UCCSD at 16M agrees — single start 4.062e-2, K=2 4.989e-2, K=4 4.811e-2.

**But the selection was also being done badly**, which is worth separating. At
`audit_fraction = 0.1` of 12.8M split eight ways, each candidate is scored with
160k shots — σ ≈ 6.1e-4 against a candidate spread of ~1e-3, so the ranking is
barely better than a coin flip. Sweeping it:

| configuration | median | ≤ chemical accuracy | ratio vs single |
|---|---|---|---|
| ladder, single start | 1.691e-3 | 7/16 | 1.000 |
| K=8, audit 0.10 | 2.544e-3 | 4/16 | 1.298 |
| **K=8, audit 0.30** | **1.517e-3** | **9/16** | 0.827 (9/7, p=0.804) |
| K=8, audit 0.50 | 2.281e-3 | 4/16 | 1.244 |
| K=3, audit 0.30 | 2.073e-3 | 6/16 | 1.402 |

Non-monotone, with an optimum: too little audit makes the ranking unreliable,
too much starves the starts. **At the best setting multi-start is
statistically indistinguishable from a single run** (ratio 0.827, p = 0.804) —
it reaches chemical accuracy on 9/16 seeds against 7/16, which is worth noting
and is not significant.

**Conclusion:** the optimiser variance is real, and diversification cannot
capture it at equal budget because selecting the good run costs about as much as
producing it.

### Result 40 — automatic trust-radius estimation: three principled attempts, three failures

`rhobeg` has no universal value — H₂'s hardware-efficient ansatz wants 1.0 and
H₄'s UCCSD wants 0.1, each 2–9× worse at the other's setting (Results 26, 31) —
so estimating it from the problem seemed like the obvious cleanup. It does not
work, and the three failures are each instructive:

| attempt | rule | H₂ (wants 1.0) | H₄ (wants 0.1) | why it failed |
|---|---|---|---|---|
| 1 | step that moves E by 10σ | 0.8 ✓ | 0.2 / 0.8, unstable ✗ | keys on *noise* scale; UCCSD's noise is large while its parameter scale is small |
| 2 | where quadratic breaks, `E(2r)=4E(r)` | 0.05 ✗ | 0.05 ✗ | wrong model — at a non-stationary point the energy is linear-dominated, ratio 2 not 4, so the test fails everywhere |
| 3 | where linear breaks, `E(2r)=2E(r)` | 0.05 ✗ | 0.05 ✗ | right test (COBYLA *is* a linear-model method), defeated by noise: σ is 2.7e-3 / 1.4e-2 at 8192 shots and swamps the departure from linearity |

Attempt 3 is the one matched to the algorithm and it still fails, which makes
the common cause clear: **the signal — where a model stops holding — is smaller
than the shot noise at any affordable probe cost.** Each estimate cost 90k shots
and bought nothing.

So the function now refuses rather than guessing, and the measured values live
in a documented table with the physical reason they differ: a hardware-efficient
ansatz's parameters are rotation angles with period 2π and no preferred scale,
while a coupled-cluster ansatz's are cluster amplitudes, physically O(0.1), so a
1-radian step leaves the region where the ansatz means anything.

Shipping a heuristic that is right on one system and wrong on the other would be
worse than requiring the choice.

### Result 41 — the shot ladder already *is* the practical stochastic trust region

The last open optimiser item, and it closes negatively for a reason worth
keeping.

A STORM-style method ties its **sample count to its trust radius**: the model
must be fully linear on the region, `|model − f| ≤ κ·radius²`, and with shot
noise `σ₁/√n` per evaluation that is `n ≥ (σ₁/(κ·radius²))²`. Shots therefore
grow as `radius⁻⁴` — halving the radius costs 16× the shots, which is exactly
the escalation `shot_ladder` performs by hand, arriving from the geometry
instead of from tuning. That is the design Result 26 argues for.

It loses anyway. H₂, `hea:2`, 12.8M shots, 16 seeds, paired against the ladder,
sweeping the accuracy constant over four orders of magnitude:

| configuration | median | iterations | ≤ chem acc | ratio | p |
|---|---|---|---|---|---|
| **shot ladder** | **1.691e-3** | — | **7/16** | 1.000 | — |
| STORM κ=1, 4 probes | 1.152e-2 | 68 | 0/16 | 6.24 | 0.000 |
| STORM κ=100, 4 probes | 9.875e-3 | 112 | 1/16 | 4.44 | 0.001 |
| **STORM κ=1000, 4 probes** | **8.263e-3** | 187 | 1/16 | **3.28** | 0.001 |
| STORM κ=10000, 4 probes | 1.127e-2 | 7340 | 0/16 | 4.45 | 0.000 |

**Never closer than 3.3×, always significant.** And κ=10 000 gives 7 340
iterations while still losing 4.4×, so it is not iteration starvation either.

One real bug was found and fixed along the way, worth its own note. The first
version estimated the gradient with a *single* simultaneous-perturbation probe
and then stepped a full radius **along that random direction**. In 12 dimensions
a random direction has cosine ~1/√n with the true gradient, so the realised
decrease fell far short of the predicted one, every step was rejected, and each
rejection shrank the radius — which under the `radius⁻⁴` rule quadruples the
next iteration's cost. A death spiral, and measurably so: **6 iterations, all
rejected, the energy never moving off Hartree-Fock, and the whole 12.8M budget
spent.** Averaging four probes and stepping along the average fixed the spiral
and bought 2× (9.5 → 6.2), which was not enough.

**Why it loses, and the useful part of the finding.** The comparison is not
really "trust region versus ladder". It is *a four-probe gradient plus a trust
region* versus *a full COBYLA model plus precision escalation*. COBYLA builds an
`n`-dimensional linear model from `n+1` points and takes model-informed steps; a
four-probe average in 12 dimensions is a far cruder direction. Doing STORM
properly needs the `n+1`-point interpolation set — at which point it *is*
COBYLA with adaptive sampling, which is what the ladder already assembles.

So: **the precision-escalation idea is right and already implemented; the model
is what the ladder gets for free by delegating to COBYLA, and hand-rolling a
cheaper one loses more than the adaptive radius gains.**

### Session 4 closing state

All three optimiser sub-items are now measured and all three close negatively:

| item | result |
|---|---|
| Multi-start | significantly worse at small budgets (2.44×, p = 0.004); indistinguishable at best (0.827, p = 0.80) |
| Automatic `rhobeg` | three principled heuristics, three failures; the signal is below the shot noise |
| Noise-aware trust region | never closer than 3.3×, p ≤ 0.001 across four orders of magnitude in κ |

`shot_ladder` with a per-ansatz `rhobeg` remains the best optimiser found, at
1.691e-3 median and 7/16 chemical accuracy on H₂ at 12.8M shots.

