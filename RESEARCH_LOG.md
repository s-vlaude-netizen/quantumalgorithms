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

### Result 3 — variance-adaptive shot allocation is the big win

Experiment 001, H₂, ideal simulator, 120k shot budget, 4 seeds, all
configurations stopped at exactly the same shot count. Paired against
`cobyla|qwc|uniform`, on `noiseless_error` (ratio < 1 is better):

| configuration | ratio | W/L/T |
|---|---|---|
| `qwc \| coefficient` | 0.628 | 4/0/0 |
| `commuting \| coefficient` | 0.628 | 4/0/0 |
| **`qwc \| adaptive`** | **0.264** | **4/0/0** |
| **`commuting \| adaptive`** | **0.264** | **4/0/0** |
| `spsa \| commuting \| adaptive` | 0.487 | 4/0/0 |

**Neyman allocation on the running empirical variance gives a 3.8× error
reduction at equal shot count.** Since a shot-noise-limited error scales as
1/√N, that is worth roughly **14× fewer shots for the same accuracy**.
Coefficient-weighted allocation captures about half the benefit. `p = 0.125`
is the floor achievable with 4 seeds (4/0 wins); needs more seeds to call
significant, but the direction is unambiguous and consistent.

H₂ is *not* a discriminating test for grouping — QWC and commuting both give 2
groups, so their rows are identical by construction. Grouping needs H₄/LiH.

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

### Open question carried forward

Scaling `fake_kolkata`'s gate errors to 0.5× / 0.25× / 0.1× left the energy
error at the reference state essentially unchanged (0.089 / 0.084 / 0.089 Ha),
i.e. it saturates. Disabling thermal relaxation moved it only slightly
(−1.2570 → −1.2473), so T1/T2 is *not* the explanation. Unresolved — most
likely readout error, or the scaling not reaching the channel it should.
`[unverified]` Needs a controlled ablation before any claim is made about where
the noise floor comes from.

---
