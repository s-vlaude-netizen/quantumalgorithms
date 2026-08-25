# Next Steps

Working queue, highest value first. Each item says what to measure and what
would count as a result, so a session can pick one up cold.

Update this file at the end of every session: tick off what was done, re-rank
what is left, add what the results suggested.

---

## Now

### 0. Build an ansatz that couples to double excitations at shallow depth  ← **the blocker**
Result 10 is the finding that reorders everything else. The hardware-efficient
ansatz has **exactly zero gradient** at Hartree-Fock — 0 of 46 parameters at
2 reps, 0 of 114 at 6 reps — because a real-amplitude ansatz's first-order
response reaches only single excitations, and Brillouin's theorem kills those.
Correlation energy is in the doubles. So H₄ runs land on Hartree-Fock and
recover *zero* correlation energy, at any depth and any budget.

UCCSD reaches chemical accuracy (9.7e-6) precisely because its generators are
double excitations — at 1096 two-qubit gates, which is device-fatal.

**The gap to close: UCCSD-like coupling to doubles at hardware-efficient depth.**
Candidates, cheapest first:

- **Quantum-number-preserving gate fabrics** (Anselmetti et al.): four-qubit
  blocks that conserve particle number and spin and span the doubles directly.
- **Givens-rotation / particle-conserving networks**: shallow, and their
  generators are single+double excitations by construction.
- **k-UpCCGSD**: paired doubles only, so far fewer terms than UCCSD, tunable
  depth via k.

Acceptance test: non-zero gradient at Hartree-Fock, chemical accuracy on H₄
under exact optimisation, and two-qubit count within ~5× of the
hardware-efficient ansatz. Until something passes that, no H₄ measurement of
anything else means much.

### 1. Re-run the measurement studies once an ansatz converges  *(blocked on 0)*
Every H₄ number in this session was taken in a regime where the optimiser could
not move, so the H₄ allocation/grouping/optimiser comparisons say nothing about
those methods. Experiment 001 on H₄ gave ratios 0.90–1.01, all with p ≥ 0.29 —
not a negative result about allocation, just a dead problem.

The H₂ results stand (adaptive allocation ratio 0.264, 4/4 seeds), because H₂
converges. Repeat on H₄/LiH once item 0 lands, and under `medium` device noise.

### 1b. Covariance grouping end-to-end  *(also blocked on 0)*
Now reachable via `grouping="covariance"` / `"covariance+refine"`. Validated at
a fixed state: 0.708 (H₄) / 0.632 (LiH) variance ratio, empirically confirmed
against 300 sampled estimates. But at a 150k budget on H₄ it changes nothing
(5.15e-2 vs 5.07e-2) because the run is not shot-noise-limited — see item 0.

Still to measure once it can matter: whether the win survives a full run, given
that covariances are computed once at Hartree-Fock and the optimiser walks away
from there. And whether re-grouping partway (from the empirical group variances
already collected) recovers any decay.

### 1c. Better partition search
Local refinement converges to a local optimum of *single-term* moves (Result 9).
Untried: pairwise swaps, multi-start from perturbed greedy solutions, annealing.
Unknown how far 0.632 is from optimal — worth bounding exactly on a small case.

### 2. Resolve the noise-floor question
Scaling gate errors 10× down did not move the error. Ablate properly: readout
on/off, thermal relaxation on/off, gate error scaled — one factor at a time, at
a fixed reference state so there is no optimiser in the loop.

### 3. Budget scaling: what does chemical accuracy actually cost?
Nothing tested reaches 1.594 mHa on H₂ reliably (best: 1/8 seeds). Sweep the
budget over 10⁵–10⁷ shots for the best configuration and find where chemical
accuracy becomes reliable (≥ 6/8 seeds). *Shots to chemical accuracy* is the
benchmark everything else should be measured against, and it is unknown even
for H₂.

---

## Next

### 4. Error mitigation, measured against its cost
Zero-noise extrapolation and readout mitigation both *buy* accuracy with
*extra shots*. Under budget parity that trade is not obviously favourable, and
it is usually reported without it. Implement ZNE (gate folding) and M3-style
readout correction, and measure error at matched total shots. A negative result
here would be worth writing down.

### 5. Larger parameter counts, where the shot-frugal literature should win
The adaptive-optimiser negative result was on 12 parameters (H₂). COBYLA
degrades badly above ~50. Re-test adaptive SPSA on LiH (≈ 100+ parameters)
before concluding it does not help.

**iCANS is impractical here and should not be re-run naively:** it needs `2n`
circuit evaluations per step, which measured at ~3.5 h for 8 seeds of H₄ (46
parameters) and was aborted. If it is worth testing at scale, it needs the
random-operator-sampling variant (Rosalin) that avoids the full parameter-shift
sweep, not the plain version.

### 6. QAOA parameter transfer  *(driver now built — `qres/qaoa.py`)*
The driver exists and is tested; nothing has been measured with it yet. Test the
fixed-angle / parameter-transfer literature (LINXFER and relatives, 2025):
pre-trained angles claim to remove instance-specific optimisation entirely,
which under a shot-budget metric would be a very large win. Transfer from small
random-regular MaxCut instances to larger ones and measure approximation ratio
vs. shots.

### 7. Circuit duration as the lever
The cost model charges `circuit_duration + reset_delay` per shot, and the reset
delay (250 µs) dominates everything. Check whether that is realistic for current
hardware — if devices support active reset at ~1–10 µs, circuit duration starts
to matter and depth-reduction work becomes worth doing. Currently depth
optimisation has almost no effect on the headline metric, and it is worth
knowing whether that is the cost model's fault or a real fact.

---

## Later / speculative

- **Ablate the variance model.** `decay = 0.85` and `prior_strength = 64` are
  guesses. Sweep them; check whether the non-stationarity argument for `decay`
  holds up.
- **ADAPT-VQE.** Grow the ansatz operator by operator from a measured gradient
  pool. Expensive, but it is the method that reaches chemical accuracy with the
  shallowest circuits, and the measurement machinery here is what it needs.
- **Protein folding / lattice models.** The user named this. HP-lattice folding
  maps to a QUBO and drops straight into the existing Ising path — cheap to add
  once the QAOA driver (item 6) exists.
- **Beyond-classical honesty check.** Nothing here is a quantum advantage claim
  and nothing should be presented as one. The useful comparison is against a
  good *classical* baseline on the same problem (CCSD(T) for chemistry,
  Goemans-Williamson for MaxCut). Add those baselines so every result is scored
  against what a laptop can already do.

---

## Standing rules for this project

1. Check the **ideal-simulator** result before attributing any error to noise.
   Result 5 was a four-hour detour into "noise" that was never noise.
2. Budget-matched or it is not a comparison. Watch for hyperparameter schedules
   anchored to `maxiter` (see the bug list in the log).
3. Multiple seeds, median + IQR, paired sign test. Single-seed VQE numbers are
   noise.
4. Verify new measurement machinery **algebraically** against statevector
   expectation values, not by whether the energies look plausible.
5. Record negative results in `RESEARCH_LOG.md`. They are the cheapest thing in
   the repository to produce and the most expensive to re-derive.
6. Every greedy or sorting step needs an explicit tie-break, and any result
   depending on one must be checked **across processes**, not just across calls.
   Single-process determinism proves nothing — see Result 8.
