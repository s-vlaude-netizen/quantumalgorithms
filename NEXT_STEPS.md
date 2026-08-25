# Next Steps

Working queue, highest value first. Each item says what to measure and what
would count as a result, so a session can pick one up cold.

Update this file at the end of every session: tick off what was done, re-rank
what is left, add what the results suggested.

---

## Now

### 1. Confirm the allocation win where grouping actually differs
Result 3 (3.8× error reduction from adaptive Neyman allocation) was measured on
H₂, where QWC and general-commuting both give 2 groups — so the grouping axis
was degenerate. Experiment 001 on **H₄** (8 seeds, 300k budget, ideal) is
running; **LiH** (631 terms, 172→41 groups) is still to do, as is the `medium`
device-noise repeat of both.

*Result if it works:* the allocation win holds or grows with group count, and
the grouping axis separates. *Result if it fails:* adaptive allocation was an
H₂ artefact — equally worth knowing.

Note on cost: the `none` grouping is ~11× slower to simulate than `commuting`
(520 s vs 47 s for 8 seeds of H₄), tracking group count almost linearly, since
Aer's cost is per-circuit. Budget for it or drop it once its baseline role is
served.

### 1b. Wire covariance-aware grouping into `ShotEstimator`  ← **the gap**
Results 7 and 9 give a validated 0.708 (H₄) / 0.632 (LiH) variance ratio — 1.4×
to 1.6× fewer shots — but every number is measured **at a fixed state**. The
grouping is not reachable from `run_vqe`, so no end-to-end optimisation has ever
used it. Add `grouping="covariance"` with a reference state, then re-run
experiment 001 and check the win survives a full run, where the state moves away
from the reference the covariances were computed at.

That last point is the real open risk: the covariances are evaluated once at
Hartree-Fock, and the optimiser walks away from it. Two things to measure:
whether the win decays along the trajectory, and whether re-grouping partway
(using the empirical group variances already being collected) recovers it.

### 1c. Better partition search
Local refinement converges to a local optimum of *single-term* moves (Result 9).
Cheap things left untried: pairwise swaps, multi-start from perturbed greedy
solutions, simulated annealing. Unknown how far 0.632 is from optimal — worth
bounding on a small case where the optimum can be found exactly.

### 2. Resolve the noise-floor question (open question, Result 6 section)
Scaling gate errors 10× down did not move the error. Ablate properly:
readout error on/off, thermal relaxation on/off, gate error scaled — one factor
at a time, at the fixed reference state so there is no optimiser in the loop.
Until this is settled, no claim about *where* device error comes from is safe.

### 3. Budget scaling: what does chemical accuracy actually cost?
Nothing tested so far reaches 1.594 mHa on H₂ (best: 1/8 seeds). Sweep the
budget over 10⁵–10⁷ shots for the best-known configuration and find where
chemical accuracy becomes reliable (≥ 6/8 seeds). That number — *shots to
chemical accuracy* — is the headline benchmark everything else should be
measured against, and it is currently unknown even for H₂.

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
