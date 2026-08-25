# Next Steps

Working queue, highest value first. Each item says what to measure and what
would count as a result, so a session can pick one up cold.

Update this file at the end of every session: tick off what was done, re-rank
what is left, add what the results suggested.

---

## Now

### 0. The scaling wall, and what could actually move it  ← **the finding that reframes everything**
Result 34 costs the barrier: the shots to reach chemical accuracy scale as

    shots  ~  (Σ|c|)² · n / ε²

measured at ~3.1M for H₂ (12 parameters, Σ|c| = 2.04) and **~170M for H₄/UCCSD**
(26 parameters, Σ|c| = 10.94) — a 55× jump for one more pair of hydrogens,
matching the 64× the formula predicts.

Everything this repository has built buys a *constant factor* against that:
general-commuting grouping ~15×, covariance refinement ~1.6× in predicted
variance, the shot ladder ~2× in error. None of them touches the exponent, and
`Σ|c|` grows with system size regardless.

So the honest question for the remaining sessions is **what reduces `Σ|c|` or
`n`, not what reduces the constant**:

1. ~~**Hamiltonian factorisation.**~~ **Measured and closed (Result 35).**
   Double factorisation is **4–10× worse** for VQE sampling variance (H₂ 4.00×,
   H₄ 10.05×, LiH 6.82×), despite needing fewer measurement settings, because
   each factor is a *squared* one-body operator and squaring inflates variance.
   Its 1-norm advantage is real for qubitisation/QPE, where λ enters a block
   encoding's query complexity — a different quantity from a sampling variance.
   Kept as a regression test; reopen only if that ratio inverts.
2. **Active-space reduction.** Fewer orbitals means fewer terms and fewer
   parameters, at the cost of accuracy. Worth mapping the trade-off: what does
   `Σ|c|` and the shot requirement do as the active space shrinks?
3. ~~**Classical shadows (random Pauli).**~~ **Measured and closed (Result 36).**
   **38–369× worse** than grouping, and worsening with system size, because the
   `3^k` weight penalty falls on exactly the high-weight Paulis that
   Jordan-Wigner strings produce: on LiH, 33 of 631 terms carry 67% of the
   variance.

   **Still open: derandomised shadows** (Huang et al. 2021), which choose
   settings to target the observables at hand rather than sampling bases at
   random. That is the last untested candidate, and the same cheap check
   applies — predicted variance at a reference state, before any circuits.
   If it also fails, VQE's measurement cost on molecules is not fixable at the
   estimator level and the remaining routes are algorithmic.

### 1. Retire or re-scope the covariance grouping
Result 33: with the numerical-stability fix, the end-to-end benefit is
0.780 / 1.085 / 0.864 across three budgets, **none significant**. The predicted
variance benefit (0.708 H₄, 0.632 LiH) is real at a fixed state and does not
survive a full run.

Either find why (the reference goes stale as the optimiser moves — testable by
re-grouping partway from the empirical group variances already collected), or
record it as a fixed-state-only result and stop quoting it as a shot saving.
Do not spend another session on it without deciding which.

### 2. Verify the 170M prediction  *(running)*
H₄ + UCCSD at 256M shots, 8 seeds. If it converges there and not at 64M, the
cost model in Result 34 is validated and becomes the benchmark every future
result is scored against. If it does not, the model is missing something.

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
7. **At least 24 seeds before believing a VQE error comparison.** Measured, not
   assumed: a 4-seed result showing a 3.8× effect with 4/0 wins became 1.3× and
   p = 0.54 at 24 seeds (Results 3 and 18). At 4 seeds, 4/0 is the best
   obtainable outcome and carries almost no evidence.
8. **Check the gradient at the starting point** before running anything. Three
   separate multi-hour detours this session were an ansatz sitting on an exact
   stationary point (Results 10, 19, 20). It costs `n` statevector evaluations
   to rule out.
