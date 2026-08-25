# Next Steps

Working queue, highest value first. Each item says what to measure and what
would count as a result, so a session can pick one up cold.

Update this file at the end of every session: tick off what was done, re-rank
what is left, add what the results suggested.

---

## Now

### The measurement side is settled; the optimiser side is not
Three sessions of measurement-scheme work end in a clean ranking (Result 38).
Five schemes, one metric — total estimator variance under Neyman allocation at
the Hartree-Fock reference — three molecules:

| scheme | H₂ | H₄ | LiH | settings on LiH |
|---|---|---|---|---|
| **general commuting + Neyman** | **1.00** | **1.00** | **1.00** | 41 |
| QWC + Neyman | 1.00 | 1.67 | 3.24 | 172 |
| derandomised shadows | 3.00 | 3.20 | 4.79 | 351 |
| double factorisation | 4.00 | 10.05 | 6.82 | **21** |
| random-Pauli shadows | 38.1 | 44.5 | 368.5 | — |

**What the repository already defaults to wins on every molecule**, and none of
the alternatives changes the `(Σ|c|)²` scaling. Do not re-open this without a
scheme that is not on this list.

### 1. The optimiser is what binds — and two of three sub-items are now closed
Result 37 falsified the shot-cost model as a *budget*: H₄ + UCCSD at 256M, above
the 170M the estimator noise predicted, still reaches 0/8 chemical accuracy. The
estimator side is right; what it omits is that a precise evaluation is only
useful if the optimiser can exploit it.

**Closed this session:**

* ~~Multi-start~~ (Result 39). Significantly *worse* at small budgets
  (2.44× at 3.2M, p = 0.004) and indistinguishable at the best audit setting
  (0.827, p = 0.804). The optimiser variance is real, but selecting the good
  run costs about as much as producing it, so diversification cannot capture it
  at equal budget.
* ~~Automatic `rhobeg`~~ (Result 40). Three principled heuristics, three
  failures — the last one correctly matched to COBYLA's linear model and still
  defeated by shot noise. The signal is smaller than σ at any affordable probe
  cost. Now a documented table, `optimizers.TRUST_RADIUS`, with the measured
  values and the physical reason they differ by 10×.

**Still open, and now the only optimiser item left:**

**A noise-aware trust region.** `shot_ladder` restarts COBYLA, throwing away its
model at every rung, and COBYLA assumes exact function values. A stochastic
trust-region method (STORM-style) instead sizes its *sample count* from the
model's own uncertainty: it re-samples until the model is trustworthy at the
current radius, shrinks the radius when it is not, and accepts a step only when
the predicted decrease is large against the noise.

That is the one design that addresses Result 26 directly — near the optimum the
differences a model interpolates shrink quadratically while noise stays flat, so
the sample count *has* to be tied to the radius. It would also subsume
`shot_ladder`'s hand-set `growth` and `levels`, and possibly `rhobeg` too, since
the radius becomes adaptive rather than initial.

Acceptance test: beat `shot_ladder`'s 1.691e-3 median and 7/16 chemical-accuracy
rate on H₂ at 12.8M over ≥ 16 seeds, and do it without a per-ansatz constant.

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
