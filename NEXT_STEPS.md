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

### The optimiser line is closed too; what is left is not optimiser work
All three optimiser sub-items are measured and all three close negatively
(Results 39, 40, 41):

| item | result |
|---|---|
| multi-start | worse at small budgets (2.44×, p = 0.004); indistinguishable at best |
| automatic `rhobeg` | three heuristics, three failures — signal is below the noise |
| noise-aware trust region | never closer than 3.3×, p ≤ 0.001 over four orders of magnitude in κ |

`shot_ladder` with a per-ansatz `rhobeg` stands as the best optimiser found:
**1.691e-3 median, 7/16 chemical accuracy on H₂ at 12.8M shots**. The useful
insight from the last one: the ladder already *is* the practical stochastic
trust region — COBYLA supplies the model, the ladder supplies the precision
escalation, and hand-rolling a cheaper model loses more than an adaptive radius
gains.

**Where that leaves the project.** Two whole lines are now closed by
measurement:

* *Measurement schemes* (Session 3): five schemes ranked, the repository's
  default wins, none changes the `(Σ|c|)²` scaling.
* *Optimisers* (Session 4): three approaches, none beats the shot ladder.

So the remaining levers are neither estimator nor optimiser. In order of what
the measurements point at:

### 1. Fewer parameters, not better optimisation  ← **the live direction**
Result 34's scaling is `shots ~ (Σ|c|)² · n / ε²`, and `n` is the one factor
still untouched. ADAPT-VQE grows the ansatz operator by operator from a measured
gradient pool, reaching a given accuracy with far fewer parameters than a fixed
UCCSD — and every measurement here says parameter count is what the budget buys.
The measurement machinery it needs (grouped estimation of an operator pool's
gradients) already exists.

Acceptance test: reach chemical accuracy on H₄ at a budget where fixed UCCSD
does not (it fails at 256M, Result 37).

### 2. Smaller problems, honestly scored
Everything here is H₂/H₄/LiH at STO-3G. Before any of these conclusions is
generalised, one of them should be checked on a system where the answer is
known and the size is larger — H₂O or an active-space BeH₂ — to see whether the
rankings hold or are an artefact of very small Hamiltonians.

### 3. A classical baseline on every result
Nothing in this repository is compared against what a laptop does with the same
problem. CCSD(T) reaches chemical accuracy on all of these in milliseconds. That
comparison belongs in the README, not because it is flattering, but because
without it none of the shot counts mean anything.

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
