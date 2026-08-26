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

### 1. ADAPT-VQE: measured, improved 4.6×, and now at cost parity
Built, measured, and made cheaper (Results 44–47).

**What holds.** The parameter reduction is real and grows with system size: 1 vs
3 on H₂, 9 vs 26 on H₄, 5 vs 92 on LiH.

**What did not.** A parameter count is not a cost. Standard ADAPT costs 647
evaluations on H₄ against fixed UCCSD's 134 — 5× *worse* — because it
re-optimises after every growth step. The scaling argument that motivated the
direction (`n` is the untouched factor in `shots ~ (Σ|c|)² n / ε²`) was about
the final parameter count, not the cost of finding it.

**What fixes it.** ADAPT's cost is *(growth steps)* × *(cost of one
re-optimisation)*. The lazy schedule shrinks the second 3.1×, batching shrinks
the first, and together they give **4.6×**:

| variant | evaluations | vs fixed UCCSD |
|---|---|---|
| standard ADAPT | 647 | 0.21× |
| lazy only | 207 | 0.65× |
| **batch 5 + lazy** *(now the default)* | **141** | **0.95×** |

At cost parity, that ansatz uses **10 parameters against 26**, which transpiles
to **665 two-qubit gates against 1350** — 2.1× shallower and **18× more
surviving signal** on `fake_torino`. It reaches 3.29e-4.

**Open:**

1. **Finish LiH.** The 18× parameter reduction is largest there and its UCCSD
   arm is still running — COBYLA on 92 parameters with a deep circuit per
   evaluation. It is the case that could show more than parity.
2. **Shrink the pool.** A gradient sweep costs 28× an energy on LiH because the
   commutators barely share measurement groups with `H`. Qubit-ADAPT pools are
   much smaller. The sweeps are only 1% of the cost at H₄'s size, but they scale
   with the pool and the pool scales as `N⁴`.
3. **Take it to shot noise.** Everything in Results 44–47 is exact arithmetic.
   The batched+lazy schedule makes far fewer, larger optimisation calls, which
   is a different noise profile from standard ADAPT — untested.

### 2. The size wall, now confirmed from both sides — this is the top item
Result 42 (chemistry) and Result 50 (optimisation) reach the same conclusion by
completely different routes: **at the sizes this project can score honestly, the
classical method already wins outright.** FCI solves H₂/H₄/LiH to 12 decimals in
~100 ms; Goemans-Williamson finds the *exact* MaxCut optimum on 100% of
instances up to n = 20 in 66–191 ms. Every ratio measured below those ceilings
is measured on instances with no hard part left.

That makes size the binding constraint on this whole project, not an item on a
list. Both branches need the same thing — a reference that survives past brute
force:

* **Chemistry:** H₂O or active-space BeH₂, where FCI is still available but
  strained, to see whether the Result 38/47 rankings are small-Hamiltonian
  artefacts.
* **Optimisation:** *answered — Result 51.* Certainty ends between **n = 40 and
  n = 60**: up to 40 the SDP and iterated local search agree on every instance
  (and are both exactly optimal wherever brute force can confirm it); at 60 they
  first diverge; by 100 they disagree on all twelve. So a QAOA number means
  something only at **n ≥ 60** — and the target there is not Goemans-Williamson's
  0.878 guarantee but a fifty-line hill-climber that finds a 2.55% better cut in
  one second. GW never wins once at a matched budget.

### 3. A classical baseline on every result  *(done — Results 42 and 50)*
Both areas the user named now have one: CCSD(T)/FCI for chemistry
(`qres/classical.py`), and greedy / local search / Goemans-Williamson for
MaxCut (`qres/classical_optimization.py`). The README states the classical
answer first. Keep it that way for anything added.

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

### 6. QAOA parameter transfer  *(driver built and now baselined — Result 50)*
The driver works and QAOA reaches the optimum on n = 10–18 3-regular MaxCut, at
67k–135k shots against local search's 0.8 ms. Test the fixed-angle /
parameter-transfer literature (LINXFER and relatives, 2025): pre-trained angles
claim to remove instance-specific optimisation entirely, which under a
shot-budget metric would be a very large win.

**But not below n = 60**, which Result 51 pins down. Below 40 both strong
classical methods are still exactly optimal, so any approximation ratio there is
measuring nothing. Parameter transfer is the right protocol for that constraint
— transfer from small instances to large ones is the one QAOA variant that does
not need the large instance to be optimised on, which is what makes n ≥ 60
reachable at all. Score it against `iterated_local_search` at matched
wall-clock, not against GW.

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
- **Beyond-classical honesty check.** *(baselines now exist — Results 42, 50.)*
  Nothing here is a quantum advantage claim and nothing should be presented as
  one. Both baselines currently win outright at every size that can be scored,
  which is the finding, not a gap in the comparison.

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
9. **Never skip a failed instance silently, and print the denominator.** A
   `try/except: continue` around instance generation, with the rate still
   divided by the number *attempted*, turned 3 successes out of 10 into "optimal
   on 30% of instances" (Result 50). Report `built` beside every rate.
10. **A rate and a mean that contradict each other mean the harness is wrong,
    not the algorithm.** "Optimal on 30%" beside "mean ratio 1.0000" is
    impossible; that inconsistency is what exposed the bug above. Print both
    where they can be compared — five of this project's bugs produced entirely
    plausible individual numbers, and not one was caught by reading the code.
