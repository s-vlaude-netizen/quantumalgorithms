# Next Steps

Working queue, highest value first. Each item says what to measure and what
would count as a result, so a session can pick one up cold.

Update this file at the end of every session: tick off what was done, re-rank
what is left, add what the results suggested.

---

## Now

### Where this stands after Results 60-63: three classes measured, three failures

The question was narrowed to one shape (Result 61): a problem whose **required
relative accuracy** is loose enough to fit the device's gate budget, and whose
**classical baseline** is genuinely hard. The budget itself is measured — a
two-qubit gate costs ~0.1% of the observable's scale on a Heron-class device,
constant across two problem classes, so `max gates ~ required accuracy / 0.001`.

| | accuracy requirement | classical baseline | encoding cost |
|---|---|---|---|
| **chemistry** | **fails** — 0.019% needs a 0.2-gate budget; an ansatz needs 665–1300 | fails — FCI exact in 100 ms | — |
| **MaxCut** | passes — 5%, budget 50, QAOA p=1 uses 48 | **fails** — 1 ms hill-climb beats it (Result 51) | passes |
| **HP folding** | passes — 7%, budget ~70 | **fails** — 3 of 4 literature optima in under a minute (Result 64) | **fails** — 1 538 gates at N = 6 |

Three classes, three different failure points, all measured rather than argued.

**And the wall is sized (Result 65).** Measured across seven device generations,
the cost per gate *is* the device's two-qubit error rate (ratio 0.64–1.09,
`(2q error)^0.93`, log-log correlation 0.952). Batched ADAPT's 665 gates need an
error rate of **~1.6e-6**; the best snapshot available is 0.00127, **a factor of
~800**, against 6× delivered by recent generations. Nothing algorithmic in this
repository closes an 800× hardware gap, and every constant-factor result here is
on the wrong side of it.

**The one thing that could still move this**, and the only concrete item left:

1. **The low-locality folding encoding.** Result 63 measured the *naive* turn
   encoding, which is dense (2ⁿ terms at full weight) because self-avoidance has
   no locality. Published constructions (Robert et al. and relatives) spend
   ancilla qubits to bound interaction weight. Build one, measure its gate count
   the same way, and check it against the ~70-gate budget. This is the only
   candidate that passes the accuracy test and might pass the classical test,
   so it is the only place a positive result could come from.

2. **Fewer gates for the same accuracy**, if item 1 fails. Result 47 (ADAPT: 665
   against 1300) and Givens compilation (2.43×) are the right kind of result and
   still orders of magnitude short. Qubit-ADAPT pools are the next concrete step.

Everything below is kept for the record and for the domain caveats it carries,
not because it is the next thing to do.

### The measurement side is settled — and Result 55 gave it a domain

Everything below about measurement schemes optimises estimator **variance**, and
that is the right target only where the error is shot noise. Measured
(Result 55): on an ideal simulator 16x the shots buys 3.0x the accuracy; on a
Heron-class noise model at the same depths it buys **4.3%**, because the error
there equals the device bias to three digits. The IQR still falls as sqrt(n)
exactly as designed — the total error simply contains almost no statistics.

So the ranking below **inverts** under device noise: general-commuting grouping
beats QWC 1.7x on LiH ideally and loses 1.37x on `heron`, because its Clifford
basis changes cost 125 two-qubit gates against QWC's zero. Both land at 5-9e-2
Hartree against chemical accuracy at 1.6e-3.

**This is why error mitigation is now item 1 rather than item 4.**

### The ranking itself, which holds where variance is the error
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

### 1. Error mitigation — measured, and readout is the whole story (Result 56)
Done, and it is the first thing in this project to reduce device error rather
than variance. On `heron`, H₄ from the Hartree-Fock reference, every arm charged
the same 200k total shots with calibration paid out of that budget:

| arm | median error | vs unmitigated |
|---|---|---|
| unmitigated | 5.263e-2 | 1.00 |
| ZNE (three variants) | 5.45–5.60e-2 | **1.04–1.06** |
| **readout, 25% calibration** | **5.583e-3** | **0.11** |

**ZNE is worse than nothing**, because 97% of the bias is readout error and
folding amplifies state preparation — two X gates on an HF reference. Diagnosing
the bias before choosing a method is what made this cheap.

**Open, in order:**

1. **Cheaper calibration.** 5.583e-3 is still 3.5× short of chemical accuracy,
   and the decomposition says a perfect correction would reach 1.74e-3. The 25%
   arm beating the 10% arm shows calibration statistics are the limiter. A
   tensored or M3-style approximation buys more shots per calibration point and
   also removes the 2ⁿ wall — exact calibration is hopeless past ~14 qubits.
2. **Take it to a real ansatz.** *Done — Result 59, and the verdict inverts.*
   With `puccd` (317 two-qubit gates) the gate share of the bias goes from 4.7%
   to **95.4%**, readout mitigation drops to 4%, and ZNE becomes worth 3.4× —
   but only once the extrapolation matches the physics. A depolarising channel
   *saturates*, so `E(s) = a + b r^s` is the right form; against the line it is
   worth **3.2× on identical data**, pure post-processing.

   **The rule this gives: decompose the bias first, then pick the method.** Both
   times, choosing by what the literature emphasises picked the wrong one.

   **And the sobering number:** the best mitigated result with a real ansatz is
   0.408 Hartree, 255× chemical accuracy. Adding 317 two-qubit gates costs 26×
   more error than all of Result 56's mitigation recovered. The gate budget is
   the binding constraint — the same conclusion the README's ~32× depth gap
   reached by counting circuits, now confirmed by measuring error directly.
3. **Re-run Result 55's grouping comparison with readout correction on.**
   *Done — Result 58, and the guess written here was wrong.* Readout bias is a
   property of the qubits, the same for both schemes, and was *masking* the gate
   gap rather than creating it. Mitigation widens the ratio from 1.27 to 4.33
   (H₄) and 1.56 to 5.86 (LiH). **The standing recipe is QWC grouping plus
   tensored readout mitigation at 5% of budget: 19× better than the old default
   on H₄, 16× on LiH.**

### 2. ADAPT-VQE: measured, improved 4.6×, and now at cost parity
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

### 3. The size wall, confirmed from both sides
Result 42 (chemistry) and Result 50 (optimisation) reach the same conclusion by
completely different routes: **at the sizes this project can score honestly, the
classical method already wins outright.** FCI solves H₂/H₄/LiH to 12 decimals in
~100 ms; Goemans-Williamson finds the *exact* MaxCut optimum on 100% of
instances up to n = 20 in 66–191 ms. Every ratio measured below those ceilings
is measured on instances with no hard part left.

That makes size the binding constraint on this whole project, not an item on a
list. Both branches need the same thing — a reference that survives past brute
force:

* **Chemistry:** *partly answered — Result 53.* Going past LiH turned up
  something before the rankings could even be re-tested: the `Σ|c| ~ N^2.78`
  law the headline rests on was not identifiable from the five molecules it was
  fitted on. Measured one direction at a time over thirteen, adding atoms at
  fixed nuclear charge gives `N^-0.39` and adding basis functions to a fixed
  molecule gives `N^2.86`. The headline survives (N^9.7), but **qubit count is
  the wrong figure of merit** — Σ|c| spans 6.3× at fixed orbital count.

  Still open: whether the Result 38 measurement-scheme ranking and the Result 47
  ADAPT verdict hold on H₂O / NH₃ / CH₄, which now exist in the molecule set.
  This is the next thing to run, and it is cheap: the Hamiltonians are built.
* **Optimisation:** *answered — Result 51.* Certainty ends between **n = 40 and
  n = 60**: up to 40 the SDP and iterated local search agree on every instance
  (and are both exactly optimal wherever brute force can confirm it); at 60 they
  first diverge; by 100 they disagree on all twelve. So a QAOA number means
  something only at **n ≥ 60** — and the target there is not Goemans-Williamson's
  0.878 guarantee but a fifty-line hill-climber that finds a 2.55% better cut in
  one second. GW never wins once at a matched budget.

### 4. A classical baseline on every result  *(done — Results 42 and 50)*
Both areas the user named now have one: CCSD(T)/FCI for chemistry
(`qres/classical.py`), and greedy / local search / Goemans-Williamson for
MaxCut (`qres/classical_optimization.py`). The README states the classical
answer first. Keep it that way for anything added.

---

## Next

### 5. Larger parameter counts, where the shot-frugal literature should win
The adaptive-optimiser negative result was on 12 parameters (H₂). COBYLA
degrades badly above ~50. Re-test adaptive SPSA on LiH (≈ 100+ parameters)
before concluding it does not help.

**iCANS is impractical here and should not be re-run naively:** it needs `2n`
circuit evaluations per step, which measured at ~3.5 h for 8 seeds of H₄ (46
parameters) and was aborted. If it is worth testing at scale, it needs the
random-operator-sampling variant (Rosalin) that avoids the full parameter-shift
sweep, not the plain version.

### 6. QAOA parameter transfer  *(measured and closed — Result 52)*
Transfer works perfectly and does not help. Angles trained on one 60-vertex
instance, applied to 100 and 200 without re-optimisation, lose **0.000–0.018%**
against re-optimising per instance. The light cone is `n`-independent on a
regular graph, so the angles are too — the fixed-angle literature has a
mechanism, and this repository can now say what it is.

But it closes the direction: QAOA's instance-specific outer loop was never the
bottleneck, so removing it entirely buys 0.018%. At n = 60–1 000 the exact
expected cut is **0.83–0.87 of the classical champion at p=2**, against a 1 ms
hill-climb's 0.95–1.00 — with optimal angles, no noise, and infinite shots. What
limits QAOA here is the depth-`p` ansatz, and no optimiser work touches that.

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
