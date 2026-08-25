# Next Steps

Working queue, highest value first. Each item says what to measure and what
would count as a result, so a session can pick one up cold.

Update this file at the end of every session: tick off what was done, re-rank
what is left, add what the results suggested.

---

## Now

### 0. Compile double excitations as Givens rotations  ← **the concrete lead**
Results 10, 12 and 13 converge on one buildable thing.

The situation is a scissor. The only ansatz that survives real device noise
(hardware-efficient, fidelity 0.61–0.87) has **exactly zero gradient** at
Hartree-Fock — 0 of 46 parameters at 2 reps, 0 of 114 at 6 reps — because a
real-amplitude ansatz's first-order response reaches only single excitations and
Brillouin's theorem decouples those. Correlation energy is in the doubles. So it
sits at Hartree-Fock forever, at any depth. Meanwhile every ansatz that *can*
move is destroyed by noise: PUCCD retains 10% signal on Heron, UCCSD 0.01%.

But the gate counts say the gap is a **compilation** problem, not a physics one:

| | H4 |
|---|---|
| PUCCD parameters (paired double excitations) | 4 |
| Optimal 2q cost of one double-excitation Givens rotation | ~13 CNOT |
| So achievable | **~52 two-qubit gates** |
| What qiskit-nature's `UCC` emits | **317** |

qiskit-nature Trotterises each excitation into eight Pauli-string exponentials,
each with its own CNOT ladder, and the transpiler does not exploit the shared
structure. Compiling the double excitation directly as a 4-qubit Givens rotation
should give ~6×.

**Predicted:** 317 → ~52 two-qubit gates, moving Heron fidelity from 0.099 to
~0.80 — comparable to the hardware-efficient ansatz, while keeping the non-zero
gradient at Hartree-Fock that it lacks. `[unverified]`, a gate-count estimate.

**Done so far** (`qres/fermionic.py`, Result 14): the gate itself is built and
verified — **26 two-qubit gates against qiskit-nature's ~79**, exact to 1e-12
over seven angles, with particle-number preservation, subspace confinement, the
group law and non-zero gradient at the reference all tested.

**Also done** (Result 15): Z-string handling and a full `puccd_shallow`,
verified to reproduce `qiskit_nature`'s `PUCCD` as a unitary to 8e-13 on H₄.
**2.43× fewer two-qubit gates in the abstract count** — but only **1.31×** after
routing on a heavy-hex device.

**Next, and it is now the blocking piece: spin-orbital ordering.** The routing
penalty is 2.11× for this construction against 1.14× for qiskit-nature's,
because the 4-qubit Givens gates act on `(i, M+i, a, M+a)` — scattered across the
blocked-spin ordering — while Pauli ladders run between adjacent qubits.
Interleaving (`α₀ β₀ α₁ β₁ …`) would make each excitation two adjacent pairs and
should recover most of the 2.43×.

The catch, found by attempting it: interleaving changes the Jordan-Wigner
*algebra*, not just the layout — the spin partner then falls *inside* the parity
string. So it needs the strings re-derived and re-verified against a reference in
the same ordering, which `qiskit_nature` does not provide directly. Steps:
1. Re-derive the paired-double parity string under interleaved ordering.
2. Verify by permuting `qiskit_nature`'s Hamiltonian and PUCCD into that
   ordering and comparing unitaries — the same test that caught the missing
   Z-strings.
3. Measure routed gate count and device fidelity against the blocked baseline.

**The gap this has to close, now measured** (Result 16, experiment 005 over the
whole excitation family on H₄):

* reaching chemical accuracy needs **~1300 two-qubit gates** (`ucc-d:1`, 18
  parameters, 1.08e-4 Ha)
* surviving a Heron device at fidelity > 0.5 allows **~40**
* **the gap is ~32×**, and this session's compilation buys 2.43× of it

Two routes are now closed and should not be retried:

* **Repetitions of paired doubles do nothing.** `puccd`, `puccd:2`, `puccd:3`
  give *identical* error (1.631e-2) while tripling gates — the paired-doubles
  manifold is closed under composition.
* **Singles are not worth their cost.** `puccsd` (12 params) reaches 1.589e-2
  against `puccd`'s (4 params) 1.631e-2. Brillouin again.

So expressibility has to come from *unpaired* doubles, which is where the gate
count lives. Realistic remaining levers, in order: qubit ordering (above),
per-excitation compilation below 26 gates, and dropping excitations by measured
gradient magnitude (ADAPT-style) rather than including the full set.

Acceptance test: non-zero gradient at Hartree-Fock, exact-optimisation error
better than PUCCD's 1.6e-2 on H4, and ≤ 80 two-qubit gates transpiled to
`fake_torino`. If expressibility is short of chemical accuracy, add repetitions
(k-UpCCGSD) and re-measure the fidelity trade.

### 1. Push the shot ladder further  ← **the working lead**
`shot_ladder` is the first thing here that reaches chemical accuracy with any
reliability: 5/16 seeds on H₂ at 12.8M shots against SPSA's 1/16, ratio 0.465,
and at hand-set levels 12 wins / 0 losses with p = 0.000 (Result 27). The
diagnosis behind it is solid — precision has to escalate as the run converges,
because the energy differences a model-based optimiser interpolates shrink
quadratically near the optimum while shot noise stays flat (Results 25, 26).

Open work, roughly in order of expected value:

1. **Lower the crossover.** Below ~800k on 12-parameter H₂ the ladder is
   *significantly worse* than SPSA (ratio 1.71, p = 0.021), because two rungs
   cannot both be fed. Ideas: start the ladder from an SPSA warm-start rather
   than cold, so the first rung needs fewer evaluations; or use a cheaper inner
   optimiser for the bottom rungs.
2. **Replace the restart with a proper noise-aware trust region.** Restarting
   COBYLA is a blunt instrument that throws away its model each rung. A
   stochastic trust-region method (STORM-style) that sizes its sample count from
   the model's own uncertainty should dominate it, and would remove the
   hand-set `growth` and `levels`.
3. **Tune `growth` and `evals_per_level` properly.** Only `g ∈ {4, 6, 8}` and
   `E = 2(n+1)` have been tested, on one molecule.
4. **Confirm on a second system.** Everything above is H₂ at 12 parameters.

### 1a. Does the measurement work pay off now?  *(partly answered)*
Session 1 built covariance-aware grouping (0.63 variance ratio on LiH, validated
against 300 sampled estimates) and concluded it had nothing to act on — but that
conclusion came from H₄, whose ansatz was stuck at Hartree-Fock. Result 25 shows
H₂ *is* shot-noise-limited, so the measurement side does bind after all.

H₂ cannot test it (2 groups). The test needs a system that both converges and
has groups worth arranging: H₄ + UCCSD, which reaches 1e-4 under exact
optimisation and has 10 commuting groups. In flight.

**Answered, with a caveat that matters.** At 4M shots the covariance grouping is
significantly better end to end — ratio 0.798, 10/2 seeds, p = 0.039 (Result 30).
That is the first time any of session 1's measurement work has shown a
significant benefit in a full optimisation rather than at a fixed state.

But both arms fail: their errors (0.08–0.11 Ha) exceed H₄'s correlation energy
of 0.042 Ha, so both are worse than stopping at Hartree-Fock. Even after fixing
the trust region (`rhobeg = 0.1` halves the error, Result 31) the best is
4.86e-2 against Hartree-Fock's 4.18e-2. A 20% improvement on a method that is
not working is not a validated benefit.

To finish this:
1. Find the budget at which H₄ + UCCSD actually converges — the exact-energy
   control (Result 25) says the evaluations are there, so this is a shot-count
   question. Expensive, but the 14× fast path (Result 28) makes it feasible.
2. Re-run the grouping comparison there.
3. Chase the unexplained group count: `covariance+refine` produced **19** groups
   through `ShotEstimator` against the **11** experiment 003 reports for the same
   molecule and reference. One of the two paths is not doing what it says.

### 1d. `rhobeg` must scale with the ansatz
No universal value exists: H₂'s hardware-efficient ansatz wants a *large* trust
region on restart (1.0 was 9× better than 0.1, Result 26) and H₄'s UCCSD wants a
*small* one (0.1 was 2× better than 1.0, Result 31). `shot_ladder` currently
hard-codes 1.0. It should estimate the scale — from the spread of a few
random-direction energy differences, or from the ansatz family — rather than
being handed it.

### 1a. Re-validate the measurement studies with enough seeds
Results 3 and 18: the 3.8× allocation win became 1.3× and lost significance at
24 seeds. Everything measured at 4 seeds in this session should be re-run at
≥ 24 before it is believed, and the covariance-grouping win (0.63 variance ratio
on LiH) has still only been validated **at a fixed state**, never end-to-end.

The end-to-end test needs a configuration that actually converges — see item 1.
Attempting it with UCCSD hit the budget rule instead (Result 17).

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

**Size every future budget against the parameter count** (Result 17):

    usable budget  ≳  10 × n_parameters × shots_per_evaluation

COBYLA spends `n+1` evaluations on its initial simplex alone. A 200k budget at
4096 shots gives 48 evaluations for UCCSD's 26 parameters, so it never starts —
which is why an end-to-end test with a *converging* ansatz still produced
1.39e-1 Ha. Below this threshold a comparison measures simplex construction,
not the algorithm.

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
