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

### 1. Why does nothing reach chemical accuracy on H₂?  ← **the live question**
Session 2 closed off two explanations and left the question sharper.

Not the step count: `stochastic_parameter_shift` gets 195–781 gradient steps
instead of COBYLA's four, and is no better (Result 24). Not the optimiser
family either, beyond a point — SPSA beats COBYLA 2× with p = 0.001 (Result 23),
but still lands at 5.8e-3, and **0 of 16 seeds reach 1.594 mHa** at a 200k
budget for *any* configuration tested.

The measured shot-noise floor on H₂ is σ = 7.6e-4 at 200k shots in a single
evaluation, so the budget is not obviously the wall either — but a run spends
its budget across ~100 evaluations, each at σ ≈ 7.6e-3. Two things to measure,
in order:

1. **Budget scaling.** Sweep 10⁵–10⁷ shots for `hea:2` + SPSA, 16 seeds, and
   find where chemical accuracy becomes reliable. If it never does, the limit is
   not the budget.
2. **The exact-energy control at matched evaluation counts.** Run the same
   optimiser on noiseless energies with the same *number* of evaluations. If it
   reaches 1e-6, the gap is shot noise and the answer is better estimators; if
   it plateaus at 5e-3 too, the gap is the optimiser or the landscape.

Only the second answers *which* problem to work on, so do it first — it costs
minutes.

### 1b. Remaining optimiser candidates
Untested: Rosalin's random operator sampling (parameter-shift over a sampled
subset of Hamiltonian terms, so per-evaluation cost stops scaling with the term
count) and quantum natural gradient (expensive per step, far fewer steps —
worth a cost-per-accuracy comparison rather than a dismissal).

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
