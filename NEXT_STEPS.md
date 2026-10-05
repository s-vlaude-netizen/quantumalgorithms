# Next Steps

Working queue, highest value first. Each item says what to measure and what
would count as a result, so a session can pick one up cold.

Update this file at the end of every session: tick off what was done, re-rank
what is left, add what the results suggested.

---

## How to judge a result here

A standing rubric, because without one there is no way to tell a contribution
from a well-measured restatement — and this repository has produced a lot of the
second. Ranked:

**A — new, and proved.** Something not in the literature, demonstrated by a
script that genuinely establishes it. Symbolic tools (SymPy) count and are often
stronger than timing, because an identity in a polynomial ring is not a
coincidence of one random draw. The script must prove *the stated claim*, not a
neighbouring one.

**A′ — refutes a published claim.** Same bar, plus: the script must show both
that my statement holds *and* that the paper's does not. Getting a different
number is not a refutation; it is usually a different setup.

**B — literature implemented 1:1, and beaten in the same environment.** Implement
the published algorithm exactly as specified, time it here at small qubit counts,
then show mine is reproducibly faster *in the same environment*. This does not
prove anything about other scales and must not be written as if it does. It is
still a real data point.

**C — an honest crossover point.** Where a classical method is still faster at
small size, that is a measurement worth having: enough of them draw the curve
that says where each side wins.

**D — a negative result, measured.** Cheap to produce, expensive to re-derive,
and this repository is mostly made of them. Valuable, but not a contribution in
the sense above, and should never be presented as one.

**Below the line:** a constant factor over my own earlier baseline. That is
engineering on my own code, not a result about the world — Result 72 established
that nearly every "speedup" here was a published method correctly implemented.

### Where this repository's results actually sit

Applied honestly, as of Result 95:

| tier | results | note |
|---|---|---|
| **A** | **86, 89** | the THC gauge group, and its exact identifiability boundary; both small |
| **A′** | **93** (narrow) | IonQ + Ansys: every partition the quantum step handed LS-DYNA is computed exactly classically, faster — refutes the press release's "quantum outperforming classical", not the paper's pipeline claim |
| **B** | 47 (arguably), **87**, **90** | batched+lazy ADAPT; the exact O(T) λ reduction, whose goal CDF already covers; THC's λ lowered along exact minimisers at no accuracy cost |
| **C** | 42, 50, 51, 85 | classical wins, with the size where it stops (85's *reading* corrected by 88) |
| **D** | most of the rest | 55, 66, 68, 72, 74, 75, 81, 82, 83, 84, 88, 91, 92, 94, 95 … |

**That is two A and one narrow A′ in ninety-five results**; the two A are both on one model, and Result 72 had already said so from
the other direction. The rubric is not a scoreboard to improve; it is a filter
to apply *before* starting something, because tier-D work is much easier to
begin and this file is where the choice gets made.

**What it implies for what to do next.** Tier A needs either a structure nobody
has written down or a claim someone has written down wrongly. The two places in
this repository where that is plausible:

* **the nonlinear THC fit**, which is *not* the published LS-THC (that one fixes
  χ on a grid and solves linearly, so it has no gauge freedom at all). Anything
  structural about the joint optimisation is unexplored territory by default.
  Result 87 took the obvious next step — the same question asked of double
  factorisation — and landed at B, because the *goal* there is published even
  though the exact route is not. **That is the trap to watch: a structural
  question can be genuinely open while the thing it optimises is crowded.**
  Result 89 stayed on THC and found the exact boundary of its flat directions.
* **the sign problem's onset** (Results 85, 88), where a measured boundary
  between polynomial and exponential on real molecular Hamiltonians would be a
  genuine A. Result 88 showed Result 85's walker thresholds do not measure it
  (they move 4–8× with run length; H₂ has no sign problem at all), and put a
  deterministic sign gap through H₈ in their place.

---

## Now

### NEW — public quantum-advantage claims as refutation targets (added on request, October 2026)

**Why this belongs here:** tier **A′** — *refutes a published claim* — is the one
rung of the rubric this repository had never reached, until Result 93 reached it, narrowly, from this list. Press-release advantage
claims are the natural supply. Many have since been answered by better classical
algorithms: one survey (arXiv:2607.07530, "The NISQ Trap") counts more than
thirty such announcements and finds all but one reproduced or explained
classically within eighteen months. The bar is unchanged: a refutation must show
that our statement holds **and** that theirs does not. Getting a different number
is not a refutation.

Researched October 2026, ranked by fit to this repository (4 cores, 15 GB,
exact simulation up to ~20 qubits, strong on baselines and statistics):

| claim | status | route to a refutation | fit here |
|---|---|---|---|
| **IonQ + Ansys, March 2025** (arXiv:2503.13128): "quantum outperforming classical", LS-DYNA up to **12% faster** | **refuted here — Result 93 (A′, narrow)** | The quantum step bisects graphs coarsened to **≤ 32 vertices**, and the 12% run is the best by wall clock among the **10 lowest-energy partitions**. Enumeration computes that whole list exactly (≤ 87 s at 32 on a shared core), an MILP proves the balanced optimum in < 0.6 s, and the paper's own FM refinement from random starts reaches the optimum in 54 of 54 instances at n ≥ 20. Same input to LS-DYNA, no quantum computer. The paper's pipeline-versus-LS-GPart claim stands, as a classical pipeline. | done; follow-up below |
| **IBM tracker, July 2026**: Qedma's Floquet Ising, 74 qubits on heavy-hex (arXiv:2607.24937) | open, fresh | Heavy-hex is tree-like, which is where belief-propagation tensor networks and Pauli propagation reproduced IBM's 2023 "utility" result on a laptop. The claim rests on *late* times and on an error-mitigation extrapolation with no accuracy bound. | possible, research-grade |
| IBM tracker, July 2026: Algorithmiq's Loschmidt echo, 56 qubits (arXiv:2607.25998) | open | Classical methods disagree with each other; the rescaling heuristic has no error bound. Algorithmiq released `monoprop` to invite challengers. | hard |
| IBM/UChicago, July 2026: doped-Clifford sampling, 97 qubits, 468 T gates (arXiv:2607.25941) | open | Stabiliser-rank cost grows exponentially in the T count, unless the T placement allows cutting. | hard |
| Google "Quantum Echoes", OTOC(2), 65 qubits, "13 000×" (Nature, Oct 2025) | open as far as found | Tensor-network or Pauli-path simulation of deep echo circuits. | hard |
| Quantinuum Helios, random circuit sampling, 98 qubits (Nov 2025) | open | Tensor-network contraction at 98 fully connected qubits. | out of reach |
| D-Wave spin-glass dynamics (Science, March 2025) | **contested** | Reproduced by Flatiron/BU belief propagation in Science (May 2026). D-Wave answers that the 3D cubic and diamond lattices, the largest sizes and the fourth-order observables were not reproduced. | the remaining gap is 3D, heavy. SQA cannot attack it: it samples Monte Carlo dynamics, not Schrödinger dynamics. |
| **D-Wave/USC "scaling advantage in approximate optimization"** (Munoz-Bauza & Lidar, PRL 134, 160601, 2025) | **open — in progress as exp041** | QAC: median time-to-ε at a 1% gap scales as `N^1.69 ± 0.12`, against PT-ICM's `N^1.93 ± 0.03` — "the first demonstration of an algorithmic quantum speedup in approximate optimization". PT-ICM was the only classical opponent; SA was dropped as "not competitive". At a fixed relative gap an annealer's energy density self-averages, so a tuned classical annealer should approach `N^1`. The instance class is rebuilt from Pegasus P16, and SA, SQA and PT-ICM are running. | **good fit; second A′ candidate** |
| Kipu Quantum, "runtime quantum advantage" in optimisation (May 2025) | **refuted by others** | arXiv:2510.06337: a better classical baseline removes it. Kipu's own March 2026 benchmark concedes that classical solvers "reach or surpass" it. | done |
| "Peaked circuits" verifiable-advantage proposals | **refuted by others** | arXiv:2604.21908: efficient classical simulation. | done |
| Microsoft: Majorana 1 (Feb 2025); logical qubits with Quantinuum and Atom | not an advantage claim | These are hardware claims. Microsoft itself says the chemistry demonstration "does not demonstrate scientific quantum advantage". A classical algorithm cannot refute a qubit-physics claim. | out of scope |

**Top item done — Result 93.** All four plan steps ran: VarQITE as specified,
FEA-type meshes coarsened METIS-style, and the exact, FM, METIS and spectral
comparison. An exact classical partition of the same coarse graph is the
quantum step's output, so the 12% is a property of the pipeline.

**Follow-up done — Result 95 (D).** Same meshes, about four times larger, with
fill measured by symbolic Cholesky. METIS nested dissection is the cheapest
ordering on 9 of 9 instances. The exact-32 top split (the quantum step's best
case) costs 1.000–1.118× its flops, median 1.067. Against METIS's own bisection
through the same construction, the exact-32 split ties (median 1.001). The 10
lowest-energy partitions collapse to 1–3 distinct orderings after refinement.
Still open on this claim: LS-GPart itself, which is not available here.

**Next refutation target: Qedma's Floquet Ising (arXiv:2607.24937).** It is the
only other row that fits here, and the paper read on 2026-10-03 says what to try.

* **The model:**
  `U_F = Π_{r=1..3} e^{−iθ_zz C_r/2} e^{−iθ_z Z_Σ/2} e^{−iθ_x X_Σ/2}`
  over the three edge colours `C_r` of heavy-hex, with `θ_x ≈ π/6`,
  `θ_z ≈ π/27`, `θ_zz = π/3`. The start is `|0…0⟩`, up to 30 cycles, at 51 and 74
  qubits. The observable is the magnetisation; the claim is a long-lived
  oscillation of period ≈ 4 cycles, with "a nonzero asymptotic oscillation
  amplitude in the thermodynamic limit".
* **Their classical baselines fail by dynamics:** PEPS-BP (D = 700) at about 12
  cycles, sparse Pauli paths (W = 20) at about 15, TEBD (χ = 4096) with
  non-monotonic size scaling.
* **The route they did not take is statistical mechanics, not dynamics.**
  `3 θ_x = π/2`, so each cycle is a π/2 rotation about X times a weak remainder.
  The period-4 oscillation is the free precession; what is claimed is that its
  *amplitude* survives. In the toggling frame the leading-order effective
  Hamiltonian averages `ZZ → (ZZ + YY)/2` and the `θ_z` field to zero. That is an
  easy-plane magnet whose exact symmetry is the `Z₄` of the π/2 kick (U(1) at
  leading order). Prethermalisation theory says the plateau is the Gibbs state
  of that Hamiltonian at the initial state's energy.
* **Steps.** (1) Build `H_eff` to second order (Floquet–Magnus in the toggling
  frame) and check it against exact dynamics at 12–21 qubits. (2) Compute the
  plateau magnetisation from the Gibbs ensemble at the initial energy: exact at
  ≤ 21 qubits, and by sign-free QMC (stochastic series expansion on the
  easy-plane ferromagnet) at 51, 74 and the thermodynamic limit. (3) Compare
  with their mitigated 30-cycle data.
* **What would count.** If the Gibbs prediction matches their data, their physics
  is classically computable without simulating dynamics: a refutation of
  "classical methods fail" for the quantity they report. If the leading-order
  `H_eff` is U(1)-symmetric at finite temperature, Mermin–Wagner forbids a
  nonzero amplitude in the 2-D thermodynamic limit. Their asymptotic claim would
  then rest entirely on the `Z₄`-breaking higher-order terms, which is testable
  in the same framework. Research-grade, and the Gibbs step is the cheap first
  check.
* **Read more closely on 2026-10-03, which lowers the value.** The paper already
  draws the prethermal *plateau* from a fourth-order Magnus effective
  Hamiltonian. So a Gibbs plateau would confirm their own estimate, not refute
  anything. What is left is the oscillation *amplitude* in the thermodynamic
  limit, `M(N_c) = C e^{−γN_c} + A cos(2πN_c/T + φ)`. That is an equilibrium
  ordering question for `H_eff`. The leading order checks out (first-order
  average: `(θ_zz/4) Σ (ZZ + YY)`, the `θ_z` field averages out), so the
  leading order is U(1). But a 4-fold anisotropy is relevant in the 2-D
  low-temperature phase (José–Kadanoff–Kirkpatrick–Nelson), so true `Z₄` order
  and a nonzero `A` are *plausible*. Mermin–Wagner does not refute it. The likely
  outcome of this route is to *confirm* their physics classically, by QMC on
  `H_eff`, possibly with a sign problem at fourth order. That still answers
  "classical methods fail", but at high cost. The exact angles are given only
  as "≈", and the data only as figures. **Ranked below the Result 93 follow-up.**

### NEW — building on Troyer's critical programme (added on request, October 2026)

Matthias Troyer's group wrote the methodology this repository has been
reinventing. What to take from it, and where:

* **Rønnow et al., "Defining and detecting quantum speedup"** (Science 2014,
  arXiv:1401.2910). Scale hardware resources identically on both sides. An
  analog device with N qubits sweeps in constant time, so either divide the
  serial classical time by N, or charge the device N/N_max for unused qubits.
  *Checked for the D-Wave/USC claim: they charge N/N_max, so the timing
  convention is fair* (exp041).
* **Heim, Rønnow, Isakov & Troyer, "Quantum versus classical annealing of Ising
  spin glasses"** (Science 2015, arXiv:1411.5693). Discrete-time SQA with
  best-of-slices readout manufactures a scaling advantage over SA that vanishes
  in the continuous-time limit with physical readout. *Applied in exp041*: SQA
  is reported with both readouts, and only the single-slice one may stand in
  for an annealer. Follow-up: a continuous-time SQA arm.
* **Isakov, Zintchenko, Rønnow & Troyer, "Optimised simulated annealing for
  Ising spin glasses"** (CPC 2015, arXiv:1401.1084). Its multi-spin-coded
  codes run at about 0.1 ns per spin update; exp041's numba kernels run at
  12–17 ns. That changes absolute times, not exponents; quote it whenever an
  absolute "×-faster" is compared.
* **Hoefler, Häner & Troyer, "Disentangling hype from practicality"** (CACM
  2023, arXiv:2307.00523). Quadratic speedups do not survive realistic
  overheads; small data and super-quadratic speedups are required. *As a
  filter for refutation targets*: a polynomial exponent gap like D-Wave/USC's
  `N^0.24` cannot be practical, even before exp041. Their own exclusion of
  readout time — "can reach 200 μs per sample", and it "scales with problem
  size" — against annealing times of about 1 μs is exactly the overhead that
  paper says must be counted.
* **Beverland et al., "Assessing requirements to scale to practical quantum
  advantage"** (arXiv:2211.07629). It provides the resource-estimation
  framework. This repository's THC/DF results (76–80, 94) feed exactly that
  kind of estimate, and Result 82's 25-orbital target should be re-costed with it.
* **Goings et al., cytochrome P450** (PNAS 2022) and **Reiher et al., FeMoco**
  (PNAS 2017). These are the models of a careful classical-against-quantum
  comparison in chemistry: DMRG/CCSD(T) baselines next to fault-tolerant
  resource estimates. They are the template for the still-open DMRG baseline
  item below.
* **Carleo, Bauer & Troyer, "Simulating adiabatic quantum computation with a
  variational approach"** (arXiv:2403.05147, work from 2016). Time-dependent
  VMC with Jastrow states reproduces annealing dynamics on Chimera spin
  glasses. That is the route EPFL used against D-Wave's 2025 dynamics claim,
  and the only one in this list that can address coherent dynamics rather than
  sampling.
* **The more optimistic recent papers**: "Quantum computers will not be that
  different" (Hoefler & Troyer, arXiv:2609.19639) and "Assessing the benefits
  and risks of quantum computers" (arXiv:2401.16317). These argue engineering
  and cost-performance, not new speedups, so there is nothing to refute. Use
  them for the cost model: a QPU judged as a heterogeneous accelerator,
  quantum-classical I/O included.

### NEW — Google "Quantum Echoes" (OTOC(2)): is the *observable* classically approximable?

Prompted by the "a physical system just measuring itself" argument. Solid-state
NMR has measured OTOC-type echoes on clusters of thousands of nuclear spins for
a decade (Álvarez & Suter; Pastawski's group; arXiv:2504.15183). By "cost of
an exact classical simulation of the measured quantity" that exceeded 65 qubits
long ago, and nobody called it advantage. One reason: Elsayed & Fine
(arXiv:1409.8564) showed that classical spin simulations reproduce such NMR
signals when each spin has many neighbours. The same reason cuts the other way
for Google: the agreement degrades at four or fewer neighbours, and Google's
grid has four. **Testable here at 20–24 qubits:** exact OTOC(1)/OTOC(2) on
Google-style grid circuits against truncated Pauli paths and classical-spin
(TWA) estimates, mapping where each fails. A refutation would need the
observable to stay accurate where Google says it does not. A confirmation is
the likelier outcome and would still be worth recording (tier C).

### NEW — classical algorithm work is in scope, and it is already where the leverage was

Added as an explicit direction. It is worth stating what this repository's own
record says about it, because "classical algorithms" sounds like a detour from
quantum algorithms and here it has not been one:

**The three largest factors ever measured in this repository are classical
algorithms.** Double factorisation as a block encoding (332×, Result 76) and
tensor hypercontraction (Results 77–80) are *classical* preprocessing of the
Hamiltonian that reduces the quantum cost. Result 80's gauge fixing was a
**2× classical optimisation improvement** — removing a flat direction from a
least-squares landscape — and it beat every error-mitigation and
measurement-scheme result here. None of it touched a qubit.

And the inverse holds: **a better classical algorithm directly tests criterion 1.**
The whole framework is (is there a classical gap?) × (does the hardware reach
it?), so improving the classical side is not competition with the quantum
programme, it is the measurement that decides it.

So, ranked by what would actually move something here:

1. ~~A probabilistic classical baseline~~ — **built, Result 85; its reading
   corrected by Result 88, which changes what to measure next.** FCIQMC exists
   here and is validated against exact diagonalisation. But the walker threshold
   at fixed step count is a *statistical* cost: 4× the steps cuts it 4–8×, and H₂
   — which pays 32× when stretched — has **no sign problem at all** (a 2-determinant
   sector, annihilation exactly zero). The "38% against 0.3%" annihilation compares
   thresholds with 64–128× different walker counts; at equal counts it is 1.3–3×.
   The deterministic **sign gap** (`E0(H) − E0` of the sign-free matrix) is now
   computed through H₈ and grows with size at *both* geometries: 0.033 → 0.383 →
   2.297 Ha at equilibrium, 0.699 → 1.911 → 4.446 stretched.

   **But the complexity class is not measured, and the reason is instructive.**
   Three points over N = 2…6 on a factor-two walker ladder cannot separate
   `N^1.90 ± 0.08` from `2^(0.75 N)` — both fits land inside the ladder's own
   ±0.35 log-space resolution. The tight-looking error bar comes from three
   well-behaved points, not from resolution.

   **~~The next step is H₈ and H₁₀ stretched~~ on the same observable** —
   superseded by Result 88: extending a run-length-dependent quantity to more
   points cannot decide a complexity class. **The next step is the annihilation
   plateau** (the population at which, under a fixed shift, growth stalls until
   the sign structure is resolved — Spencer, Blunt & Foulkes 2012) for H₄…H₈ at
   both geometries, which is run-length independent and is the sign problem's
   actual cost; and the sign gap for H₁₀ (sector ~6×10⁴, one sparse eigensolve).
   Two cautions from building Result 88: detect the plateau over seeds, since on
   20-determinant sectors it is tens of walkers and noisy; and for H₈ drop
   couplings below 1e-10 before taking the sector — exp030's 1e-12 threshold
   joins eight blocks through the residue of 12-decimal coefficient rounding.
   **Still the highest-value measurement in this file**, now with the right
   observable.

2. **A DMRG baseline** — still wanted, and now second rather than first. It is
   the deterministic counterpart to the above and the competitor Result 82's
   25-orbital target actually faces. Without it no advantage claim at that size
   means anything, but FCIQMC extended to H₈/H₁₀ answers the same question more
   cheaply and is already written.
3. **More of the THC fit.** Gauge fixing took H₆ from 46 337 to 23 738 iterations
   and the λ spread from 9.52 to 1.59. **Result 89 says where the remaining
   flatness is**: past `M = N(N−1)/2 + 1` (H₄: 7, H₆: 16) the columns slide along
   exact minimisers on which λ varies — and every rank this repository fitted is
   past it. **Result 90 used that**: an unpenalised fit lowered along its exact
   minimisers reaches the penalised λ (8.2004 against 8.1995 on H₆ at M=18, from
   all 8 starts) with the tensor unchanged to 3e-15, so the penalty and its 47×
   energy-error cost are no longer needed there. Open, in order:

   * **re-measure the rank exponent (exp029) with unpenalised fits plus
     lowering**, dropping H₂ (`M ≥ P = 3` makes every tensor exactly
     representable) and reporting which points sit past the boundary;
     **done as exp038, Result 94.** Fine-grid, unpenalised thresholds: **3, 7,
     10, 16, 20 for H₂…H₁₀** (Result 78's 4, 8, 18 superseded). They track the
     `2N − 1` significant ERI pair-matrix eigenvalues, which are the chain's
     on-site and nearest-neighbour pair densities (94–99% of the eigenvector
     weight). The prediction recorded beforehand, H₁₀ at 19 ± 1, held at its
     upper edge, but only with a 600 000-iteration budget; at 150 000 it was 21.
     At the count itself (H₈ M = 15, H₁₀ M = 19) fits stall 40–60× above the
     Eckart–Young floor even with four times the budget, so the largest chains
     need `2N`. Open: (a) why THC cannot follow the floor at the count. Is it
     the rank-one column constraint, or local minima? An exact-rank test like
     Result 89's would separate them. (b) Does any of this survive away from
     hydrogen chains? The mechanism (locality) predicts not.
   * **saturation, `M ≥ P`**: at fixed `χ` the λ minimum is a linear programme
     (one H₄ M=12 fit: 5.217 → 4.655, tensor unchanged to 3e-15); moving `χ`
     jointly is open;
   * **the boundary rank's finite alternatives** at larger `N` — H₆ at M=16 has
     725 739 exact representations on one fit, too many to enumerate as `N`
     grows, so a smarter search is needed there;
   * a proper second-order method (the Gauss-Newton structure is right there and
     unexploited) and a better initialisation than perturbing the DF selection.
4. **Factorisation: honest scoping first, and the answer is probably "size it,
   don't build it".** Competing with GNFS implementations (CADO-NFS, msieve) is
   not realistic here and would not be useful if it succeeded — factoring's only
   application is breaking cryptography that is already being retired. What *is*
   cheap and worth doing is the **cross-check**: size Shor on RSA-2048 with this
   repository's own surface-code model (Results 69–71) against Gidney & Ekerå's
   published ~20M qubits / 8 hours. Result 82 did exactly this against Lee et
   al.'s T-counts and the agreement was the strongest evidence that the resource
   model is sound. A second independent anchor, on a completely different
   algorithm, is worth more than either estimate alone.

**What to avoid:** building a classical solver for a problem where the classical
side already wins outright (MaxCut, small-molecule FCI). Results 42 and 50
measured those; a faster classical answer to a question already answered
classically changes nothing.

### NEW — drug metabolism at a 25-orbital active space (Results 82, 83)

**This is the best-posed advantage candidate this project has had, and it is the
first item where a caveat points in the favourable direction.**

The question came in as: given a ~50-logical-qubit machine, simulate just the
pocket of a drug molecule that touches its target and learn something about
pharmacokinetics. Three translations were needed and two of them matter:

* **50 logical qubits is 25 spatial orbitals**, not 50 atoms — a 50-atom fragment
  is ~200 orbitals, so ~400 qubits. What 25 orbitals buys is an *active space*.
* **Pharmacokinetics is mostly not electronic structure** — ADME is conformational
  and solvation thermodynamics over thousands of atoms. **One exception decides
  clearance and half-life: cytochrome P450 metabolism**, whose Compound I is a
  high-valent iron-oxo species with real multireference character. That is a
  published advantage candidate (Goings et al., PNAS 2022, arXiv:2202.01244).
* The hardware gap, sized with this repo's own laws: **1.1e7× on the variational
  route, 460× in physical qubits on the error-corrected route.** The second is
  the smallest gap ever recorded here, and it is an *upper* bound — a surface
  code is a 2D nearest-neighbour code and an all-to-all machine does not need
  one.

**The blocking measurement is a classical baseline, and this repository does not
have it.** Exact diagonalisation is out past ~24 orbitals (2.7e13 determinants),
but exact diagonalisation is not the competitor — **DMRG is**, and Result 83
measured that these states are substantially compressible: bond dimension grows
at `2^(0.215 n)` at 1e-3 discarded weight and `2^(0.413 n)` at 1e-9, against a
maximal `2^(0.5 n)`. So the gap *opens* near 25 orbitals; it is not established.

In order:

1. **Get a strongly-correlated system into the molecule set.** Everything here is
   a hydrogen chain or a small closed-shell molecule — quasi-1D and weakly
   correlated, the best case for MPS and the worst case for showing a gap. A
   transition-metal active site is the whole point and none exists here.
2. **Measure where CCSD(T) and FCI diverge** on that system. This repo has both
   (`qres/classical.py`). The size at which a single-reference method starts
   disagreeing with the exact answer *is* the multireference onset, and it is
   measurable with tools already present.
3. **Then, and only then, a DMRG baseline.** Without it no advantage claim at 25
   orbitals means anything, and with it this becomes the one direction in the
   repository where the classical side might genuinely fail.

### NEW — trapped ions: measured, and it is 2.17× (Result 81)

Largely closed. All-to-all connectivity removes the heavy-hex routing overhead,
which is **1.35× and saturates** around 1.4; the per-gate error is **1.61×**
better; the product is **2.17×**, and 0 of 10 configurations land inside chemical
accuracy — the same verdict as on superconducting hardware.

What remains open, in descending value:

* **The logical two-qubit error rate is unpublished.** That is the number the
  whole error-corrected route needs, and until it exists the "48 logical qubits"
  figure cannot be sized. Watch for it the way item "watch one number" watches
  the median two-qubit error.
* **A trapped-ion *noise model*.** Every mitigation result here (Results 56, 59)
  is on IBM channels, and the bias decomposition that drove those conclusions —
  97% readout on a shallow circuit, 95% gate on a deep one — would be different
  where SPAM is 99.99%. Cheap to test, and it could invert a recommendation the
  same way Result 58 did.
* **Not** worth re-running the whole noise suite for 2.17×.

### NEW — is classical simulation of a quantum computer really exponential? (Result 83)

Asked as: can you not just use complex numbers and pseudorandomness in Python?

**For a general circuit, no — and it is memory, not cleverness.** A statevector is
2ⁿ amplitudes; 50 qubits is 18 PB. That is exactly what a statevector simulator
is, and it is what every result in this repository was produced on.

**But the general case is the wrong question**, and Result 83 measured the right
one. Clifford circuits are polynomial at any width (Gottesman–Knill), and states
with bounded entanglement are matrix product states — which is what DMRG
exploits. Measured on exact ground states, the bond dimension needed grows
**exponentially at every truncation, but with an exponent that depends on how
much error you accept** (43% to 83% of the maximal rate). Four points and a ±0.23
exponent do not support either "classically easy" or "classically hard".

Open:

* **Extend the series past H₈.** Four points is not a scaling law. H₁₀ exists in
  the molecule set and sparse diagonalisation reaches it.
* **Optimise the orbital ordering.** The measurement takes cuts in the mapper's
  ordering; DMRG does not. The current numbers are an upper bound of unknown
  tightness, and a reordering pass would say by how much.
* **Run it on something not quasi-1D**, which is item 1 of the drug-metabolism
  direction above. These two are the same measurement on different inputs.

### NEW — quantum machine learning: LLMs are closed, sampling is not (Result 84)

**Split verdict, and the split is the useful part.**

*Large language models: a confident no.* Amplitude-encoding a general vector of
dimension `d` costs `gates = d − 11` — measured, `d^1.037 ± 0.007` on the
asymptotic half. Loading is linear, reading classically is linear, so the load
step alone costs what the whole classical algorithm costs. One 4096×4096 weight
matrix is 1.7e7 two-qubit gates; a 70B model is 6.9e10. **No hardware generation
changes this**, and it is the same barrier dequantization attacks from the other
side.

*Machine learning generally: not settled, and the honest answer is to say so.*
The proposals that avoid the input barrier are the ones where the input is
**generated rather than read**:

* **Quantum circuit Born machines** — the model *is* the measurement
  distribution. Sampling from IQP/QAOA families is classically intractable up to
  multiplicative error under standard assumptions (Coyle et al., npj QI 2020).
  This is the strongest complexity-theoretic footing anything in this file has.
* **Quantum reservoir computing / extreme learning machines** — fixed, *untrained*
  quantum dynamics plus a classical linear readout. "Stochasticity as the
  resource, no trained quantum parameters" is the defining property, not a
  workaround.

What this repository could measure, in order:

1. **Whether a Born machine beats a classical generative baseline on data it did
   not generate.** Result 68's kernel failed exactly here — it won only on
   labels its own circuit produced — so the experimental design must fix the
   dataset *first*, from a real source, and only then pick circuits. That
   ordering is the whole methodology.
2. **Where the sampling-hardness argument starts to bite.** It is asymptotic;
   at the ≤20 qubits this repo can run, the distributions are trivially
   classically reproducible because we *simulate them to produce them*. So the
   honest statement is "cannot be tested here", and establishing the qubit count
   at which it could be is itself a result.
3. **Quantum reservoir computing is the cheapest thing on this list to try** —
   no trained quantum parameters means no optimiser, no shot-noise gradient
   loop, and no barren plateau, which removes three of the four failure modes
   this repository has already measured. It needs a temporal dataset and a ridge
   readout, both of which are a day's work.

#### Lead: the Ising Born machine (added on request; step 1 done, Result 92)

The concrete Born machine with the firmest footing, and the one to build if this
direction is taken up: Coyle, Mills, Danos & Kashefi, *"The Born supremacy:
quantum advantage and training of an Ising Born machine"*, npj Quantum
Information 6, 60 (2020).

**The model.** Hadamards on every qubit, one **commuting Ising layer**
`exp(i Σ_{i<j} J_ij Z_i Z_j + i Σ_k b_k Z_k)`, a layer of single-qubit
rotations, then a computational-basis measurement. The trained parameters are
`J`, `b` and the final angles. The final layer is not decoration: without it the
diagonal Ising phases leave every bitstring at probability `2^-n` and the model
cannot represent anything. Two settings of that layer are the known hard
families: final Hadamards make it **IQP** (Bremner, Jozsa & Shepherd 2011), and
final X-rotations make it **QAOA at p = 1**. Exact or multiplicative-error
sampling from either is classically intractable unless the polynomial hierarchy
collapses; additive-error hardness needs anticoncentration plus an average-case
conjecture.

**Why it fits this repository better than anything else in the QML list:**

* **The input is generated, not read**, so Result 84's loading barrier does not
  apply.
* **It is shallow**: one Ising layer is `n(n−1)/2` two-qubit gates all-to-all,
  and they commute, so routing and compilation choices are free to reorder them.
  `ZZ` couplings are native on trapped ions (Result 81), where all-to-all
  connectivity removes the routing overhead entirely.
* **Training need not touch the quantum device.** For IQP-type circuits, every
  Pauli-Z correlator `⟨Z_S⟩` of the output is an average of unit-modulus phases
  over uniformly random bitstrings. Classical Monte Carlo estimates it to
  additive error (Van den Nest's probabilistic simulation, 2011). A
  Gaussian-kernel MMD loss is a weighted sum of exactly such correlators, so it
  and its gradient can be computed classically while *sampling* stays hard. A
  2025 preprint (Recio-Armengol, Ahmed & Bowles, *"Train on classical, deploy on
  quantum"*) builds on this — **not yet checked here; verify before quoting.**
  That removes the shot-noise gradient loop and the optimiser fragility that sank
  this repository's variational experiments.

**What would kill it, stated before starting:**

* **The metric that makes training classical can also make the comparison
  classical.** If the loss only sees low-order correlators, a classical model
  fitted to the same correlators (a pairwise max-entropy / Boltzmann model with a
  matched parameter count) may score as well on that loss. Any advantage has to
  show up in something the loss does not directly fit: held-out likelihood at
  small `n`, or higher-order statistics of the data.
* **Noise.** IQP sampling with constant noise per qubit becomes classically
  simulable once anticoncentration holds (Bremner, Montanaro & Shepherd 2017).
  The device noise models here can measure where that happens.
* **Result 68's trap.** A model winning on data from its own circuit family
  proves nothing, so the dataset is fixed first, from a real source, before any
  circuit is chosen.

**What to measure, in order, and what would count:**

1. **Build it** in `qres` (Qiskit, ≤ 20 qubits, exact probabilities available)
   with the classical baselines alongside: a pairwise max-entropy Ising model
   with the same `J`, `b` count, and an autoregressive or RBM model. Same
   dataset, same parameter budget, **held-out** log-likelihood and total
   variation. A win on held-out real data is **B/C**; a loss is a **D** worth
   having, because it names which baseline closes the gap.

   ✅ **Done — Result 92, and it is a D.** On 16-bit UCI digits, with exact
   held-out likelihoods and L2 chosen on validation, **an RBM with the same 152
   parameters beats the Born machine on 10/10 splits** (median +0.17 bits).
   That is not a matter of starts: 12 starts instead of 3 leave the gap as it
   is. On the same Ising energy, Born against Gibbs is a tie on typical splits.
   **The IQP setting, where sampling is provably hard, is worse than
   independent bits** (8.25 against 7.79). Unregularised MLE had shown a
   spurious Born advantage — couplings diverge for the classical models — and
   that run was discarded. **Steps 2 and 3 below are downgraded**: they only
   pay if a dataset fixed in advance shows the Born machine winning, and picking
   one after the fact is Result 68's trap.
2. **The noise crossover**: total-variation distance between the noiseless and
   device-noise distributions against `n`, on this repository's IBM-device and
   trapped-ion models. This is NEXT_STEPS' "where sampling hardness starts to
   bite" item made concrete — a **C**, with the qubit count and error rate where
   the hardness argument can and cannot still apply.
3. **The proof-type angle**, which is where this repository has produced its A
   results. For the IQP and p = 1 settings, correlators have closed forms
   (products of cosines of the couplings), so the MMD loss and its gradient are
   explicit trigonometric polynomials in `J` and `b`. The exact gradient variance
   as a function of `n` and kernel bandwidth is then a SymPy computation, and it
   would say *provably* where training is and is not trainable. **Check the
   trainability literature first** (MMD-kernel barren-plateau results exist for
   generic circuits); an exact statement for the Ising family is an **A**
   candidate only if it is not already there.

#### Two 2025–26 industry claims about quantum + LLMs, read against Result 84 (added on request)

Both were read in full (arXiv HTML) before being placed here. Neither is a lead
for an advantage. One of them is a cheap, high-tier *test*.

**Multiverse Computing, *"Talking to a quantum computer: quantum hardware inside
a production large language model"* (Aizpurua, Singh, Kshetrimayum, Jahromi,
Orús; arXiv:2605.05914, May 2026).**

* **What they did:** inserted "Cayley-parameterised unitary adapters" into a
  frozen projection layer of Llama 3.1 8B. These are 1 024 independent 4×4
  orthogonal blocks (2 qubits each), 6 144 parameters in total, **trained
  entirely classically**. The quantum processor (IBM Heron r2) only runs the
  trained blocks:
  * each 4-dimensional slice is amplitude-encoded;
  * it is measured with 8 192 shots;
  * the output magnitudes are taken from the counts, with **signs restored from
    the classically stored input**, because counts carry no signs.
* **Numbers they report:** perplexity 8.877 → 8.752 (−1.4%) noiseless and 8.759
  with device noise. One full sequence takes 1 328 circuits and about 4 h 24 min.
  With 3 qubits per block, noise raises perplexity 35-fold.
* **The authors' own position:** the circuits are classically simulable and no
  advantage is claimed.
* **Read against this repository:** this is Result 84's barrier, measured by
  someone else.
  * Each block is a 4×4 matrix-vector product: 16 multiply-adds on a CPU against
    about 4 s and 8 192 shots on the QPU.
  * **The architecture has no exponential headroom.** A block cannot exceed the
    hidden size `d = 4096 = 2^12`, so 12 qubits is the ceiling, and a 4096×4096
    product is about 1.7e7 flops.
  * Reading `2^n` amplitudes from counts costs about `2^n/ε²` shots and returns
    no signs at all.
  * The −1.4% belongs to a classically trained 6 144-parameter adapter. Their
    noiseless arm *is* the classical computation, and the QPU adds noise to it.
* **What it adds here:** an external data point for Result 84 — the
  noise wall at 3 qubits per block on current hardware. Nothing further to test.

**IonQ, *"Quantum Large Language Model Fine-Tuning"* (Kim, Mei, Girotto, Yamada,
Roetteler; arXiv:2504.08732, 2025), blog "Supercharging AI with quantum
computing".**

* **What they did:** put a parameterised-circuit classification head with data
  re-uploading on top of frozen SetFit sentence embeddings (768-dimensional).
  The task is SST-2 sentiment in a low-data regime: 512 training sentences,
  9 101 test sentences. The heads have 10–18 qubits and run **in simulation
  only**.
* **Claim:** 92.70% for the best quantum head (14 qubits, 886 parameters) against
  89.56% for the best classical baseline (an SVC), "up to 3.14%".
* **Not reported:**
  * random seeds, error bars or a significance test;
  * the quantum figure is the best of a hyperparameter screen (5 learning rates
    per configuration across several architectures), and how it was selected is
    not stated;
  * the classical heads are not parameter-matched (logistic regression 769
    parameters, MLP 148 034);
  * any run on hardware.
* **Read against this repository:** this is the shape Results 67–68 measured — a
  classically simulable circuit on classical data. Re-uploading circuits compute
  truncated Fourier series of their inputs (Schuld, Sweke & Meyer, 2021). So the
  fair control is a classical head of about 900 parameters, a small MLP or
  random Fourier features, under the *same* selection protocol.
* **Testable here, and an A′ candidate under the rubric:**
  * reproduce their quantum head (a 14-qubit statevector is cheap) on the same
    embeddings and split;
  * add parameter-matched classical heads;
  * select on validation only;
  * report every arm over many seeds with confidence intervals.

  If a matched classical head reaches about 92.7% under that protocol, the
  claimed improvement is a baseline artefact. If it does not, that is a real
  data point in the quantum head's favour. Both outcomes are worth having.
  HuggingFace (SetFit models, SST-2) is reachable from this environment.
* **Order:** do this **before** the Ising Born machine. It is cheaper, it tests a
  published claim directly, and the parameter-matched-baseline machinery it needs
  is the same machinery the Born machine comparison needs.
* ✅ **Done, Result 91 — the classical side measured on the paper's own data and
  protocol.** The paper's classical 89.56% reproduces on frozen embeddings (88.96
  ± 0.52%, best of 20 seeds 89.40%), and is the ceiling for classical heads on
  them. **SetFit, the few-shot method the paper's pipeline is named after,
  fine-tunes the same transformer on the same 435 sentences and then uses plain
  logistic regression: 92.56 ± 0.14%, range 92.47–92.80%.** That range contains
  the quantum head's 92.70%. The claimed margin is the frozen-to-fine-tuned
  representation gap, which a classical pipeline closes. Graded **D**, not A′:
  the quantum head is under-specified in the paper and was not reproduced.
  **Open, only if worth the effort:** reproduce the head from IonQ's code, if it
  is released, to settle which embeddings it saw.

**Do not** re-run a variational *classifier* or kernel experiment *as a search for
an advantage* — the IonQ check above is a test of someone else's claim, not that.
Results 68 and 74 closed that shape, and the input barrier above explains why it
was never going to work. The Ising Born machine is not that shape: it generates rather than
reads, and its training can be classical.

### The one direction with real leverage: the block encoding (Result 75)

**This is now the top item, and it displaces everything below it.**

Result 75 closed Result 71's repetition question — repetitions are a VQE
artefact, qubitized phase estimation needs 3 — and in doing so found where the
leverage actually is. The naive LCU block encoding in `exp019` gives 1.1e15 T
gates for a drug-sized molecule, which is **roughly where the published field
stood in 2017**:

| construction | T gates, FeMoco-scale | runtime |
|---|---|---|
| Reiher et al. 2017, Trotter | ~1e14 | years |
| Berry/Gidney 2019, sparse qubitization | ~1e10 | — |
| Lee et al. 2021, tensor hypercontraction | 2.1e10 | ~4 days |
| this repo's naive LCU | 1.1e15 | 804 years |

**~10⁵ of algorithmic progress, in the Hamiltonian's *representation* rather than
its measurement.** That is more than every constant factor in this repository
combined, by two orders of magnitude.

So the useful work is:

1. ~~Double factorisation as a block encoding~~ — **done, Result 76.** It wins,
   and the crossover was *measured*: DF loses up to N = 8 (gain 0.93 on H₈) and
   wins at N = 10 (1.56), with the model putting the crossover at N = 9. H₁₀ was
   added specifically to test that rather than quote it. Extrapolated to 50
   orbitals: **332×**, 1.30e15 → 3.93e12 T gates. The mechanism is that the DF
   1-norm grows as `N^1.93` against the Pauli norm's `N^2.59`, so its penalty is
   transient.

2. ~~Tensor hypercontraction~~ — **attempted, Result 77, and it splits.**

   | | measured | verdict |
   |---|---|---|
   | 1-norm | λ_THC = 0.56 × λ_Pauli at N=8; **6.8× below DF's** | THC delivers |
   | rank | `M ~ N^2.26 ± 0.13`, interval [2.01, 2.52] | **does not reach O(N)** |

   The sizing note above was right that the gap has to come from the per-walk
   cost, and that is exactly the half this construction fails to deliver.

   **The reason is that this experiment *selects* χ from the DF eigenbasis and
   solves for Z, where published THC *optimises* χ and Z jointly by nonlinear
   least squares.** A selection cannot beat its pool. So the open item is now
   narrower and much better defined than "implement THC":

   ~~Fit χ properly.~~ — **done, Result 78, and the answer is yes.**

   | | exponent | ±2σ interval |
   |---|---|---|
   | selected χ (R77) | `N^2.26 ± 0.13` | [2.00, 2.52] |
   | **optimised χ (R78)** | **`N^1.33 ± 0.26`** | **[0.80, 1.86]** |
   | THC's claim | `N^1` | — |

   The intervals do not overlap, so optimisation genuinely helps, and the
   optimised one contains 1. Result 77's diagnosis was right: the selection was
   the limitation, not the THC form. λ improves alongside — 1.37 → 0.86 → 0.43
   times the Pauli norm across H₂/H₄/H₆.

3. ~~The fit's reproducibility~~ — **solved where the optimiser converges
   (Result 79), and it uncovered a convergence problem underneath.**

   A smooth `|Z|` penalty at α = 1e-4 collapses the spread across eight
   independent fits to **exactly 1.0** on H₄ (from 1e5 in residual and 6× in λ),
   at no cost: 8/8 inside chemical accuracy against 7/8, median λ 5.59 → 4.37.
   H₆ replicates the λ collapse and improves λ more (14.82 → 8.20). Too much
   penalty destroys the fit entirely (0/8 at α = 1e-2), so the working range has
   both edges measured.

   **But H₆ converges 1 fit in 8 — with or without the penalty.** Its λ spread
   of 1.0 may mean all eight found the same optimum, or merely that all eight
   stopped at the same iteration cap. This experiment cannot tell those apart,
   so H₆ replicates the number without replicating the evidence.

4. ~~The convergence failure~~ — **done, Result 80, and all three measurements
   landed.**

   * **Why it stops: the iteration cap**, `status 1`, `nit` exactly at the limit.
     Not line search, not tolerances. And not merely tight — 20 000 was *below
     the median requirement* of 46 337, so every H₆ number in Results 78 and 79
     was read off fits stopped a third of the way through.
   * **The λ collapse is real.** It survives at a converging budget: spread 1.00
     against the unpenalised 9.52. Result 79's caveat can be lifted.
   * **The reparameterisation works, and I had reported it as not working.** A
     single-start probe said 2.5%; across eight starts the gauge fixing is worth
     **1.95× in iterations, 8/8 convergence instead of 7/8, 6× in λ spread with
     no penalty, and 4.3× in energy error**. The probe used the unperturbed DF
     start, where the flat direction costs nothing. See Result 80 — it is the
     single-start error this file's own rule 3 exists to prevent.

   **Standing recommendation: gauge-fix always** (no hyperparameter, improves
   convergence, reproducibility and accuracy at once). ~~Add the penalty only when
   λ must be pinned, at 30× in energy error~~ — superseded by Result 90: fit
   unpenalised, then lower λ along the exact minimisers.

   **Now open:** re-measure the rank exponent. Result 78's `N^1.33 ± 0.26` was
   fitted on fits that mostly never converged, with a parameterisation now known
   to cost 2× the iterations and 6× the spread. With gauge + penalty the fits
   converge 8/8 in a median of 17 681 iterations — *below the original cap* — so
   this is now cheap, and it is the number the whole THC claim rests on.

5. **Superseded: what was open before.**

   The exponent above rests on thresholds that move between identical runs. H₄
   at M=8 was inside chemical accuracy on two runs of three and outside on the
   other; H₆ at M=18 converged on one run and not the next; λ moves
   non-monotonically across ranks (8.2, 13.4, 9.1, 30.0, 17.7 on H₆). The
   restart spread says why: 1.6e9 between best and worst of six starts at H₄
   M=12.

   **λ is the quantity the whole cost argument depends on, and this procedure
   does not pin it.** So before any further scaling work, the fit needs to
   return the *same* answer twice. Concretely, in rough order of expected value:

   * penalise the 1-norm in the objective, as published constructions do —
     currently nothing steers the optimiser towards the low-λ optimum among the
     many near-degenerate ones
   * many more restarts, or a smarter initialisation than perturbing the DF
     selection
   * report the threshold as a distribution over runs rather than a single
     number, which is the honest form given the above

   Only after that does a fourth molecule (H₈, ~40 min per rank at six restarts)
   buy anything: a fourth point on a noisy threshold does not tighten a noisy
   exponent.

This is the first item in a long time where the measured evidence says a large
factor is available and nothing in the repository has tried for it.

### The scaling question is answered, negatively (Result 74)

This file said for several sessions that `n`, the parameter count, was the one
lever that could change an exponent, and that an adaptive ansatz reaching
`n ~ N²` would put VQE at `N^7.6` against CCSD(T)'s `N⁷`. **Measured on a
homologous series H₂/H₄/H₆/H₈: it does not.**

| N | UCCSD `n` | ADAPT `n` | ratio |
|---|---|---|---|
| 2 | 3 | 1 | 3.00 |
| 4 | 26 | 10 | 2.60 |
| 6 | 117 | 45 | 2.60 |
| 8 | 360 | 145 | 2.48 |

ADAPT 3.569 ± 0.111 against UCCSD 3.439 ± 0.133 — 0.75 standard errors apart,
indistinguishable, ADAPT's the *higher* of the two. The ratio declines with N.
In shots: `N^9.3` against `N^9.1`, both against a classical `N⁷`.

The three-molecule anecdote that motivated this (1 vs 3, 9 vs 26, 5 vs 92) was a
comparison across molecules rather than along a series — the same failure Result
53 recorded for `Σ|c|`.

**So no open direction in this repository changes a complexity class.** What
would reopen it is an ansatz family with a *provably* sub-quartic parameter
count, tested the same way; nothing here is a candidate. Everything else below
is a constant factor, and the README now says so at the top.

### The question is answered (Result 66); what remains is narrow

The binding quantity is the **median two-qubit gate error**, because the cost of
a gate *is* that number (Result 65, seven devices, ratio 0.64–1.09). Gates scale
as `N^5.12` in spatial orbitals, so the threshold for a molecule is
`1.6e-3 / gates`:

| molecule | 2q gates | needed 2q error | factor from the best device |
|---|---|---|---|
| H₂ | 4 | 4.0e-4 | **3×** |
| H₄ | 1 471 | 1.1e-6 | 1 168× |
| LiH | 9 103 | 1.8e-7 | 7 226× |
| 20-orbital fragment | 5.2M | 3.1e-10 | 4.2M× |
| 50-orbital drug molecule | 572M | 2.8e-12 | **4.5e8×** |

Recent generations delivered 6×. Stacking every gate-reduction result in this
repository gives perhaps 5×.

**So the useful things left to do are not algorithmic**, and pretending otherwise
would waste whoever picks this up:

1. **Watch one number.** The median two-qubit gate error of the best available
   device, against the thresholds above. `experiments/exp015_device_generations.py`
   re-measures the relationship on any new calibration snapshot in ~4 minutes per
   device; the threshold table is `/tmp`-free and recomputable from
   `qres/ansatz.py` gate counts.

2. **Fault tolerance — sized, and it reverses the picture (Result 69).** This
   item used to read "a different project". It is a different project to build
   but not to size, and sizing it with the same measured inputs gives **1 452
   physical qubits for H₄ and ~10⁵ for a drug-sized molecule**, because code
   distance grows only logarithmically in the required fidelity. Eight orders of
   magnitude in target error cost 3.6× the distance.

   So the bare-metal conclusion above is correct and *narrower than it reads*:
   NISQ chemistry is hopeless; error-corrected chemistry is a 10⁵-qubit
   engineering problem.

   **Distillation is now included (Result 70)** and the earlier caveat was too
   pessimistic: it costs 3× on H₄ and only **1.25×** on a drug-sized molecule,
   because the factory is a fixed 12-logical-qubit footprint while the data
   register grows. Final figure **~1.3 × 10⁵ physical qubits**.

   **Time is now measured too, and it retracted the claim that used to sit here
   (Result 71).** This item said "months of wall-clock, sequentially" and called
   modelling the trade "the single most useful thing left". The arithmetic had
   never been done and was wrong by two orders of magnitude: **one factory is
   9.9 h, ten factories reach a floor of 59 min**, and past saturation
   parallelism buys nothing because T gates sit on the algorithm's critical
   path. The trade is bounded and cheap — 2.8× the qubits for 10× the time.

   **The open quantity is now repetitions, and it dominates everything.** All of
   the above is *one circuit execution*: 10³ executions is 41 days, 10⁶ is 113
   years. A variational loop needs one per energy evaluation, and this project
   measured H₄ consuming ~10⁸ shots *without* reaching chemical accuracy. So the
   next useful thing is to bound the execution count for an error-corrected
   method that is not VQE — qubitization or phase estimation, whose whole point
   is replacing a sampling loop with coherent evolution. That is the last
   dimension of the error-corrected estimate still open, and unlike the previous
   two it could plausibly move the answer by many orders of magnitude in either
   direction.

3. **The low-locality folding encoding**, if someone wants to close the one
   remaining loose end. Result 63 measured the *naive* turn encoding (dense, 2ⁿ
   terms, 1 538 gates at six residues). Published constructions bound the
   interaction weight with ancillas. It would fix the encoding column — but
   Result 64 measured folding's classical baseline failing too (3 of 4 literature
   optima in under a minute), so it no longer produces a candidate.

**What would reopen this.** Four classes have now been checked against both
criteria — a required relative accuracy above ~1%, and a classical baseline that
genuinely fails at a size needing under a hundred two-qubit gates. **None has
both**, and each fails differently: chemistry on cost, MaxCut and folding on a
strong competitor, QML on what the question is.

Quantum ML was the last live candidate and Result 68 closed it. The kernel is the
only thing in this project that *fits* the hardware — six gates per entry,
accuracy under device noise identical to exact simulation — and it is at chance
(0.51–0.55) on periodic data, on data built from precisely the pairwise-difference
form its entangling layer encodes, and on parity, which is the one dataset where
the classical side is weak (0.685). It wins nowhere except on labels its own
circuit produced.

So there is no known problem class left in this repository where a positive
result is available at current hardware. Finding one means looking somewhere
none of chemistry, MaxCut, folding or kernel methods reaches — and the two
criteria above are the filter to apply before building anything, which is the
main transferable thing here.

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
