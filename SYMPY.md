# SymPy in this repository

SymPy is where the repository's **proofs** come from (Results 86 and 88): an
identity in a polynomial ring, or a rank certified in exact arithmetic, is not a
coincidence of one random draw. This file is the working knowledge for doing
more of that, so a session can pick it up cold.

## Which SymPy

`requirements.txt` installs **the independent fork**
[`s-vlaude-netizen/sympy`](https://github.com/s-vlaude-netizen/sympy), pinned to
a commit. It is upstream SymPy plus bug fixes upstream declined under its
AI-generated-code policy (see the fork's `FORK.md` and `CHANGELOG-FORK.md`), and
imports as `sympy`.

* **Pinned by commit, not `@master`**, so a proof re-runs on the code it was
  checked with. The fork releases by `master` rather than by tag, and a commit
  hash is as reproducible as a tag would be.
* **To move the pin** after a fork fix: put the new `master` hash in
  `requirements.txt`, reinstall, re-run the symbolic tests
  (`tests/test_gauge_structure.py`, `tests/test_thc_identifiability.py`).
* **Install order matters.** `qiskit` depends on `sympy`; installed on its own
  afterwards it can pull PyPI's SymPy over the fork. `pip install -r
  requirements.txt` resolves both together and keeps the fork. Check with
  `python -c "import sympy; print(sympy.__version__)"` — the fork reports
  `...+fork`.
* **Fixing SymPy itself**: bugs found here belong in the fork, not upstream. Its
  `CLAUDE.md` has the standards (a regression test verified to fail first, the
  subpackage's full suite, a changelog entry); then move the pin here.

## Techniques that worked

* **Free symbols, then `expand`, compared to zero** (Result 86): proves an
  identity for all values, not for a sample.
* **Let SymPy differentiate the model** and check any hand-derived or
  closed-form Jacobian against it entry by entry (Result 88's tests). The fast
  closed form then carries the sweep; the proof does not rest on the derivation.
* **Rank at an integer point, modulo a large prime** (Result 88): a non-zero
  minor mod p is non-zero over the integers, so it is a *certified lower bound*
  on the generic rank. About 7× faster than rationals, whose entries grow during
  elimination. Pair it with an argued upper bound (a symmetry, a dimension
  count); where the two meet, the count is exact.
* **Exact directional derivatives over a rational kernel**
  (`DomainMatrix.nullspace()` over `QQ`): decides "is this quantity constant
  along every flat direction" with no tolerance to choose.

## Known SymPy limitations hit here, with workarounds

* `lambdify(args, Matrix(...), "math")` fails at call time with
  `NameError: ImmutableDenseMatrix` — the pure-Python printer emits a SymPy
  class the `math` namespace does not have. Workaround: evaluate with
  `modules="sympy"` for exact integers, or `"numpy"` for floats, or write the
  closed form directly.
* `DomainMatrix.rank()` over `QQ` is slow without `gmpy2` (pure-Python
  integers): 22 s against 6.4 s with it on a 441 × 216 matrix. `gmpy2` is in
  `requirements.txt`.
* `sympy.physics.secondquant`: a commutator of one-body operators with
  *general* indices, `Commutator(Fd(p)*F(q), Fd(r)*F(s)).doit(wicks=True)`,
  raises `SubstitutionOfAmbigousOperatorFailed` — general indices cannot be
  normal-ordered against the Fermi vacuum. Use indices declared
  `above_fermi`/`below_fermi`, or work with explicit matrices.
* Matrix expressions: `Trace(X*Y) - Trace(Y*X)` does not simplify to zero (no
  cyclic canonicalisation), and `ask(Q.real_elements(X), Q.orthogonal(X))` gives
  `None`. For invariance proofs under orbital rotations, expand to components or
  parametrise the rotation explicitly (a rational Cayley form keeps everything
  polynomial after clearing denominators).
