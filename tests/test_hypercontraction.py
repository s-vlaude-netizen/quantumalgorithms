"""THC's two halves (Result 77): the 1-norm delivers, the rank does not.

The experiment behind this measures a *selection*-based THC construction -- pick
chi vectors from the double-factorisation eigenbasis, solve for the coupling --
and finds it splits:

* the 1-norm is genuinely good: 0.56x the Pauli norm at N=8, ~1/7 of DF's
* the rank is not: M ~ N^2.26, where THC's whole claim is O(N)

Both halves are pinned, because reporting only the first would be the kind of
selective quotation this repository keeps catching itself at. The numerical
guards are pinned too: the fit is severely ill-conditioned, and two separate
criteria were needed before any lambda here was trustworthy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

RESULTS = Path("results/exp021_tensor_hypercontraction.json")


def load():
    if not RESULTS.exists():
        pytest.skip("run experiments.exp021_tensor_hypercontraction first")
    return json.loads(RESULTS.read_text())


def test_the_rank_does_not_achieve_the_linear_scaling_thc_claims():
    """The negative half, and the reason this construction is not enough.

    THC's argument is M ~ O(N). Selecting chi from the DF eigenbasis gives
    M ~ N^2.26, and the +-2 standard error interval excludes 1. A selection can
    only ever be as good as its pool; real THC optimises chi nonlinearly, which
    is exactly the step skipped here.
    """
    data = load()
    exponent = data["rank_exponent"]
    stderr = data["rank_exponent_stderr"]

    assert exponent is not None
    assert exponent - 2 * stderr > 1.0, (
        f"M ~ N^{exponent:.2f} +- {stderr:.2f} now includes linear scaling; "
        "the negative conclusion needs re-deriving"
    )


def test_the_one_norm_beats_the_pauli_representation_at_larger_sizes():
    """The positive half: lambda is what THC actually delivers here."""
    data = load()
    usable = [r for r in data["rows"] if r.get("smallest_sufficient")]
    assert len(usable) >= 3

    largest = max(usable, key=lambda r: r["orbitals"])
    ratio = largest["smallest_sufficient"]["one_norm"] / largest["pauli_one_norm"]
    assert ratio < 1.0, f"THC 1-norm no longer beats Pauli at N={largest['orbitals']}"

    # and the advantage should grow, not merely exist
    smallest = min(usable, key=lambda r: r["orbitals"])
    small_ratio = smallest["smallest_sufficient"]["one_norm"] / smallest["pauli_one_norm"]
    assert ratio < small_ratio, "the 1-norm advantage should improve with size"


def test_every_reported_rank_is_both_stable_and_converged():
    """Two guards, both of which were needed and neither of which is redundant.

    A plain pseudo-inverse gave couplings around 1e26 that reproduced the tensor
    by cancellation (lambda = 116109 for H4). Ridge regularisation fixed that but
    still passed H8 at M=64 with lambda = 1548 where M=80 gave 50 -- stable under
    the regulariser, yet nowhere near converged. Both criteria are load-bearing.
    """
    data = load()
    for row in data["rows"]:
        best = row.get("smallest_sufficient")
        if best is None:
            continue
        assert best["stable"], f"{row['molecule']}: reported an unstable fit"
        assert best.get("converged"), f"{row['molecule']}: reported an unconverged fit"
        assert best["energy_error"] < data["chemical_accuracy"]


def test_the_energy_error_falls_monotonically_with_rank_among_valid_fits():
    """More information must not make a fit worse.

    The violation of this was what exposed the conditioning bug: energy error ran
    3e-3 -> 5e10 -> 1.6e-12 across consecutive ranks. Checked only over fits that
    pass the stability guard, since unstable ones are exactly the ones that break
    it and are already excluded from every conclusion.
    """
    data = load()
    for row in data["rows"]:
        stable = [s for s in row["sweep"] if s["stable"]]
        if len(stable) < 3:
            continue
        errors = [s["energy_error"] for s in stable]
        # allow small non-monotonicity from the ridge floor, but not orders of magnitude
        for earlier, later in zip(errors, errors[1:]):
            assert later <= max(earlier * 10, 1e-7), (
                f"{row['molecule']}: error rose from {earlier:.2e} to {later:.2e} "
                "with more information"
            )


def test_the_reconstruction_carries_chemists_eightfold_symmetry():
    """Without it qiskit-nature rejects the tensor and the sweep truncates.

    A first version lost the symmetry at the 1e-16 level and crashed at M/N >= 3
    on every molecule but H2 -- which silently cut the sweep off *before* the
    ranks where THC starts working, and would have produced a far more negative
    conclusion than the data supports.
    """
    import sys

    sys.argv = ["test"]
    from qres.factorization import molecular_integrals
    from qres.problems.chemistry import build_molecule

    from experiments.exp021_tensor_hypercontraction import thc_fit

    problem = build_molecule("H4")
    one_body, two_body, _ = molecular_integrals(problem)
    fit = thc_fit(one_body, two_body, 16)
    tensor = fit["two_body"]

    np.testing.assert_allclose(tensor, tensor.transpose(1, 0, 2, 3), atol=1e-12)
    np.testing.assert_allclose(tensor, tensor.transpose(0, 1, 3, 2), atol=1e-12)
    np.testing.assert_allclose(tensor, tensor.transpose(2, 3, 0, 1), atol=1e-12)
