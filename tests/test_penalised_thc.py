"""The 1-norm penalty (Result 79), and the gradient it adds.

Result 78's finding was that the THC fit does not reproduce: lambda -- the
quantity a block encoding's cost is *proportional to* -- varied by 6x across
independent fits of the identical configuration, and the residual by 1e5. The
diagnosis was that nothing in the objective preferred the low-lambda optimum
among many near-degenerate ones.

This pins the fix and its limits. The penalty has a working range: too weak and
it does nothing, too strong and it destroys the fit entirely (0/8 inside
chemical accuracy at alpha=1e-2). Both ends are tested, because a penalty
reported only where it helps is not a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp023_penalised_thc import DELTA, _objective, one_norm, spread

RESULTS = Path("results/exp023_penalised_thc_H4_M12.json")


def _pack(chi, coupling):
    return np.concatenate([chi.ravel(), coupling.ravel()])


@pytest.mark.parametrize("alpha", [0.0, 1e-4, 1e-2])
def test_penalised_gradient_matches_finite_differences(alpha):
    """Including at alpha=0, so the penalty cannot silently break the base term.

    Result 78's gradient was wrong on both blocks and only a finite-difference
    check caught it. The penalty adds two more terms; they get the same check.
    """
    rng = np.random.default_rng(5)
    orbitals, rank = 4, 5

    target = rng.normal(size=(orbitals,) * 4)
    target = 0.5 * (target + target.transpose(2, 3, 0, 1))
    target = 0.5 * (target + target.transpose(1, 0, 2, 3))
    target = 0.5 * (target + target.transpose(0, 1, 3, 2))

    chi = rng.normal(size=(orbitals, rank)) * 0.5
    coupling = rng.normal(size=(rank, rank))
    packed = _pack(chi, coupling)

    _, gradient = _objective(packed, orbitals, rank, target, alpha)

    epsilon = 1e-7
    worst = 0.0
    for index in range(len(packed)):
        step = np.zeros_like(packed)
        step[index] = epsilon
        high, _ = _objective(packed + step, orbitals, rank, target, alpha)
        low, _ = _objective(packed - step, orbitals, rank, target, alpha)
        numeric = (high - low) / (2 * epsilon)
        worst = max(worst, abs(numeric - gradient[index]) / max(abs(numeric), 1e-9))

    assert worst < 1e-4, f"alpha={alpha}: gradient off by {worst:.2e}"


def test_the_smooth_absolute_value_tracks_the_real_one():
    """The penalty must be |Z| where it matters, not merely something convex."""
    values = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0])
    smoothed = np.sqrt(values**2 + DELTA**2) - DELTA

    # The right property is an ABSOLUTE error bounded by delta, not a relative
    # one: the smoothing subtracts delta outright, so at |z| = 1e-3 that is a
    # 0.1% shift. A first version asserted rtol=1e-6 and failed -- the test was
    # wrong, not the function, and asserting the achievable bound is the fix
    # rather than loosening the number until it passes.
    assert np.all(smoothed <= values)                       # never overstates |z|
    assert np.all(values - smoothed <= DELTA + 1e-15)       # and never by more than delta
    np.testing.assert_allclose(smoothed[2:], values[2:], rtol=1e-5)  # exact where it matters

    # and differentiable at zero, which |z| is not
    _, gradient = _objective(
        np.zeros(2 * 2 + 2 * 2), 2, 2, np.zeros((2, 2, 2, 2)), 1e-2
    )
    assert np.all(np.isfinite(gradient))


def test_one_norm_uses_the_column_scales():
    """chi is free here, so a coupling entry means nothing without |chi|^2."""
    chi = np.array([[2.0, 0.0], [0.0, 3.0]])
    coupling = np.array([[1.0, 0.0], [0.0, 1.0]])
    # |chi_0|^2 = 4, |chi_1|^2 = 9 -> 0.5 * (1*16 + 1*81) = 48.5
    assert one_norm(chi, coupling) == pytest.approx(48.5)


def test_spread_is_a_ratio_and_ignores_degenerate_input():
    assert spread([1.0, 2.0, 4.0]) == pytest.approx(4.0)
    assert spread([3.0]) != spread([3.0])  # nan for a single value
    assert spread([2.0, 2.0]) == pytest.approx(1.0)


def test_the_penalty_tightens_the_spread_without_losing_accuracy():
    """Result 79's headline, read off the recorded run."""
    if not RESULTS.exists():
        pytest.skip("run experiments.exp023_penalised_thc first")
    data = json.loads(RESULTS.read_text())
    rows = {r["penalty"]: r for r in data["rows"]}

    baseline = rows[0.0]
    tuned = rows[1e-4]

    assert tuned["one_norm_spread"] < baseline["one_norm_spread"] / 2, (
        "the penalty no longer tightens the 1-norm spread"
    )
    assert tuned["inside_chemical_accuracy"] >= baseline["inside_chemical_accuracy"], (
        "the penalty now costs accuracy it did not cost before"
    )


def test_too_much_penalty_destroys_the_fit():
    """The other end of the range, which is why alpha cannot just be raised.

    At alpha=1e-2 nothing converges and nothing lands inside chemical accuracy.
    A penalty reported only where it helps would be selective quotation.
    """
    if not RESULTS.exists():
        pytest.skip("run experiments.exp023_penalised_thc first")
    data = json.loads(RESULTS.read_text())
    rows = {r["penalty"]: r for r in data["rows"]}

    strong = rows[1e-2]
    assert strong["inside_chemical_accuracy"] == 0, (
        "the strongest penalty no longer fails; the working range has moved and "
        "the recommended alpha should be re-derived rather than this test relaxed"
    )
