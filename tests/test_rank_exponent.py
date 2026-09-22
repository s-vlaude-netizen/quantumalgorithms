"""The re-measured rank exponent (Result 87, pending its run).

The load-bearing thing here is not the fit, it is the rank grid. Sweeping only
multiples of N quantises the threshold to a multiple of N, so an exponent of
exactly 1.0 can be an artefact of the grid rather than a measurement. That is
pinned below, because it is invisible in the output and would look like a clean
result.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp029_rank_exponent import (
    PENALTY,
    _fit_once,
    fit_exponent,
    rank_grid,
)


def test_the_coarse_grid_cannot_resolve_a_threshold_between_multiples():
    """The methodological trap, stated as a test.

    For N = 6 the coarse grid jumps 6 -> 12, so a true threshold of 8 or 9 is
    reported as 12. Every molecule then lands at the same multiple and the
    fitted exponent is 1.0 by construction.
    """
    coarse = rank_grid(6, fine=False)
    assert 6 in coarse and 12 in coarse
    assert not any(6 < rank < 12 for rank in coarse), (
        "the coarse grid is supposed to be unable to see between multiples; "
        "if it can, this test and the fine grid are both pointless"
    )

    fine = rank_grid(6, fine=True)
    assert [7, 8, 9, 10, 11] == [r for r in fine if 6 < r < 12]


def test_the_fine_grid_steps_by_one_and_spans_N_to_3N():
    for orbitals in (2, 4, 6, 8):
        fine = rank_grid(orbitals, fine=True)
        assert fine[0] == max(2, orbitals)
        assert fine[-1] == 3 * orbitals
        assert all(b - a == 1 for a, b in zip(fine, fine[1:]))


def test_the_coarse_grid_is_the_one_result_78_used():
    """Comparability: the coarse arm must reproduce Result 78's rank set."""
    assert rank_grid(4, fine=False) == [4, 8, 12, 16, 24, 32]


def test_the_coarse_grid_deduplicates():
    """At N = 2 several multiples collide; a repeated rank would be refitted."""
    coarse = rank_grid(2, fine=False)
    assert len(coarse) == len(set(coarse))


def test_fit_exponent_recovers_a_known_slope():
    orbitals = np.array([2.0, 4.0, 6.0, 8.0])
    slope, stderr = fit_exponent(orbitals, 3.0 * orbitals**1.5)
    assert slope == pytest.approx(1.5, abs=1e-9)
    assert stderr < 1e-9


def test_fit_exponent_refuses_two_points():
    """Two points give a slope with no residual and so no honest error bar."""
    slope, stderr = fit_exponent([2.0, 4.0], [4.0, 8.0])
    assert np.isnan(slope) and np.isnan(stderr)


def test_a_perfectly_proportional_threshold_gives_exponent_one():
    """Documents what the grid artefact would look like, so it is recognisable.

    If every threshold is exactly 2N, the fit returns 1.00 with zero error --
    which is indistinguishable from a real linear law by looking at the number
    alone. That is precisely why the fine grid exists.
    """
    orbitals = np.array([2.0, 4.0, 6.0, 8.0])
    slope, stderr = fit_exponent(orbitals, 2.0 * orbitals)
    assert slope == pytest.approx(1.0, abs=1e-9)
    assert stderr == pytest.approx(0.0, abs=1e-9)


def test_restart_selection_uses_the_residual_not_the_penalised_objective():
    """Otherwise 'best of several starts' would prefer a worse fit with small λ.

    The objective L-BFGS minimises includes the penalty, so `result.fun` is not
    the residual. The two must differ on a real fit, or the distinction is not
    being made.
    """
    from qres.factorization import molecular_integrals
    from qres.problems.chemistry import build_molecule

    problem = build_molecule("H2")
    one_body, two_body, _ = molecular_integrals(problem)
    rng = np.random.default_rng(0)
    start = rng.normal(size=(one_body.shape[0], 4))

    fit = _fit_once(one_body, two_body, 4, start, PENALTY, iterations=500)
    assert fit["residual"] <= fit["penalised_objective"] + 1e-12
    assert fit["penalised_objective"] > fit["residual"], (
        "the penalty contributes nothing; restart selection would be identical "
        "either way and the distinction in _fit_once is dead code"
    )


def test_the_fit_returns_unit_norm_columns():
    """Gauge-fixed by construction; the reported λ assumes it."""
    from qres.factorization import molecular_integrals
    from qres.problems.chemistry import build_molecule

    problem = build_molecule("H2")
    one_body, two_body, _ = molecular_integrals(problem)
    rng = np.random.default_rng(1)
    fit = _fit_once(one_body, two_body, 4,
                    rng.normal(size=(one_body.shape[0], 4)), PENALTY, iterations=200)
    np.testing.assert_allclose(np.linalg.norm(fit["chi"], axis=0), 1.0, atol=1e-10)
