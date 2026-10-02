"""Lowering THC's lambda without moving the tensor (Result 90).

Everything is tested on random models whose regime is known from Result 89's
counting, so the claims do not depend on a molecule or a fit: nothing moves
strictly inside the boundary, a finite and complete set of alternatives exists
on it, and past it lambda falls while the tensor stays put.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp035_thc_fibre import (
    TENSOR_TOLERANCE,
    alternatives,
    columns,
    one_norm_exact,
    rank_one_points,
    regime,
    slide,
)


def random_model(orbitals, rank, seed):
    rng = np.random.default_rng(seed)
    chi = rng.normal(size=(orbitals, rank))
    chi /= np.linalg.norm(chi, axis=0)
    coupling = rng.normal(size=(rank, rank))
    return chi, 0.5 * (coupling + coupling.T)


def tensor(chi, coupling):
    block = columns(chi)
    return block @ coupling @ block.T


def test_regimes_follow_the_boundary():
    assert [regime(4, m) for m in (6, 7, 8, 9)] == ["inside", "boundary", "slides", "slides"]
    assert [regime(6, m) for m in (15, 16, 17)] == ["inside", "boundary", "slides"]


def test_strictly_inside_only_the_columns_are_rank_one():
    """More quadrics than dimensions: nothing besides the columns themselves."""
    chi, _ = random_model(4, 6, seed=1)
    assert len(rank_one_points(chi, starts=600)) == 6


def test_strictly_inside_the_slide_moves_nothing():
    chi, coupling = random_model(4, 6, seed=2)
    best, _ = slide(chi, coupling, starts=2)
    assert best[0] == pytest.approx(one_norm_exact(chi, coupling), rel=1e-9)


def test_on_the_boundary_bezout_forces_one_extra_real_point():
    """3 quadrics in P^3 meet in 8 points; 7 are real columns, so the 8th is real."""
    chi, _ = random_model(4, 7, seed=3)
    assert len(rank_one_points(chi, starts=1500)) == 8


def test_on_the_boundary_every_alternative_is_the_same_tensor():
    chi, coupling = random_model(4, 7, seed=4)
    best, count, options = alternatives(chi, coupling)
    assert count == 8
    exact = [o for o in options if o["tensor_change"] < TENSOR_TOLERANCE]
    assert len(exact) >= 2, "the extra point should give at least one alternative"
    original = one_norm_exact(chi, coupling)
    assert any(o["one_norm"] == pytest.approx(original, rel=1e-8) for o in exact)
    assert best[0] <= original + 1e-9


def test_past_the_boundary_lambda_falls_and_the_tensor_does_not_move():
    chi, coupling = random_model(4, 8, seed=5)
    before = one_norm_exact(chi, coupling)
    best, attempts = slide(chi, coupling, starts=2)
    assert best[0] < 0.99 * before
    assert np.max(np.abs(tensor(best[1], best[2]) - tensor(chi, coupling))) < TENSOR_TOLERANCE
    assert np.allclose(np.linalg.norm(best[1], axis=0), 1.0)


def test_slide_refuses_saturated_ranks():
    chi, coupling = random_model(3, 6, seed=6)
    with pytest.raises(ValueError):
        slide(chi, coupling)
