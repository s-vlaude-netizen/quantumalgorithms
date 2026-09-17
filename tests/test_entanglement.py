"""Bond-dimension measurement (Result 83).

The whole result is one function, and it has three ways to be quietly wrong: a
mean over cuts instead of a maximum (understates), a discarded-weight convention
off by one (understates), and a normalisation error (either way). Each is pinned
against a state whose answer is known analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp027_entanglement import TRUNCATIONS, bond_dimensions


def test_a_product_state_needs_bond_dimension_one():
    """|0000> factorises across every cut."""
    state = np.zeros(16)
    state[0] = 1.0
    measured = bond_dimensions(state)
    assert measured["exact_schmidt_rank"] == 1
    assert measured["max_entropy_bits"] == pytest.approx(0.0, abs=1e-12)
    for target in TRUNCATIONS:
        assert measured["bond_dimension"][str(target)] == 1


def test_a_ghz_state_needs_exactly_two():
    """Maximal *entropy* at one bit, but Schmidt rank two at every cut.

    This separates the two quantities: GHZ is often called maximally entangled,
    and its bond dimension is nonetheless 2. A measure that reported entropy as
    the cost would overstate it badly.
    """
    state = np.zeros(16)
    state[0] = state[-1] = 1 / np.sqrt(2)
    measured = bond_dimensions(state)
    assert measured["exact_schmidt_rank"] == 2
    assert measured["max_entropy_bits"] == pytest.approx(1.0, abs=1e-12)
    assert measured["bond_dimension"][str(min(TRUNCATIONS))] == 2


def test_a_random_state_needs_the_maximal_bond_dimension():
    """The other end: a Haar-random state is not compressible at all."""
    rng = np.random.default_rng(0)
    state = rng.normal(size=64) + 1j * rng.normal(size=64)
    measured = bond_dimensions(state)
    assert measured["qubits"] == 6
    assert measured["exact_schmidt_rank"] == measured["maximal_rank"] == 8


def test_the_maximum_over_cuts_is_taken_not_the_mean():
    """One bad cut forces the representation; averaging would flatter the result.

    Built so cuts differ: qubits 0-1 are entangled, qubits 2-3 are not, so the
    middle cut is cheap and the first cut is not.
    """
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    product = np.array([1.0, 0.0, 0.0, 0.0])
    state = np.kron(bell, product)

    measured = bond_dimensions(state)
    # the cut inside the Bell pair costs 2; a mean over the three cuts would be
    # below 2 and this assertion is what catches that
    assert measured["exact_schmidt_rank"] == 2


def test_looser_truncation_never_costs_more():
    """Monotonicity in the discarded-weight target, which the convention must obey."""
    rng = np.random.default_rng(3)
    state = rng.normal(size=256)
    measured = bond_dimensions(state)
    ordered = sorted(TRUNCATIONS, reverse=True)       # loosest first
    values = [measured["bond_dimension"][str(t)] for t in ordered]
    assert all(a <= b for a, b in zip(values, values[1:])), values


def test_an_unnormalised_state_gives_the_same_answer():
    """Scaling the input must not move the discarded weight."""
    rng = np.random.default_rng(5)
    state = rng.normal(size=64)
    assert (
        bond_dimensions(state)["bond_dimension"]
        == bond_dimensions(state * 137.0)["bond_dimension"]
    )


def test_bond_dimension_is_enough_to_reconstruct_within_the_target():
    """The reported chi must actually achieve the discarded weight it claims.

    Truncating to chi singular values has to leave less than the target weight
    behind -- an off-by-one in the cumulative sum would break this while leaving
    every other test passing.
    """
    rng = np.random.default_rng(9)
    state = rng.normal(size=256)
    state = state / np.linalg.norm(state)
    measured = bond_dimensions(state)

    for target in TRUNCATIONS:
        chi = measured["bond_dimension"][str(target)]
        worst = 0.0
        for cut in range(1, measured["qubits"]):
            singular = np.linalg.svd(state.reshape(2**cut, -1), compute_uv=False)
            weights = singular**2 / np.sum(singular**2)
            worst = max(worst, float(np.sum(weights[chi:])))
        assert worst < target, (
            f"chi={chi} leaves {worst:.2e} behind, target was {target:.0e}"
        )
