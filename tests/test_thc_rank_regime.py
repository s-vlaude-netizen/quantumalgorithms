"""Small-molecule THC thresholds: chemistry or algebra (Result 93).

The algebraic count is the reference every threshold is compared with, so it is
pinned here, together with the independent-entry count it rests on.
"""

from __future__ import annotations

import pytest

from experiments.exp038_thc_rank_regime import algebraic_rank, independent_entries


@pytest.mark.parametrize("orbitals, entries", [(2, 6), (4, 55), (6, 231), (8, 666)])
def test_independent_entries_of_an_eightfold_symmetric_tensor(orbitals, entries):
    assert independent_entries(orbitals) == entries


def brute_force_independent_entries(orbitals):
    """Count orbits of (p, q, r, s) under the 8-fold permutational symmetry."""
    seen = set()
    for p in range(orbitals):
        for q in range(orbitals):
            for r in range(orbitals):
                for s in range(orbitals):
                    orbit = {(p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                             (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)}
                    seen.add(min(orbit))
    return len(seen)


@pytest.mark.parametrize("orbitals", [2, 3, 4, 5])
def test_the_count_is_the_symmetry_orbit_count(orbitals):
    assert independent_entries(orbitals) == brute_force_independent_entries(orbitals)


@pytest.mark.parametrize("orbitals, rank", [(2, 3), (4, 8), (6, 17), (8, 30)])
def test_algebraic_rank(orbitals, rank):
    assert algebraic_rank(orbitals) == rank


def test_algebraic_rank_is_the_first_rank_that_reaches_the_count():
    for orbitals in range(2, 12):
        rank = algebraic_rank(orbitals)
        free = lambda m: orbitals * m + m * (m - 1) // 2  # noqa: E731
        assert free(rank) >= independent_entries(orbitals) > free(rank - 1)


def test_algebraic_rank_grows_quadratically():
    """M^2/2 ~ P^2/2 with P ~ N^2/2, so M ~ N^2/2: the ratio to N^2 tends to 1/2."""
    ratios = [algebraic_rank(n) / n**2 for n in (20, 40, 80, 160)]
    assert ratios == sorted(ratios)
    assert abs(ratios[-1] - 0.5) < 0.01
