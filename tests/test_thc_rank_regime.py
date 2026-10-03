"""Small-molecule THC thresholds: chemistry or algebra (Result 94).

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
        def free(m, orbitals=orbitals):
            return orbitals * m + m * (m - 1) // 2
        assert free(rank) >= independent_entries(orbitals) > free(rank - 1)


def test_algebraic_rank_grows_quadratically():
    """M^2/2 ~ P^2/2 with P ~ N^2/2, so M ~ N^2/2: the ratio to N^2 tends to 1/2."""
    ratios = [algebraic_rank(n) / n**2 for n in (20, 40, 80, 160)]
    assert ratios == sorted(ratios)
    assert abs(ratios[-1] - 0.5) < 0.01


# ------------------------------------------------- the spectral lower bound

def random_thc(orbitals, rank, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    chi = rng.normal(size=(orbitals, rank))
    coupling = rng.normal(size=(rank, rank))
    return chi, 0.5 * (coupling + coupling.T)


def symmetric_tensor(orbitals, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    t = rng.normal(size=(orbitals,) * 4)
    for axes in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        t = 0.5 * (t + t.transpose(axes))
    return t


def test_a_rank_m_thc_tensor_has_at_most_m_nonzero_eigenvalues():
    from experiments.exp022_optimised_thc import thc_tensor
    from experiments.exp038_thc_rank_regime import eckart_young_bound, pair_spectrum

    chi, coupling = random_thc(5, 6, seed=0)
    spectrum = pair_spectrum(thc_tensor(chi, coupling))
    assert eckart_young_bound(spectrum, 6) < 1e-20
    assert spectrum[5] > 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_no_thc_model_beats_the_eckart_young_bound(seed):
    """The bound is a theorem about every rank-M model, so test it on random ones."""
    import numpy as np

    from experiments.exp022_optimised_thc import thc_tensor
    from experiments.exp038_thc_rank_regime import eckart_young_bound, pair_spectrum

    target = symmetric_tensor(4, seed)
    chi, coupling = random_thc(4, 5, seed + 100)
    residual = 0.5 * float(np.sum((thc_tensor(chi, coupling) - target) ** 2))
    assert residual >= eckart_young_bound(pair_spectrum(target), 5) - 1e-12


def test_the_bound_is_attained_by_the_truncated_eigendecomposition():
    """Same metric as the THC residual: the bound is tight, not just valid."""
    import numpy as np

    from experiments.exp038_thc_rank_regime import eckart_young_bound, pair_spectrum

    orbitals, rank = 4, 6
    target = symmetric_tensor(orbitals, 7)
    index = [(p, q) for p in range(orbitals) for q in range(p, orbitals)]
    weight = np.sqrt([1.0 if p == q else 2.0 for p, q in index])
    matrix = np.array([[target[p, q, r, s] for r, s in index] for p, q in index])
    values, vectors = np.linalg.eigh(weight[:, None] * matrix * weight[None, :])
    keep = np.argsort(np.abs(values))[::-1][:rank]
    truncated = (vectors[:, keep] * values[keep]) @ vectors[:, keep].T
    pair = truncated / np.outer(weight, weight)
    full = np.zeros_like(target)
    for i, (p, q) in enumerate(index):
        for j, (r, s) in enumerate(index):
            for a, b in {(p, q), (q, p)}:
                for c, d in {(r, s), (s, r)}:
                    full[a, b, c, d] = pair[i, j]
    residual = 0.5 * float(np.sum((full - target) ** 2))
    assert residual == pytest.approx(eckart_young_bound(pair_spectrum(target), rank), rel=1e-10)


def test_the_spectrum_is_invariant_under_orbital_rotations():
    import numpy as np

    from experiments.exp038_thc_rank_regime import pair_spectrum

    target = symmetric_tensor(4, 3)
    rotation, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(4, 4)))
    rotated = np.einsum("ap,bq,cr,ds,pqrs->abcd", rotation, rotation, rotation, rotation, target)
    np.testing.assert_allclose(pair_spectrum(rotated), pair_spectrum(target), atol=1e-10)


@pytest.mark.parametrize("name", ["H2", "H4"])
def test_fci_energy_equals_the_series_qiskit_route(name):
    """The energy criterion changed implementation for H10; it must not change value."""
    import numpy as np

    from experiments.exp021_tensor_hypercontraction import energy_of
    from experiments.exp038_thc_rank_regime import fci_energy, integrals
    from qres.factorization import molecular_integrals
    from qres.problems.chemistry import build_molecule

    problem = build_molecule(name)
    one_body, two_body, _ = molecular_integrals(problem)
    ours = integrals(name)
    np.testing.assert_allclose(ours[1], two_body, atol=1e-12)
    noise = np.random.default_rng(0).normal(scale=1e-4, size=two_body.shape)
    for axes in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        noise = 0.5 * (noise + noise.transpose(axes))
    for tensor in (two_body, two_body + noise):
        assert fci_energy(one_body, tensor) == pytest.approx(
            energy_of(problem, one_body, tensor), abs=1e-10)


@pytest.mark.parametrize("name, orbitals", [("H4", 4), ("H6", 6)])
def test_the_significant_eigenvalues_are_the_local_pair_densities(name, orbitals):
    """2N - 1 eigenvalues above 0.01, and their eigenvectors live on the 2N - 1
    on-site and nearest-neighbour pair densities of atom-localised orbitals."""
    from experiments.exp038_thc_rank_regime import (
        SIGNIFICANT,
        local_pair_weight,
        spectrum_of,
    )

    spectrum = spectrum_of(name, None)
    assert int((spectrum > SIGNIFICANT).sum()) == 2 * orbitals - 1
    assert spectrum[2 * orbitals - 2] / spectrum[2 * orbitals - 1] > 20
    weight = local_pair_weight(name)
    assert len(weight) == 2 * orbitals - 1
    assert weight.min() > 0.9
