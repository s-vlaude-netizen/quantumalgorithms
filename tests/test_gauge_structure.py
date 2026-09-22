"""The THC gauge group (Result 86).

This is the first result in the repository whose core claim is a *proof* rather
than a measurement, so the tests are the proof's load-bearing parts: that the
symbolic identity really is symbolic (free symbols, not lucky numbers), that the
generators really are the tangent of the finite transformation, and that the one
place the invariance breaks is the place claimed.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from experiments.exp023_penalised_thc import DELTA, _objective, one_norm
from experiments.exp031_gauge_structure import (
    gauge_generators,
    smoothed_penalty_violation,
    symbolic_gauge_invariance,
)
from experiments.exp022_optimised_thc import thc_tensor


def test_the_invariance_is_symbolic_not_numerical():
    """Free symbols throughout, so a zero is an identity in the polynomial ring.

    A numerical check on random draws could pass by coincidence on a measure-zero
    set; this cannot.
    """
    proof = symbolic_gauge_invariance(orbitals=3, rank=2)
    assert proof["tensor_invariant"], "the reconstructed tensor is not gauge invariant"
    assert proof["one_norm_invariant"], "the 1-norm is not gauge invariant"
    assert proof["one_norm_residual"] == "0"
    assert proof["tensor_terms_checked"] == 3**4


def test_the_invariance_holds_at_another_shape():
    """Not a coincidence of one (orbitals, rank) pair."""
    proof = symbolic_gauge_invariance(orbitals=2, rank=3)
    assert proof["tensor_invariant"] and proof["one_norm_invariant"]


def test_the_gauge_is_the_only_scaling_that_works():
    """The exponents matter: Z must scale as 1/(s_m^2 s_n^2), not 1/(s_m s_n).

    Without this the test above would pass for a whole family of wrong
    transformations and would not pin the group.
    """
    chi = sp.Matrix(2, 2, lambda p, m: sp.Symbol(f"c{p}{m}", real=True))
    coupling = sp.Matrix(2, 2, lambda m, n: sp.Symbol(f"z{m}{n}", real=True))
    scale = [sp.Symbol(f"s{m}", positive=True) for m in range(2)]

    def entry(chi_, coupling_):
        return sp.expand(sum(
            chi_[0, m] * chi_[0, m] * coupling_[m, n] * chi_[1, n] * chi_[1, n]
            for m in range(2) for n in range(2)
        ))

    gauged_chi = sp.Matrix(2, 2, lambda p, m: scale[m] * chi[p, m])
    wrong = sp.Matrix(2, 2, lambda m, n: coupling[m, n] / (scale[m] * scale[n]))
    assert sp.simplify(entry(chi, coupling) - entry(gauged_chi, wrong)) != 0


def test_generators_are_the_tangent_of_the_finite_transformation():
    """The numerical null-space check is only meaningful if these are right."""
    rng = np.random.default_rng(0)
    orbitals, rank = 4, 3
    chi = rng.normal(size=(orbitals, rank))
    coupling = rng.normal(size=(rank, rank))
    coupling = 0.5 * (coupling + coupling.T)

    generators = gauge_generators(chi, coupling)
    epsilon = 1e-6
    for k in range(rank):
        scales = np.ones(rank)
        scales[k] = np.exp(epsilon)
        moved_chi = chi * scales
        moved_coupling = coupling / np.outer(scales**2, scales**2)

        finite = np.concatenate([
            (moved_chi - chi).ravel(), (moved_coupling - coupling).ravel()
        ]) / epsilon
        np.testing.assert_allclose(finite, generators[k], rtol=1e-4, atol=1e-8)


def test_the_tensor_is_invariant_under_a_finite_gauge_numerically():
    """The symbolic proof, checked once against the code that is actually run."""
    rng = np.random.default_rng(3)
    chi = rng.normal(size=(4, 3))
    coupling = rng.normal(size=(3, 3))
    coupling = 0.5 * (coupling + coupling.T)
    scales = np.array([0.4, 1.7, 2.9])

    np.testing.assert_allclose(
        thc_tensor(chi, coupling),
        thc_tensor(chi * scales, coupling / np.outer(scales**2, scales**2)),
        atol=1e-10,
    )


def test_the_one_norm_does_not_move_along_the_orbit():
    """The claim that kills 'a penalty could do the same job'."""
    rng = np.random.default_rng(4)
    chi = rng.normal(size=(4, 3))
    coupling = rng.normal(size=(3, 3))
    coupling = 0.5 * (coupling + coupling.T)

    base = one_norm(chi, coupling)
    for scales in (np.array([0.5, 1.0, 2.0]), np.array([3.0, 0.2, 1.1])):
        moved = one_norm(chi * scales, coupling / np.outer(scales**2, scales**2))
        assert moved == pytest.approx(base, rel=1e-12)


def test_the_residual_does_not_move_along_the_orbit():
    """Same for the fitting objective, at alpha = 0."""
    rng = np.random.default_rng(5)
    orbitals, rank = 4, 3
    target = rng.normal(size=(orbitals,) * 4)
    target = 0.5 * (target + target.transpose(2, 3, 0, 1))
    target = 0.5 * (target + target.transpose(1, 0, 2, 3))
    target = 0.5 * (target + target.transpose(0, 1, 3, 2))

    chi = rng.normal(size=(orbitals, rank))
    coupling = rng.normal(size=(rank, rank))
    scales = np.array([0.6, 1.4, 2.2])

    before, _ = _objective(np.concatenate([chi.ravel(), coupling.ravel()]),
                           orbitals, rank, target, 0.0)
    after, _ = _objective(
        np.concatenate([(chi * scales).ravel(),
                        (coupling / np.outer(scales**2, scales**2)).ravel()]),
        orbitals, rank, target, 0.0)
    assert after == pytest.approx(before, rel=1e-10)


def test_the_smoothed_penalty_is_the_part_that_breaks():
    """Bounded, nonzero, and attributable to delta rather than to the 1-norm.

    If this ever returned exactly zero the smoothing would be invariant too and
    the experiment's second section would have nothing to report.
    """
    violation = smoothed_penalty_violation(samples=200, seed=1)
    assert violation["worst_relative"] > 0.0, "the smoothing must break invariance"
    assert violation["worst_relative"] < 1e-2, (
        f"the smoothing breaks invariance by {violation['worst_relative']:.1e}, "
        "which is large enough that delta is no longer a small perturbation"
    )
    assert violation["delta"] == DELTA
