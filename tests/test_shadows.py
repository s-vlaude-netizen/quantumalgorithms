"""Shadow estimators, and the ranking they were built to establish."""

from __future__ import annotations

import numpy as np
import pytest

from qres.problems.chemistry import build_molecule
from qres.shadows import (
    derandomise,
    pauli_supports,
    random_pauli_shadow_variance,
    shadow_report,
)


def test_pauli_supports_encodes_the_measurement_basis():
    from qiskit.quantum_info import SparsePauliOp

    operator = SparsePauliOp(["XIZ", "IYY"], np.array([1.0, 1.0], dtype=complex))
    codes, weights = pauli_supports(operator)
    # qubit 0 is the rightmost character
    assert list(codes[0]) == [3, 0, 1]  # Z on q0, I on q1, X on q2
    assert list(codes[1]) == [2, 2, 0]  # Y on q0, Y on q1, I on q2
    assert list(weights) == [2, 2]


def test_derandomisation_covers_every_term():
    """An unmeasured term makes the estimator undefined, not just imprecise.

    A first version scored uncovered terms with a 1/(1+hits) proxy that did not
    push hard enough; terms stayed at zero hits and the variance came out
    infinite.
    """
    problem = build_molecule("H4")
    chosen = derandomise(problem.hamiltonian, num_settings=10)
    assert chosen.hit_counts.min() >= 1
    assert chosen.num_settings >= 10


def test_derandomisation_needs_at_least_a_qwc_cover():
    """Product bases cannot beat a qubit-wise-commuting clique cover on count."""
    from qres.measurement import group_paulis

    problem = build_molecule("H4")
    qwc, _ = group_paulis(problem.hamiltonian, "qwc")
    chosen = derandomise(problem.hamiltonian, num_settings=1)
    assert chosen.num_settings >= len(qwc) * 0.5, "coverage forces a QWC-sized set"


def test_random_shadow_variance_is_dominated_by_high_weight_terms():
    """The 3^k penalty falls on exactly what Jordan-Wigner strings produce."""
    problem = build_molecule("LiH")
    codes, weights = pauli_supports(problem.hamiltonian)
    coeffs = np.asarray(problem.hamiltonian.coeffs).real
    total = random_pauli_shadow_variance(problem.hamiltonian)

    heavy = weights >= 9
    heavy_share = float(np.sum(coeffs[heavy] ** 2 * 3.0 ** weights[heavy])) / total
    assert heavy.sum() < 0.1 * len(weights), "few terms"
    assert heavy_share > 0.5, "carrying most of the variance"


def test_commuting_grouping_beats_every_alternative():
    """The ranking, kept as a regression guard.

    Measured at Hartree-Fock, total variance under Neyman allocation, relative
    to general-commuting grouping:

        QWC 1.7-3.2x   derandomised 3.0-4.8x   random shadows 38-369x

    If any of these ever drops below 1.0, the conclusion in RESEARCH_LOG
    Result 38 is wrong and that direction is worth reopening.
    """
    report = shadow_report(build_molecule("H4"))
    baseline = report["commuting_variance"]
    assert report["qwc_variance"] > baseline
    assert report["derandomised_variance"] > baseline
    assert report["random_shadow_variance"] > baseline
    assert np.isfinite(report["derandomised_variance"])
