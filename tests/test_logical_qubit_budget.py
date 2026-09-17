"""The 50-logical-qubit sizing (Result 82).

Everything here is arithmetic on top of measured laws, which is exactly the kind
of result that goes wrong quietly: a factor of two in the qubits-per-orbital
translation, or a power law extrapolated in the wrong direction, produces a
perfectly plausible number. The translation and the fit are therefore pinned.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.exp026_logical_qubit_budget import (
    HELIOS_LOGICAL_QUBITS,
    HELIOS_PHYSICAL_QUBITS,
    determinants,
    fit_power_law,
)


def test_a_qubit_is_a_spin_orbital_not_an_atom():
    """The translation the whole result turns on.

    50 logical qubits is 25 *spatial* orbitals, because each spatial orbital
    holds two spin orbitals and each spin orbital is one qubit. Reading it as
    50 atoms overstates what the machine buys by roughly an order of magnitude.
    """
    assert 50 // 2 == 25
    # and a 50-atom organic fragment is nowhere near 25 orbitals: even a minimal
    # basis gives 5 functions per heavy atom, so ~25 heavy atoms is ~125
    minimal_basis_functions = 25 * 5 + 25 * 1
    assert minimal_basis_functions > 4 * 25


def test_determinants_counts_a_cas_space():
    """CAS(n, m) at half filling is C(m, n/2)^2 -- alpha and beta independently."""
    assert determinants(4, 4) == pytest.approx(math.comb(4, 2) ** 2)
    assert determinants(2, 2) == pytest.approx(4.0)
    # and it passes any exact-diagonalisation limit well before 25 orbitals
    assert determinants(25) > 1e12
    assert determinants(10) < 1e6


def test_determinants_grows_with_the_active_space():
    sizes = [determinants(n) for n in (8, 12, 16, 20, 24)]
    assert all(b > a for a, b in zip(sizes, sizes[1:]))


def test_power_law_fit_recovers_a_known_exponent():
    """A slope without a standard error is not a measurement in this project."""
    sizes = np.array([1, 2, 4, 8, 16], float)
    values = 3.0 * sizes**2.5
    slope, intercept, stderr = fit_power_law(sizes, values)
    assert slope == pytest.approx(2.5, abs=1e-6)
    assert np.exp(intercept) == pytest.approx(3.0, rel=1e-6)
    assert stderr < 1e-6


def test_power_law_fit_reports_uncertainty_on_noisy_data():
    rng = np.random.default_rng(0)
    sizes = np.array([1, 2, 4, 8, 16], float)
    values = 3.0 * sizes**2.5 * np.exp(rng.normal(scale=0.1, size=5))
    _, _, stderr = fit_power_law(sizes, values)
    assert stderr > 0.0 and np.isfinite(stderr)


def test_power_law_fit_refuses_too_few_points():
    """Two points give a slope with no residual and no honest error bar."""
    slope, _, stderr = fit_power_law([1, 2], [1, 4])
    assert math.isnan(slope) and math.isnan(stderr)


def test_the_helios_figures_are_the_published_ones():
    """98 physical, 48 logical at 2:1 -- an encoding rate, not a surface code."""
    assert HELIOS_PHYSICAL_QUBITS == 98
    assert HELIOS_LOGICAL_QUBITS == 48
    # 2:1 is the claim, and it is what makes these error-DETECTED qubits: a
    # distance-2 code cannot correct, so they are not comparable to the deep
    # surface code the phase-estimation row assumes
    assert HELIOS_PHYSICAL_QUBITS / HELIOS_LOGICAL_QUBITS < 2.5


def test_the_recorded_sizing_is_internally_consistent():
    """Read back the run: the shortfalls must follow from the stored inputs."""
    import json
    from pathlib import Path

    path = Path("results/exp026_logical_qubit_budget.json")
    if not path.exists():
        pytest.skip("run experiments.exp026_logical_qubit_budget first")
    data = json.loads(path.read_text())

    assert data["target_orbitals"] * 2 == 50, "the target must be the 50-qubit budget"
    assert data["needed_two_qubit_error"] == pytest.approx(
        1.6e-3 / data["predicted_two_qubit_gates"], rel=1e-9
    )
    assert data["variational_shortfall"] == pytest.approx(
        data["physical_error"] / data["needed_two_qubit_error"], rel=1e-9
    )
    assert data["qubit_shortfall"] == pytest.approx(
        data["physical_qubits"] / HELIOS_PHYSICAL_QUBITS, rel=1e-9
    )
    # the finding that makes this worth recording: the error-corrected route is
    # orders of magnitude closer than the variational one
    assert data["qubit_shortfall"] < data["variational_shortfall"] / 1000
