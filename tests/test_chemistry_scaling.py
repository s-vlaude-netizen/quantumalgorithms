"""Correctness of the molecule set and the coefficient sum that scales from it.

`Sum|c|` sets the entire shot budget (`shots ~ (Sum|c|)^2 n / eps^2`), so a slip
in how it is computed propagates straight into the project's headline scaling
claim.  Two specific ways it can go wrong, both of which happened:

* **Including the identity term.**  It carries no variance -- a constant on
  every shot -- so it must not enter a shot count.  Including it inflates H2O
  from 72.00 to 127.61, and a scaling study built on that reported a
  "correction" that was an artefact of its own arithmetic.
* **Mislabelling the orbital count.**  The parity mapping with two-qubit
  reduction means `num_qubits // 2` is one short of the spatial orbital count,
  which shifts every point in a log-log fit.

The nuclear charges are the second variable in Result 53, so they are checked
against the geometries rather than trusted as a hand-written table.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from qres.problems.chemistry import GEOMETRIES, NUCLEAR_CHARGE, build_molecule

ATOMIC_NUMBER = {"H": 1, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9}


def coefficient_sum(problem, include_identity: bool = False) -> float:
    return float(
        sum(
            abs(complex(coeff))
            for pauli, coeff in zip(problem.hamiltonian.paulis, problem.hamiltonian.coeffs)
            if include_identity or np.any(pauli.z) or np.any(pauli.x)
        )
    )


def test_nuclear_charges_match_the_geometries():
    """The table is a claim about each geometry; check it against the atoms."""
    for name, charge in NUCLEAR_CHARGE.items():
        assert name in GEOMETRIES, f"{name} has a charge but no geometry"
        atoms = re.findall(r"([A-Z][a-z]?)\s", GEOMETRIES[name]() + " ")
        computed = sum(ATOMIC_NUMBER[a] for a in atoms)
        assert computed == charge, f"{name}: geometry gives {computed}, table says {charge}"


def test_every_geometry_has_a_charge():
    for name in GEOMETRIES:
        assert name in NUCLEAR_CHARGE, f"{name} needs an entry in NUCLEAR_CHARGE"


@pytest.mark.parametrize("name", ["H2", "H4", "LiH", "HF"])
def test_identity_is_excluded_from_the_coefficient_sum(name):
    """The identity dominates and contributes nothing to shot noise."""
    problem = build_molecule(name)
    with_identity = coefficient_sum(problem, include_identity=True)
    without = coefficient_sum(problem)
    assert without < with_identity
    assert without > 0


def test_recorded_coefficient_sums_are_reproduced():
    """The five values Result 42 was built on, as a regression anchor.

    If any of these move, the scaling discussion in RESEARCH_LOG has to be
    re-derived rather than quietly inheriting a different number.
    """
    expected = {"H2": 0.99, "H4": 8.47, "LiH": 12.34, "BeH2": 21.52, "H2O": 72.00}
    for name, value in expected.items():
        assert coefficient_sum(build_molecule(name)) == pytest.approx(value, abs=0.01)


@pytest.mark.parametrize("name", ["H6", "HF", "NH3", "BH3"])
def test_added_geometries_build(name):
    """The molecules added to break the N/Z collinearity must actually work."""
    problem = build_molecule(name)
    assert problem.hamiltonian.num_qubits > 0
    assert len(problem.hamiltonian) > 1
    assert np.isfinite(problem.hartree_fock_energy)


def test_the_fixed_charge_group_really_shares_a_charge():
    """Result 53's part 2 is only meaningful if Z is genuinely constant."""
    group = ["HF", "H2O", "NH3", "CH4"]
    charges = {NUCLEAR_CHARGE[name] for name in group}
    assert charges == {10}


def test_coefficient_sum_is_flat_at_fixed_nuclear_charge():
    """Adding atoms at constant Z does not grow Sum|c| -- the Result 53 claim.

    Stated as a bound rather than an exact exponent: the measured fit is
    N^-0.39, and what matters is that it is nowhere near the N^2.78 a
    one-variable law predicts.
    """
    orbitals, sums = [], []
    for name in ("HF", "H2O", "NH3", "CH4"):
        problem = build_molecule(name)
        orbitals.append(problem.num_spatial_orbitals)
        sums.append(coefficient_sum(problem))

    exponent = np.polyfit(np.log(orbitals), np.log(sums), 1)[0]
    assert exponent < 0.5, f"expected flat at fixed charge, got N^{exponent:.2f}"


def test_coefficient_sum_grows_steeply_with_basis_at_fixed_molecule():
    """The other direction: same molecule, richer basis, Sum|c| climbs."""
    small = coefficient_sum(build_molecule("H2", basis="sto3g"))
    larger = coefficient_sum(build_molecule("H2", basis="631g"))
    assert larger > 5 * small


def test_orbital_count_is_read_from_the_problem_not_the_qubit_count():
    """H2 at STO-3G has two spatial orbitals but only two qubits after parity.

    `num_qubits // 2 + 1` agrees under the default parity mapping with two-qubit
    reduction -- which always removes exactly two, active space or not -- and
    stops agreeing the moment the mapper changes.  A log-log fit built on the
    arithmetic instead of the recorded count would shift every point by one.
    """
    problem = build_molecule("H2", basis="sto3g")
    assert problem.hamiltonian.num_qubits == 2
    assert problem.num_spatial_orbitals == 2

    # Jordan-Wigner keeps 2N qubits, so the shortcut is off by one there
    jw = build_molecule("LiH", mapper="jordan_wigner", two_qubit_reduction=False)
    assert jw.hamiltonian.num_qubits == 2 * jw.num_spatial_orbitals
    assert jw.num_spatial_orbitals != jw.hamiltonian.num_qubits // 2 + 1
