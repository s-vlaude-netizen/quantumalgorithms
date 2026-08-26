"""Classical baselines, and the anchoring they provide."""

from __future__ import annotations

import pytest

from qres.classical import classical_baselines
from qres.problems.chemistry import CHEMICAL_ACCURACY_HA, build_molecule


def test_baselines_are_on_the_same_energy_scale_as_the_hamiltonian():
    """PySCF reports totals including nuclear repulsion; the qubit Hamiltonian
    does not.  Getting that subtraction wrong shifts every comparison by ~1 Ha,
    which is 600x chemical accuracy.
    """
    problem = build_molecule("H2")
    results = {r.method: r for r in classical_baselines(problem, methods=("HF", "FCI"))}
    assert results["HF"].energy == pytest.approx(problem.hartree_fock_energy, abs=1e-8)
    assert results["FCI"].energy == pytest.approx(problem.fci_energy, abs=1e-8)


@pytest.mark.parametrize("molecule", ["H2", "H4"])
def test_classical_methods_reach_chemical_accuracy_in_milliseconds(molecule):
    """The anchor for every shot count in this repository.

    CCSD and FCI solve these on one CPU core faster than a single VQE energy
    evaluation takes to simulate. Recorded as a test so it cannot quietly stop
    being true, or quietly stop being mentioned.
    """
    problem = build_molecule(molecule)
    results = {r.method: r for r in classical_baselines(problem, methods=("CCSD", "FCI"))}
    assert results["CCSD"].error < CHEMICAL_ACCURACY_HA
    assert results["FCI"].error < 1e-9
    assert results["FCI"].seconds < 5.0


def test_hartree_fock_is_the_bar_a_vqe_run_must_clear():
    """A run that does not beat HF has recovered no correlation energy at all."""
    problem = build_molecule("H4")
    results = {r.method: r for r in classical_baselines(problem, methods=("HF",))}
    assert results["HF"].error == pytest.approx(abs(problem.correlation_energy), abs=1e-8)
