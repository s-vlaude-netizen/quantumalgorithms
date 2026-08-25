"""Ansatz properties that experiments silently depend on."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qres.ansatz import build_ansatz, hardware_efficient, hartree_fock_state
from qres.problems.chemistry import build_molecule


def test_hartree_fock_state_matches_bitstring():
    qc = hartree_fock_state("0110")
    sv = Statevector(qc)
    idx = int(np.argmax(np.abs(sv.data)))
    assert format(idx, "04b") == "0110"


@pytest.mark.parametrize("molecule", ["H2", "H4"])
@pytest.mark.parametrize("entangler", ["cry", "crx", "crz"])
def test_controlled_entanglers_preserve_the_reference(molecule, entangler):
    """At theta = 0 the ansatz must BE the Hartree-Fock reference.

    This is the property that makes "start near zero" a meaningful strategy.
    Fixed CX entanglers do not have it -- see the companion test below -- and
    an optimiser that starts on a state orthogonal to the reference never
    recovers within a realistic shot budget.
    """
    problem = build_molecule(molecule)
    ansatz = build_ansatz(f"hea:2:linear:{entangler}", problem)
    zero = Statevector(ansatz.assign_parameters(np.zeros(ansatz.num_parameters)))
    reference = Statevector(hartree_fock_state(problem.hf_bitstring))

    assert abs(zero.inner(reference)) == pytest.approx(1.0, abs=1e-10)
    energy = zero.expectation_value(problem.hamiltonian).real
    assert energy == pytest.approx(problem.hartree_fock_energy, abs=1e-9)


def test_fixed_cx_entanglers_destroy_the_reference():
    """Documents the trap, so nobody 'fixes' the cry default back to cx.

    Measured on H4: overlap with Hartree-Fock is exactly zero and the
    zero-parameter energy is 4 Ha above it.
    """
    problem = build_molecule("H4")
    ansatz = build_ansatz("hea:2", problem)
    zero = Statevector(ansatz.assign_parameters(np.zeros(ansatz.num_parameters)))
    reference = Statevector(hartree_fock_state(problem.hf_bitstring))

    assert abs(zero.inner(reference)) < 1e-9
    assert zero.expectation_value(problem.hamiltonian).real > problem.hartree_fock_energy + 3.0


@pytest.mark.parametrize("entangler", ["cx", "cry"])
def test_ansatz_can_reach_the_ground_state(entangler):
    """Expressibility check: exact optimisation must find H2's ground state."""
    from scipy.optimize import minimize

    problem = build_molecule("H2")
    spec = "hea:2" if entangler == "cx" else f"hea:2:linear:{entangler}"
    ansatz = build_ansatz(spec, problem)

    def energy(x):
        return Statevector(ansatz.assign_parameters(x)).expectation_value(problem.hamiltonian).real

    best = min(
        minimize(
            energy,
            np.random.default_rng(s).normal(0, 0.5, ansatz.num_parameters),
            method="BFGS",
        ).fun
        for s in range(4)
    )
    assert best == pytest.approx(problem.fci_energy, abs=1e-6)


def test_device_entanglement_uses_only_coupling_map_edges():
    from qres.noise import make_environment

    env = make_environment("medium")
    edges = {tuple(sorted(e)) for e in env.coupling_map.get_edges()}
    qc = hardware_efficient(6, reps=1, entanglement="device", coupling_map=env.coupling_map)
    index = {q: i for i, q in enumerate(qc.qubits)}
    for inst in qc.data:
        if len(inst.qubits) == 2:
            pair = tuple(sorted(index[q] for q in inst.qubits))
            assert pair in edges
