"""Excitation gates, verified against their exact target matrices.

A nearly-correct excitation gate produces plausible energies and silently wrong
chemistry, so these are checked as operators, not by whether a VQE run looks
sensible.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector

from qres.fermionic import (
    double_excitation,
    double_excitation_matrix,
    single_excitation,
    single_excitation_matrix,
    two_qubit_count,
)

ANGLES = [0.0, 0.3, 1.1, -0.7, np.pi, 2.5, -np.pi / 3]


@pytest.mark.parametrize("theta", ANGLES)
def test_single_excitation_matches_target(theta):
    got = Operator(single_excitation(theta)).data
    np.testing.assert_allclose(got, single_excitation_matrix(theta), atol=1e-12)


@pytest.mark.parametrize("theta", ANGLES)
def test_double_excitation_matches_target(theta):
    got = Operator(double_excitation(theta)).data
    np.testing.assert_allclose(got, double_excitation_matrix(theta), atol=1e-12)


@pytest.mark.parametrize("theta", [0.4, 1.3])
def test_excitations_preserve_particle_number(theta):
    """Every basis state must map into states of the same Hamming weight.

    This is the property that makes these gates useful: they cannot leak into
    the wrong particle-number sector, where the Hamiltonian's spectrum is
    meaningless.
    """
    for circuit, n in ((single_excitation(theta), 2), (double_excitation(theta), 4)):
        matrix = Operator(circuit).data
        for column in range(2**n):
            weight = bin(column).count("1")
            for row in range(2**n):
                if bin(row).count("1") != weight:
                    assert abs(matrix[row, column]) < 1e-12


def test_double_excitation_acts_only_on_the_intended_subspace():
    """Everything outside {|0011>, |1100>} must be untouched."""
    matrix = Operator(double_excitation(0.9)).data
    for state in range(16):
        if state in (0b0011, 0b1100):
            continue
        column = matrix[:, state]
        expected = np.zeros(16, dtype=complex)
        expected[state] = 1.0
        np.testing.assert_allclose(column, expected, atol=1e-12)


def test_double_excitation_is_a_rotation_group():
    """G(a) G(b) = G(a+b), so the parameter is a genuine rotation angle."""
    a, b = 0.4, 0.9
    lhs = Operator(double_excitation(b)) @ Operator(double_excitation(a))
    rhs = Operator(double_excitation(a + b))
    np.testing.assert_allclose(lhs.data, rhs.data, atol=1e-12)


def test_double_excitation_has_gradient_at_the_reference():
    """The whole point: non-zero derivative at the Hartree-Fock-like state.

    A hardware-efficient ansatz has exactly zero gradient there (RESEARCH_LOG
    Result 10), which is why it never leaves Hartree-Fock.
    """
    reference = Statevector.from_int(0b0011, dims=(2,) * 4)
    h = 1e-6
    plus = reference.evolve(double_excitation(h))
    minus = reference.evolve(double_excitation(-h))
    derivative = (plus.data - minus.data) / (2 * h)
    assert np.linalg.norm(derivative) > 0.1


def test_parameterised_circuit_binds_correctly():
    theta = Parameter("t")
    circuit = double_excitation(theta)
    bound = circuit.assign_parameters([1.1])
    np.testing.assert_allclose(
        Operator(bound).data, double_excitation_matrix(1.1), atol=1e-12
    )


def test_gate_counts_beat_the_trotterised_compilation():
    """26 two-qubit gates against qiskit-nature's ~79 per double excitation."""
    assert two_qubit_count(single_excitation(0.3)) <= 4
    assert two_qubit_count(double_excitation(0.3)) <= 30


def test_relative_phase_shortcut_is_rejected():
    """Documents a 14-CNOT variant that is cheaper and wrong.

    Using an rccx chain in place of the multi-controlled rotation gives 14
    two-qubit gates, but the relative phases do not cancel between the two
    occurrences.  The error grows with the angle -- 0.15 at theta=0.3, 1.0 at
    theta=pi -- so the check is taken over a range rather than at one angle,
    where a small-angle value could sneak under any single threshold.

    Kept as a regression guard against 'optimising' the gate back into being
    incorrect.
    """
    errors = [_rccx_variant_error(t) for t in (0.3, 0.9, 1.1, 2.5, np.pi)]
    assert max(errors) > 0.5, "if this now passes, the cheap variant is worth adopting"


def _rccx_variant_error(theta: float) -> float:
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 2)
    qc.x(1)
    qc.x(3)
    qc.ry(-theta / 2, 0)
    qc.rccx(1, 2, 3)
    qc.cx(3, 0)
    qc.ry(theta / 2, 0)
    qc.cx(3, 0)
    qc.rccx(1, 2, 3)
    qc.x(3)
    qc.x(1)
    qc.cx(0, 2)
    qc.cx(2, 3)
    qc.cx(0, 1)
    return float(np.abs(Operator(qc).data - double_excitation_matrix(theta)).max())


def test_puccd_shallow_reproduces_qiskit_nature():
    """The shallow build must be the *same operator*, not merely similar.

    Checked as a unitary at random parameters. An early version handled no
    Jordan-Wigner Z-strings at all; it matched H2 exactly -- orbitals 0 and 1
    are adjacent so the string is empty -- and was off by 1.4 on H4. Both
    molecules are checked here for that reason.
    """
    from qiskit_nature.second_q.circuit.library import PUCCD, HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    from qres.fermionic import puccd_shallow
    from qres.problems.chemistry import build_molecule

    for molecule in ("H2", "H4"):
        problem = build_molecule(molecule, mapper="jordan_wigner")
        mapper = JordanWignerMapper()
        reference = PUCCD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=HartreeFock(
                problem.num_spatial_orbitals, problem.num_particles, mapper
            ),
        )
        theirs = QuantumCircuit(reference.num_qubits).compose(reference.decompose(reps=4))
        mine = puccd_shallow(problem)
        assert mine.num_parameters == theirs.num_parameters

        rng = np.random.default_rng(0)
        for _ in range(3):
            x = rng.normal(0, 0.6, theirs.num_parameters)
            np.testing.assert_allclose(
                Operator(mine.assign_parameters(x)).data,
                Operator(theirs.assign_parameters(x)).data,
                atol=1e-9,
                err_msg=f"{molecule}: shallow PUCCD is not the same operator",
            )


def test_puccd_shallow_uses_fewer_gates_than_the_trotterised_build():
    """Abstract two-qubit count, with device routing factored out.

    Routing is measured separately (RESEARCH_LOG Result 15) because it eats
    most of this advantage on a heavy-hex topology.
    """
    from qiskit import transpile
    from qiskit_nature.second_q.circuit.library import PUCCD, HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    from qres.fermionic import puccd_shallow
    from qres.problems.chemistry import build_molecule

    problem = build_molecule("H4", mapper="jordan_wigner")
    mapper = JordanWignerMapper()
    reference = PUCCD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper),
    )
    theirs = QuantumCircuit(reference.num_qubits).compose(reference.decompose(reps=4))

    def count(circuit):
        isa = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3)
        return sum(1 for inst in isa.data if len(inst.qubits) == 2)

    assert count(puccd_shallow(problem)) < 0.6 * count(theirs)


def test_puccd_shallow_rejects_the_wrong_mapping():
    from qres.fermionic import puccd_shallow
    from qres.problems.chemistry import build_molecule

    problem = build_molecule("H4")  # parity mapping
    with pytest.raises(ValueError, match="jordan_wigner"):
        puccd_shallow(problem)
