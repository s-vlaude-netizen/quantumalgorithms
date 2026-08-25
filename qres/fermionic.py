"""Fermionic excitation gates compiled directly, not through Pauli exponentials.

Why this module exists (RESEARCH_LOG Results 10, 12, 13):

A hardware-efficient ansatz has **exactly zero gradient** at the Hartree-Fock
determinant, at any depth -- its first-order response reaches only single
excitations, and Brillouin's theorem decouples those from HF.  Correlation
energy lives in the *doubles*.  So the only ansätze that can move are the
coupled-cluster family, and qiskit-nature compiles those by Trotterising each
excitation into eight Pauli-string exponentials, each with its own CNOT ladder.
On H4 that turns PUCCD's **four** paired double excitations into **317**
two-qubit gates, which retains ~10% signal on a Heron-class device.

A double excitation is a Givens rotation on the two-dimensional subspace
spanned by |1100> and |0011>.  Building it directly as a two-level unitary
costs far less than eight Pauli ladders.

Everything here is verified against the exact target matrix, because a nearly
correct excitation gate produces plausible energies and silently wrong
chemistry.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression


def double_excitation_matrix(theta: float) -> np.ndarray:
    """The 16x16 target: a Givens rotation mixing |0011> and |1100>.

    Convention (matching PennyLane's ``DoubleExcitation``), with qubit 0 as the
    least significant bit so that ``|q3 q2 q1 q0>`` indexes as ``q3*8 + ... +
    q0``:

        |0011> ->  cos(t/2) |0011> + sin(t/2) |1100>
        |1100> -> -sin(t/2) |0011> + cos(t/2) |1100>

    every other computational basis state is left alone.
    """
    matrix = np.eye(16, dtype=complex)
    a = 0b0011
    b = 0b1100
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    matrix[a, a] = c
    matrix[b, b] = c
    matrix[b, a] = s
    matrix[a, b] = -s
    return matrix


def double_excitation(theta: float | ParameterExpression) -> QuantumCircuit:
    """Circuit for the double-excitation Givens rotation on 4 qubits.

    Strategy: CNOTs map the two states of interest onto a pair differing in a
    single bit, a controlled rotation acts there, and the CNOTs are undone.

        CX(0->1), CX(2->3), CX(0->2)  sends  |0011> -> |0101>,  |1100> -> |0100>

    which differ only in qubit 0, both having q1 = q3 = 0 and q2 = 1.  So the
    rotation is an RY on qubit 0 controlled on ``q1 = 0, q2 = 1, q3 = 0``.

    That controlled rotation is what costs gates.  Qiskit's ``mcry`` synthesis
    gives **26** two-qubit gates; building it from two ``mcx`` around half-angle
    RYs gives 34.  A relative-phase (``rccx``) chain would give 14, but it is
    **not correct** here -- the relative phases do not cancel between the two
    occurrences, and the resulting operator is off by 9.5e-1.  The test suite
    keeps that variant out.

    For scale: qiskit-nature's Trotterised compilation costs ~79 two-qubit gates
    per double excitation on the same problem.
    """
    qc = QuantumCircuit(4, name="G2")

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 2)

    # rotation on qubit 0, active only when q1=0, q2=1, q3=0.
    # The CNOT relabelling above sends |0011> to the q0=1 branch, so a positive
    # RY here would rotate |0011> towards |1100> with the *opposite* sign to the
    # convention in double_excitation_matrix.  Hence -theta.
    qc.x(1)
    qc.x(3)
    qc.mcry(-theta, [1, 2, 3], 0)
    qc.x(3)
    qc.x(1)

    qc.cx(0, 2)
    qc.cx(2, 3)
    qc.cx(0, 1)
    return qc


def single_excitation(theta: float | ParameterExpression) -> QuantumCircuit:
    """Givens rotation mixing |01> and |10> on two qubits.

        |01> ->  cos(t/2)|01> + sin(t/2)|10>
        |10> -> -sin(t/2)|01> + cos(t/2)|10>

    This is the number-preserving single excitation; two CNOTs and a controlled
    rotation suffice.
    """
    qc = QuantumCircuit(2, name="G1")
    qc.cx(0, 1)
    # negative angle for the same reason as in double_excitation
    qc.cry(-theta, 1, 0)
    qc.cx(0, 1)
    return qc


def single_excitation_matrix(theta: float) -> np.ndarray:
    matrix = np.eye(4, dtype=complex)
    a, b = 0b01, 0b10
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    matrix[a, a] = c
    matrix[b, b] = c
    matrix[b, a] = s
    matrix[a, b] = -s
    return matrix


def two_qubit_count(circuit: QuantumCircuit, backend=None) -> int:
    """Two-qubit gates after transpilation to a real basis."""
    from qiskit import transpile

    if backend is None:
        isa = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3)
    else:
        isa = transpile(circuit, backend, optimization_level=3)
    return sum(
        1 for inst in isa.data if len(inst.qubits) == 2 and inst.operation.name != "barrier"
    )
