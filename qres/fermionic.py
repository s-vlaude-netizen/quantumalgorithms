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


def double_excitation(
    theta: float | ParameterExpression, z_string: int = 0
) -> QuantumCircuit:
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

    ``z_string`` adds that many extra qubits (appended after the four acting
    ones) carrying the Jordan-Wigner parity string.  The excitation operator is
    ``G (x) Z_string``, so the rotation runs *backwards* on the odd-parity
    branch -- and conjugating the rotation target by a CNOT from each string
    qubit does exactly that, since ``X RY(t) X = RY(-t)``.  Two CNOTs per string
    qubit, which is far cheaper than conditioning the whole gate.
    """
    qc = QuantumCircuit(4 + z_string, name="G2")

    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(0, 2)

    # rotation on qubit 0, active only when q1=0, q2=1, q3=0.
    # The CNOT relabelling above sends |0011> to the q0=1 branch, so a positive
    # RY here would rotate |0011> towards |1100> with the *opposite* sign to the
    # convention in double_excitation_matrix.  Hence -theta.
    qc.x(1)
    qc.x(3)
    for q in range(4, 4 + z_string):
        qc.cx(q, 0)
    qc.mcry(-theta, [1, 2, 3], 0)
    for q in reversed(range(4, 4 + z_string)):
        qc.cx(q, 0)
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


def paired_double_excitations(num_spatial_orbitals: int, num_particles) -> list[tuple[int, ...]]:
    """Qubit quadruples for every paired double excitation, Jordan-Wigner.

    With blocked-spin ordering the spin-orbitals are
    ``[a_0 .. a_{M-1}, b_0 .. b_{M-1}]``, so moving *both* electrons of spatial
    orbital ``i`` into spatial orbital ``a`` touches qubits
    ``(i, M+i, a, M+a)`` -- in the order this module's gate expects:
    occupied-alpha, occupied-beta, virtual-alpha, virtual-beta.
    """
    m = num_spatial_orbitals
    n_alpha = num_particles[0] if isinstance(num_particles, (tuple, list)) else num_particles // 2
    occupied = range(n_alpha)
    virtual = range(n_alpha, m)
    return [(i, m + i, a, m + a) for i in occupied for a in virtual]


def puccd_shallow(problem) -> QuantumCircuit:
    """PUCCD built from direct Givens rotations instead of Pauli exponentials.

    Verified to reproduce ``qiskit_nature``'s ``PUCCD`` exactly (to 1e-13) up to
    the factor-of-two angle convention -- qiskit-nature's parameter is half this
    module's rotation angle.

    Requires the Jordan-Wigner mapping.  Each paired excitation carries the
    parity of the spin-orbitals lying *between* its occupied and virtual indices,
    in both spin blocks.  On H2 that set is empty -- orbitals 0 and 1 are
    adjacent -- which is why an early version with no string handling matched
    H2 exactly and H4 not at all.
    """
    from qiskit.circuit import ParameterVector

    from .ansatz import hartree_fock_state

    if problem.metadata.get("mapper") != "jordan_wigner":
        raise ValueError(
            "puccd_shallow needs the jordan_wigner mapping; "
            f"problem uses {problem.metadata.get('mapper')!r}"
        )

    quads = paired_double_excitations(problem.num_spatial_orbitals, problem.num_particles)
    theta = ParameterVector("t", len(quads))
    qc = QuantumCircuit(problem.num_qubits, name="puccd_shallow")
    qc.compose(hartree_fock_state(problem.hf_bitstring), inplace=True)
    m = problem.num_spatial_orbitals
    for k, (occ_a, occ_b, vir_a, vir_b) in enumerate(quads):
        string = [q for q in range(occ_a + 1, vir_a)] + [
            q for q in range(occ_b + 1, vir_b)
        ]
        gate = double_excitation(2 * theta[k], z_string=len(string))
        qc.compose(gate, qubits=[occ_a, occ_b, vir_a, vir_b] + string, inplace=True)
    return qc
