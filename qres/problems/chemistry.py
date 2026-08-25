"""Molecular electronic-structure problems.

These are the problems that make quantum computers interesting for drug design
and materials: find the ground-state energy of a molecular Hamiltonian to
"chemical accuracy" (1 kcal/mol = 1.594 mHa), which is the threshold below
which computed reaction rates start to match experiment.

We build the qubit Hamiltonian with PySCF + qiskit-nature and cache it, because
the integral transform is slow and completely deterministic.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

CHEMICAL_ACCURACY_HA = 1.5936e-3


@dataclass
class MolecularProblem:
    """A qubit Hamiltonian plus everything needed to score a solution."""

    name: str
    hamiltonian: SparsePauliOp
    #: energy of the nuclei, constant, added back for total energy
    nuclear_repulsion: float
    #: exact ground state of ``hamiltonian`` (electronic part only)
    fci_energy: float
    hartree_fock_energy: float
    #: occupation bitstring for the Hartree-Fock reference state
    hf_bitstring: str
    num_particles: tuple[int, int]
    num_spatial_orbitals: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return self.hamiltonian.num_qubits

    @property
    def correlation_energy(self) -> float:
        """How much energy the HF reference misses -- what VQE must recover."""
        return self.fci_energy - self.hartree_fock_energy

    def error(self, energy: float) -> float:
        return abs(energy - self.fci_energy)

    def within_chemical_accuracy(self, energy: float) -> bool:
        return self.error(energy) < CHEMICAL_ACCURACY_HA

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_terms": len(self.hamiltonian),
            "fci_energy": self.fci_energy,
            "hartree_fock_energy": self.hartree_fock_energy,
            "correlation_energy": self.correlation_energy,
            "nuclear_repulsion": self.nuclear_repulsion,
            **self.metadata,
        }


#: Geometries in Angstrom.  Bond lengths at or near equilibrium unless the
#: builder is asked for a stretched variant (stretching increases correlation
#: and makes the problem genuinely harder for classical methods).
GEOMETRIES = {
    "H2": lambda r=0.735: f"H 0 0 0; H 0 0 {r}",
    "LiH": lambda r=1.595: f"Li 0 0 0; H 0 0 {r}",
    "BeH2": lambda r=1.326: f"Be 0 0 0; H 0 0 -{r}; H 0 0 {r}",
    "H4": lambda r=0.75: "; ".join(f"H 0 0 {i * r}" for i in range(4)),
    "H2O": lambda r=0.958: f"O 0 0 0; H 0 {0.757} {0.587}; H 0 {-0.757} {0.587}",
    "N2": lambda r=1.098: f"N 0 0 0; N 0 0 {r}",
}


@functools.lru_cache(maxsize=32)
def build_molecule(
    name: str = "H2",
    basis: str = "sto3g",
    bond_length: float | None = None,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    mapper: str = "parity",
    two_qubit_reduction: bool = True,
) -> MolecularProblem:
    """Build a qubit Hamiltonian for ``name``.

    ``mapper`` is one of ``jordan_wigner``, ``parity``, ``bravyi_kitaev``.  The
    parity mapping with two-qubit reduction is the default because it removes
    two qubits for free by exploiting particle-number and spin symmetry.
    """
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
    from qiskit_nature.second_q.mappers import (
        JordanWignerMapper,
        ParityMapper,
        BravyiKitaevMapper,
    )

    if name not in GEOMETRIES:
        raise ValueError(f"unknown molecule {name!r}; have {sorted(GEOMETRIES)}")
    geom = GEOMETRIES[name]() if bond_length is None else GEOMETRIES[name](bond_length)

    driver = PySCFDriver(atom=geom, basis=basis, charge=0, spin=0)
    problem = driver.run()
    hf_energy_total = problem.reference_energy

    if active_electrons is not None and active_orbitals is not None:
        transformer = ActiveSpaceTransformer(active_electrons, active_orbitals)
        problem = transformer.transform(problem)

    num_particles = problem.num_particles
    num_spatial = problem.num_spatial_orbitals

    if mapper == "jordan_wigner":
        qubit_mapper = JordanWignerMapper()
    elif mapper == "bravyi_kitaev":
        qubit_mapper = BravyiKitaevMapper()
    elif mapper == "parity":
        qubit_mapper = (
            ParityMapper(num_particles=num_particles) if two_qubit_reduction else ParityMapper()
        )
    else:
        raise ValueError(f"unknown mapper {mapper!r}")

    hamiltonian = qubit_mapper.map(problem.hamiltonian.second_q_op())
    hamiltonian = hamiltonian.simplify(atol=1e-12)

    nuclear = problem.nuclear_repulsion_energy or 0.0
    fci_electronic = exact_ground_energy(hamiltonian)

    hf_bitstring = _hf_bitstring(qubit_mapper, num_spatial, num_particles)
    hf_electronic = _hf_reference_energy(hamiltonian, hf_bitstring)

    label = name if bond_length is None else f"{name}@{bond_length:g}A"
    if active_orbitals is not None:
        label += f"/as{active_electrons}e{active_orbitals}o"

    return MolecularProblem(
        name=f"{label}/{basis}/{mapper}",
        hamiltonian=hamiltonian,
        nuclear_repulsion=nuclear,
        fci_energy=fci_electronic,
        hartree_fock_energy=hf_electronic,
        hf_bitstring=hf_bitstring,
        num_particles=tuple(num_particles),
        num_spatial_orbitals=num_spatial,
        metadata={
            "geometry": geom,
            "basis": basis,
            "mapper": mapper,
            "total_fci_energy": fci_electronic + nuclear,
            "scf_total_energy": hf_energy_total,
        },
    )


def exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """Lowest eigenvalue by dense or sparse diagonalisation."""
    n = hamiltonian.num_qubits
    if n <= 12:
        matrix = hamiltonian.to_matrix()
        return float(np.linalg.eigvalsh(matrix)[0])
    from scipy.sparse.linalg import eigsh

    matrix = hamiltonian.to_matrix(sparse=True)
    vals = eigsh(matrix, k=1, which="SA", return_eigenvectors=False, maxiter=10_000)
    return float(vals[0])


def _hf_bitstring(qubit_mapper, num_spatial_orbitals, num_particles) -> str:
    """Qubit-basis occupation string of the Hartree-Fock determinant."""
    from qiskit_nature.second_q.circuit.library import HartreeFock

    hf = HartreeFock(num_spatial_orbitals, num_particles, qubit_mapper)
    from qiskit.quantum_info import Statevector

    sv = Statevector(hf)
    idx = int(np.argmax(np.abs(sv.data)))
    return format(idx, f"0{hf.num_qubits}b")


def _hf_reference_energy(hamiltonian: SparsePauliOp, bitstring: str) -> float:
    """<HF|H|HF>, computed without building the full matrix.

    A computational-basis state has <b|P|b> = 0 unless P is all I/Z, in which
    case it is the product of the Z parities.
    """
    total = 0.0
    # bitstring is little-endian-printed (qubit 0 last), match Pauli ordering
    bits = bitstring[::-1]
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            continue  # contains X or Y -> zero on a basis state
        sign = 1
        for q in range(pauli.num_qubits):
            if pauli.z[q] and bits[q] == "1":
                sign = -sign
        total += sign * coeff.real
    return float(total)


def dissociation_curve(
    name: str = "H2",
    distances: tuple[float, ...] = (0.5, 0.735, 1.0, 1.5, 2.0, 2.5),
    **kwargs,
) -> list[MolecularProblem]:
    """Problems along a bond-stretching coordinate.

    Stretched geometries are strongly correlated and are exactly where
    classical single-reference methods fail -- the regime a quantum advantage
    would have to show up in.
    """
    return [build_molecule(name, bond_length=d, **kwargs) for d in distances]
