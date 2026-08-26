"""Classical baselines, so the quantum numbers have something to mean.

Nothing in this package was compared against what an ordinary computer does
with the same problem, which makes every shot count in it unanchored.  A shot
budget is only interesting relative to the alternative, and for the molecules
studied here the alternative is very good and very fast.

This module measures that alternative on exactly the same Hamiltonians, with
wall-clock times, so any claim about quantum resources can be stated next to it.
It is not a flattering comparison and is not meant to be.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClassicalResult:
    method: str
    energy: float
    seconds: float
    error: float
    converged: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def classical_baselines(problem, methods=("HF", "MP2", "CCSD", "CCSD(T)", "FCI")) -> list:
    """Run standard quantum-chemistry methods on the same molecule.

    Energies are returned as *electronic* energies on the same scale as
    ``problem.fci_energy`` -- PySCF reports totals including nuclear repulsion,
    so it is subtracted.  Getting that wrong would make every comparison here
    off by a constant of order 1 Ha, which is 600x chemical accuracy and would
    not be subtle.
    """
    from pyscf import cc, fci, gto, mp, scf

    mol = gto.M(
        atom=problem.metadata["geometry"],
        basis=problem.metadata["basis"],
        charge=0,
        spin=0,
        verbose=0,
    )
    nuclear = float(mol.energy_nuc())
    reference = problem.fci_energy
    results: list[ClassicalResult] = []

    t0 = time.perf_counter()
    mean_field = scf.RHF(mol)
    mean_field.kernel()
    hf_seconds = time.perf_counter() - t0
    hf_energy = float(mean_field.e_tot) - nuclear

    def record(name, energy, seconds, converged=True, **meta):
        results.append(
            ClassicalResult(
                method=name,
                energy=float(energy),
                seconds=float(seconds),
                error=abs(float(energy) - reference),
                converged=converged,
                metadata=meta,
            )
        )

    if "HF" in methods:
        record("HF", hf_energy, hf_seconds)

    if "MP2" in methods:
        t0 = time.perf_counter()
        corr = mp.MP2(mean_field).kernel()[0]
        record("MP2", hf_energy + corr, time.perf_counter() - t0 + hf_seconds)

    if "CCSD" in methods or "CCSD(T)" in methods:
        t0 = time.perf_counter()
        coupled = cc.CCSD(mean_field)
        coupled.kernel()
        ccsd_seconds = time.perf_counter() - t0 + hf_seconds
        if "CCSD" in methods:
            record(
                "CCSD",
                float(coupled.e_tot) - nuclear,
                ccsd_seconds,
                converged=bool(coupled.converged),
            )
        if "CCSD(T)" in methods:
            t1 = time.perf_counter()
            triples = coupled.ccsd_t()
            record(
                "CCSD(T)",
                float(coupled.e_tot) + float(triples) - nuclear,
                ccsd_seconds + (time.perf_counter() - t1),
                converged=bool(coupled.converged),
            )

    if "FCI" in methods:
        t0 = time.perf_counter()
        solver = fci.FCI(mean_field)
        energy = solver.kernel()[0]
        record("FCI", float(energy) - nuclear, time.perf_counter() - t0 + hf_seconds)

    return results


def baseline_table(problem, quantum_error: float | None = None,
                   quantum_shots: int | None = None) -> str:  # pragma: no cover - display
    """Format the classical baselines, optionally beside a quantum result."""
    from .problems.chemistry import CHEMICAL_ACCURACY_HA

    rows = classical_baselines(problem)
    lines = [
        f"{problem.name}  ({problem.num_qubits} qubits, "
        f"{len(problem.hamiltonian)} Pauli terms)",
        f"{'method':<12}{'error / Ha':>14}{'chem acc':>10}{'wall time':>12}",
        "-" * 48,
    ]
    for row in rows:
        mark = "yes" if row.error < CHEMICAL_ACCURACY_HA else "no"
        lines.append(f"{row.method:<12}{row.error:>14.3e}{mark:>10}{row.seconds * 1e3:>10.1f} ms")
    if quantum_error is not None:
        mark = "yes" if quantum_error < CHEMICAL_ACCURACY_HA else "no"
        shots = f"{quantum_shots:,} shots" if quantum_shots else "-"
        lines.append(f"{'VQE (best)':<12}{quantum_error:>14.3e}{mark:>10}{shots:>13}")
    return "\n".join(lines)
