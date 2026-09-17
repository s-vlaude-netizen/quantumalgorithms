"""The trapped-ion comparison (Result 81).

The load-bearing property is that the two arms differ in *one* thing. If the
basis gate set or the optimisation level differed between them, the "routing
overhead" number would be measuring the native gate set as well, and a 1.33x
result would be uninterpretable.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from experiments.exp025_trapped_ion import (
    BASIS,
    HELIOS_TWO_QUBIT_ERROR,
    IBM_BEST_TWO_QUBIT_ERROR,
    count_across_seeds,
    routed_two_qubit_count,
    survival,
)


def _long_range_circuit(qubits=10):
    """Entangles the two ends, so a line topology *must* insert SWAPs."""
    circuit = QuantumCircuit(qubits)
    circuit.h(0)
    for target in range(1, qubits):
        circuit.cx(0, target)
    return circuit


def test_routing_costs_gates_on_a_restricted_topology():
    """The premise of the whole experiment: connectivity is not free.

    If this ever fails, either the transpiler stopped inserting SWAPs or the two
    arms are no longer being compared on the same circuit -- and the routing
    factor would be meaningless either way.
    """
    circuit = _long_range_circuit(10)
    line = CouplingMap.from_line(10)

    routed = count_across_seeds(circuit, line)
    free = count_across_seeds(circuit, None)

    assert free["median"] < routed["median"], (
        f"all-to-all ({free['median']}) did not beat a line ({routed['median']})"
    )


def test_all_to_all_needs_no_swaps_on_a_circuit_that_is_already_local():
    """A local circuit must cost the same both ways, or routing is being blamed
    for something else."""
    circuit = QuantumCircuit(4)
    for control in range(3):
        circuit.cx(control, control + 1)

    line = CouplingMap.from_line(4)
    assert count_across_seeds(circuit, line)["median"] == count_across_seeds(
        circuit, None
    )["median"]


def test_both_arms_use_the_same_basis():
    """Isolating connectivity means the gate set is held fixed.

    Checked by transpiling to each arm and confirming the two-qubit gate that
    comes out is the same one; a device-target arm would emit `ecr` or `cz` and
    the counts would not be comparable.
    """
    from qiskit import transpile

    circuit = _long_range_circuit(6)
    names = set()
    for coupling_map in (None, CouplingMap.from_line(6)):
        isa = transpile(circuit, basis_gates=list(BASIS), coupling_map=coupling_map,
                        optimization_level=3, seed_transpiler=0)
        names |= {
            inst.operation.name for inst in isa.data if len(inst.qubits) == 2
        }
    assert names <= {"cx"}, f"arms emitted different two-qubit gates: {names}"


def test_survival_is_the_no_fault_probability():
    """Error *detection* discards a shot on any detected fault."""
    assert survival(0, 1e-3) == pytest.approx(1.0)
    assert survival(1, 1e-3) == pytest.approx(0.999)
    assert survival(1000, 1e-3) == pytest.approx(0.999**1000)
    # and it decays, which is the entire point of reporting it
    assert survival(10000, 7.9e-4) < 1e-3


def test_the_quoted_device_figures_are_the_published_ones():
    """Pinned, because the comparison is only as good as its inputs.

    Helios: 99.921% two-qubit fidelity. IBM: the best median two-qubit error of
    the seven devices in Result 65 (`fake_boston`). Neither is measured by this
    experiment and both are labelled that way in the output.
    """
    assert HELIOS_TWO_QUBIT_ERROR == pytest.approx(1 - 0.99921, abs=1e-6)
    assert IBM_BEST_TWO_QUBIT_ERROR == pytest.approx(1.2747e-3, rel=1e-3)
    # the advantage is a factor, not an order of magnitude -- the claim the
    # experiment exists to check
    assert 1.5 < IBM_BEST_TWO_QUBIT_ERROR / HELIOS_TWO_QUBIT_ERROR < 1.8


def test_seeds_are_reported_not_averaged_away():
    """A single transpiler seed is not a gate count; the spread has to survive."""
    circuit = _long_range_circuit(12)
    result = count_across_seeds(circuit, CouplingMap.from_line(12))
    assert result["min"] <= result["median"] <= result["max"]
    assert len(result["counts"]) >= 3


def test_routed_count_ignores_barriers():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.barrier()
    circuit.cx(0, 1)
    assert routed_two_qubit_count(circuit, None, 0) == 2
