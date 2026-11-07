"""Benchmark I circuit (exp046): the circuit computes exactly its integer model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments import exp046_inverse_sqrt_phasing as e
from qres import kmlite as km
from qres.bench import RESULTS_DIR


@pytest.fixture(scope="module")
def built():
    design = e.Design()
    tables = e.cached_tables(design, RESULTS_DIR / "raw" / "exp046")
    builder, circuit, _ = e.build_circuit(design, tables)
    return design, tables, builder, circuit


def test_model_error_is_within_two_to_the_minus_24_everywhere(built):
    design, tables, _, _ = built
    relative, _ = e.phase_error(design, tables, e.all_inputs())
    assert np.abs(relative).max() < 2.0 ** -24


def test_circuit_phase_equals_the_integer_model_exactly(built):
    design, tables, _, circuit = built
    x = np.random.default_rng(11).integers(1, e.X_MAX + 1, size=20000)
    x = np.concatenate([x, [1, 2, 3, 4, 5, 15, 16, 17, 255, 256, 1 << 20, (1 << 20) + 1, e.X_MAX]])
    sim = km.Simulator(circuit, len(x), seed=5)
    sim.write(circuit.registers["x"], x)
    sim.run()
    assert (sim.read(circuit.registers["x"]) == x.astype(np.uint64)).all()
    assert int(sim.dirty_qubits(circuit.registers["x"]).sum()) == 0 and not sim.failed().any()
    shift, mantissa = e.normalise(x)
    y0, _ = e.evaluate(design, tables[0], tables[1], tables[2], mantissa)
    # bit p of the shifted y0 phases by 2^(p - 10 - f0) turns, i.e. 2^(54 - f0 + p) units
    expected = (y0.astype(object) * 2 ** (64 - 10 - design.f0) * 2 ** shift.astype(object)) % (1 << 64)
    assert (sim.phase_units().astype(object) == expected).all()


def test_circuit_meets_the_tolerance_on_every_input_and_beats_the_reference(built):
    _, _, builder, circuit = built
    result = e.verify(circuit, e.all_inputs())
    assert result["restored"] and result["dirty_shots"] == 0 and result["assert_failures"] == 0
    assert result["max_error_over_tolerance"] < 1.0 and result["margin_turns"] > 0
    cost = e.score(builder, result["margin_turns"])
    assert cost["qubits"] <= 163 and cost["toffoli"] <= 1826
    assert cost["score"] < e.REFERENCE["score"] / 5
    assert math.isfinite(cost["t_per_rotation"])
