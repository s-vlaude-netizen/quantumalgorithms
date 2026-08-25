"""Resource accounting for quantum algorithms.

The wall-clock time of a variational quantum algorithm on real hardware is
dominated by *circuit executions*, not by the classical post-processing that a
simulator makes visible.  Optimising a simulator's wall-clock time therefore
optimises the wrong thing.  This module defines the cost model we actually
optimise against:

    hardware_seconds =  n_shots   * (circuit_duration + reset_delay)
                      + n_circuits * per_circuit_overhead

``circuit_duration`` is taken from the *transpiled* circuit using the target's
real instruction durations, so a shallower circuit is genuinely cheaper here.
Every estimator in this package reports into a :class:`ResourceLedger`, which
makes the comparison between two algorithms an apples-to-apples one.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any


# IBM superconducting devices need a delay between shots for the qubits to
# relax back to |0>.  ~250 us is a representative value for the reset-by-delay
# strategy used on the Eagle/Heron families; active reset is faster but not
# universally enabled.  Held as a module constant so studies can vary it.
DEFAULT_RESET_DELAY_S = 250e-6

# Submitting a distinct circuit to the control electronics costs a fixed
# amount of load/compile time on top of its shots.  Small, but it is what makes
# "many tiny circuits" worse than "few big ones" at equal shot count.
DEFAULT_CIRCUIT_OVERHEAD_S = 1e-3


@dataclass
class ResourceLedger:
    """Accumulates the quantum resources consumed by a run."""

    shots: int = 0
    circuits: int = 0
    #: sum over submitted circuits of (duration_seconds * shots_for_that_circuit)
    shot_seconds: float = 0.0
    #: classical seconds spent inside the algorithm (optimiser, grouping, ...)
    classical_seconds: float = 0.0
    #: simulator seconds; reported for transparency, never optimised against
    simulator_seconds: float = 0.0
    #: max two-qubit gate count over circuits submitted (a noise proxy)
    max_two_qubit_gates: int = 0
    max_depth: int = 0
    #: free-form counters subclasses/experiments want to track
    counters: dict[str, float] = field(default_factory=dict)

    def record_execution(
        self,
        *,
        shots: int,
        duration_s: float,
        n_circuits: int = 1,
        two_qubit_gates: int = 0,
        depth: int = 0,
    ) -> None:
        """Record ``n_circuits`` circuits each run for ``shots`` shots."""
        self.shots += shots * n_circuits
        self.circuits += n_circuits
        self.shot_seconds += shots * n_circuits * duration_s
        self.max_two_qubit_gates = max(self.max_two_qubit_gates, two_qubit_gates)
        self.max_depth = max(self.max_depth, depth)

    def bump(self, key: str, amount: float = 1.0) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + amount

    def hardware_seconds(
        self,
        reset_delay_s: float = DEFAULT_RESET_DELAY_S,
        circuit_overhead_s: float = DEFAULT_CIRCUIT_OVERHEAD_S,
    ) -> float:
        """Estimated time this run would take on the device."""
        return (
            self.shot_seconds
            + self.shots * reset_delay_s
            + self.circuits * circuit_overhead_s
        )

    def merge(self, other: "ResourceLedger") -> None:
        self.shots += other.shots
        self.circuits += other.circuits
        self.shot_seconds += other.shot_seconds
        self.classical_seconds += other.classical_seconds
        self.simulator_seconds += other.simulator_seconds
        self.max_two_qubit_gates = max(self.max_two_qubit_gates, other.max_two_qubit_gates)
        self.max_depth = max(self.max_depth, other.max_depth)
        for k, v in other.counters.items():
            self.bump(k, v)

    @contextmanager
    def time_classical(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.classical_seconds += time.perf_counter() - t0

    @contextmanager
    def time_simulator(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.simulator_seconds += time.perf_counter() - t0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hardware_seconds"] = self.hardware_seconds()
        return d

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"shots={self.shots:,} circuits={self.circuits:,} "
            f"hw={self.hardware_seconds():.2f}s "
            f"classical={self.classical_seconds:.2f}s "
            f"sim={self.simulator_seconds:.2f}s "
            f"2q_max={self.max_two_qubit_gates}"
        )


def circuit_duration_seconds(circuit, target) -> float:
    """Duration of an ISA circuit under the backend's real gate timings.

    Falls back to a depth-based estimate when the target carries no durations
    (some fake backends and all ideal simulators).
    """
    try:
        # Qiskit's scheduling analysis needs a fully-scheduled circuit; the
        # cheap and robust route is to sum the critical path ourselves via the
        # target's per-instruction durations.
        return _critical_path_duration(circuit, target)
    except Exception:
        # 500 ns per layer is a rough stand-in for a 2q-dominated circuit.
        return circuit.depth() * 500e-9


def _critical_path_duration(circuit, target) -> float:
    """Longest-path duration through the circuit using per-qubit availability."""
    available = [0.0] * circuit.num_qubits
    index = {q: i for i, q in enumerate(circuit.qubits)}
    for inst in circuit.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qubits = [index[q] for q in inst.qubits]
        try:
            dur = target[name][tuple(qubits)].duration
        except (KeyError, TypeError):
            dur = None
        if dur is None:
            dur = 68e-9 if len(qubits) == 1 else 500e-9
        start = max(available[q] for q in qubits)
        for q in qubits:
            available[q] = start + dur
    return max(available) if available else 0.0


def dump_json(path, payload) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
