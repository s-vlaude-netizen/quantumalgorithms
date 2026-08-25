"""Shot-based expectation-value estimation with honest resource accounting.

This is the object every algorithm in the package calls.  It owns the grouping,
the shot allocation, the transpiled circuits and the :class:`ResourceLedger`,
so that any two algorithms compared through it are charged for exactly the same
things.

Design notes that matter for speed:

* Circuits are transpiled **once** per group, with the ansatz parameters left
  free, and only re-bound per evaluation.  Transpiling inside the optimiser
  loop dominates everything else and makes runtime comparisons meaningless.
* Counts are processed as (unique bitstring, count) pairs rather than expanded
  to one row per shot, so a 100k-shot evaluation costs the same as a 1k-shot
  one classically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .measurement import (
    ALLOCATION_STRATEGIES,
    MeasurementGroup,
    VarianceModel,
    allocate_shots,
    group_paulis,
    grouping_report,
)
from .noise import NoiseEnvironment
from .resources import ResourceLedger, circuit_duration_seconds


@dataclass
class EstimationResult:
    value: float
    #: 1-sigma statistical uncertainty of ``value``
    stderr: float
    shots_used: int
    circuits_used: int
    per_group_shots: np.ndarray = field(default=None)


class ShotEstimator:
    """Estimate <psi(theta)|H|psi(theta)> under a noise environment."""

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        ansatz: QuantumCircuit,
        environment: NoiseEnvironment,
        *,
        grouping: str = "commuting",
        allocation: str = "adaptive",
        ledger: ResourceLedger | None = None,
        min_shots_per_group: int = 8,
        prior_strength: float = 64.0,
        optimization_level: int = 2,
    ):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.env = environment
        self.allocation = allocation
        self.ledger = ledger if ledger is not None else ResourceLedger()
        self.min_shots_per_group = min_shots_per_group

        t0 = time.perf_counter()
        self.groups, self.identity_coeff = group_paulis(hamiltonian, method=grouping)
        self.grouping_seconds = time.perf_counter() - t0
        self.variance_model = VarianceModel.from_groups(self.groups, prior_strength)

        self._isa: list[QuantumCircuit] = []
        self._durations: list[float] = []
        self._build_circuits(optimization_level)

        self._rng = np.random.default_rng(environment.seed)
        self.report = grouping_report(self.groups, self.identity_coeff)
        self.report["transpile_seconds"] = self.transpile_seconds
        self.report["grouping_seconds"] = self.grouping_seconds

    # -- setup ------------------------------------------------------------

    def _build_circuits(self, optimization_level: int) -> None:
        t0 = time.perf_counter()
        for group in self.groups:
            qc = self.ansatz.copy()
            if group.basis_change is not None:
                qc.compose(group.basis_change, inplace=True)
            qc.measure_all()
            isa = self.env.prepare(qc, optimization_level=optimization_level)
            self._isa.append(isa)
            self._durations.append(circuit_duration_seconds(isa, self.env.target))
        self.transpile_seconds = time.perf_counter() - t0

    @property
    def num_parameters(self) -> int:
        return self.ansatz.num_parameters

    def two_qubit_gate_count(self) -> int:
        return max(_two_qubit_count(c) for c in self._isa) if self._isa else 0

    def circuit_depth(self) -> int:
        return max(c.depth() for c in self._isa) if self._isa else 0

    # -- estimation -------------------------------------------------------

    def shot_plan(self, total_shots: int) -> np.ndarray:
        weights = ALLOCATION_STRATEGIES[self.allocation](self.groups, self.variance_model)
        return allocate_shots(total_shots, np.asarray(weights, dtype=float), self.min_shots_per_group)

    def estimate(self, params: Sequence[float], total_shots: int) -> EstimationResult:
        """Sample the energy with ``total_shots`` spread over the groups."""
        with self.ledger.time_classical():
            plan = self.shot_plan(total_shots)
            bound = [
                circuit.assign_parameters(np.asarray(params, dtype=float))
                for circuit in self._isa
            ]

        total = self.identity_coeff
        variance_sum = 0.0
        used_shots = 0
        used_circuits = 0

        for gi, counts, n_shots in self._sample_groups(bound, plan):
            mean, var = self._group_statistics(gi, counts)
            total += mean
            variance_sum += var / max(1, n_shots)
            used_shots += n_shots
            used_circuits += 1
            self.ledger.record_execution(
                shots=n_shots,
                duration_s=self._durations[gi],
                n_circuits=1,
                two_qubit_gates=_two_qubit_count(self._isa[gi]),
                depth=self._isa[gi].depth(),
            )

        self.ledger.bump("estimate_calls")
        return EstimationResult(
            value=float(total),
            stderr=float(np.sqrt(max(variance_sum, 0.0))),
            shots_used=used_shots,
            circuits_used=used_circuits,
            per_group_shots=plan,
        )

    def _sample_groups(self, bound, plan):
        """Yield ``(group_index, counts, shots)`` for every group with shots.

        Aer's cost is dominated by per-``run``-call overhead, not by shots: on
        this machine a 6-qubit noisy circuit costs ~0.45 s at both 1k and 4k
        shots.  Adaptive allocation hands every group a *different* shot count,
        which under naive batching means one call per group and a ~10x
        simulation slowdown for exactly the configuration we most want to
        study.

        So when the plan is spread out, we run every circuit once in a single
        batch at ``max(plan)`` shots and then draw each group's allotment from
        the returned counts without replacement.  A subset of iid multinomial
        draws is distributed exactly as that many iid draws, so this is
        statistically identical to having asked for the smaller number -- it
        only spends surplus *simulator* effort, and the ledger continues to
        charge the planned shots, which is what hardware would cost.
        """
        active = [(gi, int(n)) for gi, n in enumerate(plan) if n > 0]
        if not active:
            return
        distinct = {n for _, n in active}

        if len(distinct) <= 2 or len(active) <= 2:
            for n_shots in sorted(distinct):
                idx = [gi for gi, n in active if n == n_shots]
                with self.ledger.time_simulator():
                    counts_list = self.env.run([bound[i] for i in idx], shots=n_shots)
                if not isinstance(counts_list, list):
                    counts_list = [counts_list]
                for gi, counts in zip(idx, counts_list):
                    yield gi, counts, n_shots
            return

        top = max(n for _, n in active)
        idx = [gi for gi, _ in active]
        with self.ledger.time_simulator():
            counts_list = self.env.run([bound[i] for i in idx], shots=top)
        if not isinstance(counts_list, list):
            counts_list = [counts_list]
        for (gi, n_shots), counts in zip(active, counts_list):
            yield gi, (counts if n_shots == top else _subsample(counts, n_shots, self._rng)), n_shots

    def _group_statistics(self, group_index: int, counts: dict) -> tuple[float, float]:
        """Weighted mean and variance of this group's per-shot value.

        The empirical variance already contains every intra-group covariance,
        which is exactly what the shot allocation needs and what a
        coefficient-based proxy gets wrong.
        """
        group = self.groups[group_index]
        with self.ledger.time_classical():
            bits, weights = _counts_to_arrays(counts, self.ansatz.num_qubits)
            values = group.shot_values(bits)
            n = float(weights.sum())
            mean = float(values @ weights / n)
            if n > 1:
                var = float(((values - mean) ** 2) @ weights / (n - 1))
            else:
                var = float(group.weight() ** 2)
            self.variance_model.observe(group_index, var, n)
        return mean, var

    # -- reference values -------------------------------------------------

    def exact(self, params: Sequence[float]) -> float:
        """Noiseless, infinite-shot energy -- the target the sampler estimates."""
        bound = self.ansatz.assign_parameters(np.asarray(params, dtype=float))
        return float(Statevector(bound).expectation_value(self.hamiltonian).real)




def _subsample(counts: dict, n_shots: int, rng) -> dict:
    """Draw ``n_shots`` shots without replacement from a realised counts dict.

    A subset of iid multinomial draws is distributed exactly as that many iid
    draws, so this is statistically identical to having run the circuit for
    ``n_shots`` in the first place.
    """
    keys = list(counts.keys())
    totals = np.fromiter((counts[k] for k in keys), dtype=np.int64, count=len(keys))
    available = int(totals.sum())
    if n_shots >= available:
        return counts
    drawn = rng.multivariate_hypergeometric(totals, n_shots)
    return {k: int(v) for k, v in zip(keys, drawn) if v > 0}


def _counts_to_arrays(counts: dict, num_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """(unique bitstrings as a bool matrix, their multiplicities).

    Qiskit prints counts keys with qubit 0 rightmost; column q of the returned
    matrix is qubit q.
    """
    keys = list(counts.keys())
    weights = np.fromiter((counts[k] for k in keys), dtype=float, count=len(keys))
    cleaned = [k.replace(" ", "") for k in keys]
    bits = np.zeros((len(keys), num_qubits), dtype=np.int8)
    for r, key in enumerate(cleaned):
        rev = key[::-1]
        for q in range(min(num_qubits, len(rev))):
            if rev[q] == "1":
                bits[r, q] = 1
    return bits, weights


def _two_qubit_count(circuit: QuantumCircuit) -> int:
    return sum(1 for inst in circuit.data if len(inst.qubits) == 2 and inst.operation.name != "barrier")
