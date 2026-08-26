"""Error mitigation, charged for what it costs.

RESEARCH_LOG Result 55 is why this exists.  Everything else in this package
optimises estimator *variance*, and on a Heron-class noise model that is the
wrong target: 16x the shots buys 4.3% because the error equals the device bias
to three digits.  Mitigation is the only tool here that attacks bias.

It is also usually reported dishonestly.  Both methods below *buy* accuracy with
*extra circuit executions*, so the only meaningful comparison holds the total
shot count fixed -- mitigated and unmitigated must be charged the same budget,
with the mitigation's overhead coming out of its own allowance rather than being
added on top.  Every entry point here takes a total budget and divides it.

**Zero-noise extrapolation** runs the circuit at deliberately amplified noise
and extrapolates back to zero.  Amplification is by *global unitary folding*:
``C -> C (C^dag C)^k`` leaves the ideal result untouched and multiplies the gate
count by ``2k+1``.  Folding only the state-preparation circuit is deliberate --
the measurement basis changes are part of the estimator's own machinery, and
folding them would amplify a different quantity than the state's error.

**Readout mitigation** measures the assignment matrix from calibration circuits
and inverts it.  The inverse of a stochastic matrix is not stochastic, so raw
inversion produces negative probabilities and can *increase* the error; the
correction here projects back onto the simplex, which is what M3 and its
relatives do more cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit


@dataclass
class MitigationResult:
    value: float
    #: what the same budget produced with no mitigation, when measured alongside
    unmitigated: float | None = None
    shots_used: int = 0
    circuits_used: int = 0
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# zero-noise extrapolation
# --------------------------------------------------------------------------

def fold_global(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Amplify noise by ``scale`` (an odd integer) without changing the ideal state.

    ``C -> C (C^dag C)^k`` for ``scale = 2k + 1``.  ``C^dag C`` is the identity
    exactly, so a noiseless device returns the same state at every scale factor
    while a real one degrades predictably -- which is what makes the
    extrapolation back to zero meaningful.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"scale must be a positive odd integer, got {scale}")
    if scale == 1:
        return circuit.copy()

    folded = circuit.copy()
    inverse = circuit.inverse()
    for _ in range((scale - 1) // 2):
        folded.compose(inverse, inplace=True)
        folded.compose(circuit, inplace=True)
    return folded


def extrapolate(scales: Sequence[float], values: Sequence[float], method: str = "linear") -> float:
    """Extrapolate measured values back to zero noise.

    ``linear`` is a least-squares fit in the scale factor; ``richardson`` is the
    exact polynomial through every point, evaluated at zero.  Richardson uses
    all the information and is far more sensitive to noise in the points, which
    is a real trade rather than a detail -- with three noisy points it routinely
    does worse than the line.
    """
    scales = np.asarray(scales, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(scales) < 2:
        raise ValueError("extrapolation needs at least two scale factors")

    if method == "linear":
        slope, intercept = np.polyfit(scales, values, 1)
        return float(intercept)
    if method == "richardson":
        # Lagrange interpolation evaluated at 0
        total = 0.0
        for i, (x_i, y_i) in enumerate(zip(scales, values)):
            weight = 1.0
            for j, x_j in enumerate(scales):
                if i != j:
                    weight *= (0.0 - x_j) / (x_i - x_j)
            total += weight * y_i
        return float(total)
    raise ValueError(f"unknown extrapolation {method!r}")


def zne_energy(
    hamiltonian,
    ansatz: QuantumCircuit,
    environment,
    params: Sequence[float],
    total_shots: int,
    scales: Sequence[int] = (1, 3, 5),
    method: str = "linear",
    grouping: str = "qwc",
    **estimator_kwargs,
) -> MitigationResult:
    """ZNE at a fixed total budget, split evenly across the scale factors.

    ``grouping`` defaults to qubit-wise commuting rather than the package
    default: Result 55 measured that general-commuting groups lose on a device
    because their Clifford basis changes cost 125 two-qubit gates against QWC's
    zero, and folding makes deep circuits deeper still.
    """
    from .estimator import ShotEstimator

    per_scale = max(1, total_shots // len(scales))
    values, shots_used, circuits_used = [], 0, 0

    for scale in scales:
        folded = fold_global(ansatz, scale)
        estimator = ShotEstimator(
            hamiltonian, folded, environment, grouping=grouping, **estimator_kwargs
        )
        result = estimator.estimate(params, per_scale)
        values.append(result.value)
        shots_used += result.shots_used
        circuits_used += result.circuits_used

    return MitigationResult(
        value=extrapolate(scales, values, method),
        shots_used=shots_used,
        circuits_used=circuits_used,
        method=f"zne/{method}",
        metadata={"scales": list(scales), "values": values, "per_scale_shots": per_scale},
    )


# --------------------------------------------------------------------------
# readout mitigation
# --------------------------------------------------------------------------

def assignment_matrix(environment, num_qubits: int, shots: int = 4096) -> np.ndarray:
    """Measure P(read j | prepared i) for every computational basis state.

    Exact and expensive: ``2^n`` calibration circuits.  That is affordable at
    the sizes here and hopeless past ~14 qubits, which is exactly why the
    tensored and M3 approximations exist.  Measuring it exactly first means the
    approximations can be scored against something real.
    """
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension))

    circuits = []
    for prepared in range(dimension):
        circuit = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            if (prepared >> qubit) & 1:
                circuit.x(qubit)
        circuit.measure_all()
        # optimisation level 0 keeps the trivial layout.  Anything higher may
        # permute qubits, which silently transposes the assignment matrix into
        # a plausible-looking wrong answer.
        circuits.append(environment.prepare(circuit, optimization_level=0))

    # one batched call, not 2^n separate ones: at 6 qubits that was 228 s of
    # per-run simulator setup for 64 trivial circuits
    results = environment.run(circuits, shots)
    if not isinstance(results, list):
        results = [results]

    for prepared, counts in enumerate(results):
        for bitstring, count in counts.items():
            matrix[int(bitstring.replace(" ", ""), 2), prepared] += count
        column = matrix[:, prepared].sum()
        if column:
            matrix[:, prepared] /= column
    return matrix


def measured_physical_qubits(estimator) -> list[int]:
    """Which physical qubits the estimator's circuits actually measure.

    The transpiler does not use qubits 0..n-1.  At optimisation level 2 on a
    133-qubit Heron target it picked [66, 5, 87, 81, 60, 51] for a six-qubit
    problem -- hand-choosing qubits with better readout, which is worth 8x of
    bias on its own.  Calibrating on 0..n-1 and correcting counts from those
    qubits applies the wrong numbers with full confidence.
    """
    layout = estimator._isa[0].layout
    if layout is None:
        return list(range(estimator.ansatz.num_qubits))
    return list(layout.final_index_layout())


def assignment_matrix_on(
    environment, physical_qubits: Sequence[int], shots: int = 4096
) -> np.ndarray:
    """Assignment matrix measured on specific physical qubits.

    Same construction as :func:`assignment_matrix`, but the calibration circuits
    are built directly on the device register so no transpiler pass can move
    them somewhere else.
    """
    physical = list(physical_qubits)
    num_qubits = len(physical)
    dimension = 1 << num_qubits
    width = (
        environment.backend.num_qubits
        if environment.backend is not None
        else max(physical) + 1
    )

    circuits = []
    for prepared in range(dimension):
        circuit = QuantumCircuit(width, num_qubits)
        for index, qubit in enumerate(physical):
            if (prepared >> index) & 1:
                circuit.x(qubit)
        circuit.measure(physical, range(num_qubits))
        circuits.append(circuit)

    results = environment.run(circuits, shots)
    if not isinstance(results, list):
        results = [results]

    matrix = np.zeros((dimension, dimension))
    for prepared, counts in enumerate(results):
        for bitstring, count in counts.items():
            matrix[int(bitstring.replace(" ", ""), 2), prepared] += count
        column = matrix[:, prepared].sum()
        if column:
            matrix[:, prepared] /= column
    return matrix


def tensored_assignment_matrix(
    environment, physical_qubits: Sequence[int], shots: int = 4096
) -> np.ndarray:
    """Assignment matrix assuming readout errors are independent per qubit.

    Two calibration circuits -- all zeros and all ones -- give every qubit's
    2x2 response, and the full matrix is their tensor product.  That replaces
    ``2^n`` circuits with 2, which is the difference between calibration being
    a rounding error and being the entire budget: at 10 qubits it is 1 024
    circuits against 2.

    The assumption is that readout errors do not correlate across qubits.  It is
    not exactly true on real devices (crosstalk, shared readout lines), so this
    trades a modelling error for a statistical one -- and at a fixed budget the
    statistical saving is enormous, which is what makes the trade worth
    measuring rather than assuming.
    """
    physical = list(physical_qubits)
    num_qubits = len(physical)
    width = (
        environment.backend.num_qubits
        if environment.backend is not None
        else max(physical) + 1
    )

    circuits = []
    for prepared_bit in (0, 1):
        circuit = QuantumCircuit(width, num_qubits)
        if prepared_bit:
            for qubit in physical:
                circuit.x(qubit)
        circuit.measure(physical, range(num_qubits))
        circuits.append(circuit)

    results = environment.run(circuits, shots)
    if not isinstance(results, list):
        results = [results]

    # per qubit: P(read 1 | prepared 0) and P(read 0 | prepared 1)
    singles = []
    wrong = np.zeros((2, num_qubits))
    totals = np.zeros(2)
    for prepared_bit, counts in enumerate(results):
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")[::-1]  # index i is qubit i
            for index in range(num_qubits):
                if int(bits[index]) != prepared_bit:
                    wrong[prepared_bit, index] += count
        totals[prepared_bit] = sum(counts.values())

    for index in range(num_qubits):
        p01 = wrong[0, index] / totals[0] if totals[0] else 0.0  # read 1, prepared 0
        p10 = wrong[1, index] / totals[1] if totals[1] else 0.0  # read 0, prepared 1
        singles.append(np.array([[1 - p01, p10], [p01, 1 - p10]]))

    matrix = np.array([[1.0]])
    # qubit 0 is the least significant bit of the basis index, so it must be the
    # last factor in the Kronecker product
    for single in reversed(singles):
        matrix = np.kron(matrix, single)
    return matrix


def correct_counts(counts: dict, matrix: np.ndarray, num_qubits: int) -> dict:
    """Undo readout error, then project back onto the probability simplex.

    Inverting a stochastic matrix does not give a stochastic matrix, so the raw
    solution carries negative entries; left alone they make expectation values
    *worse* than no correction at all.  The projection is the cheap fix and the
    reason this is worth having as a function rather than a matrix solve.
    """
    dimension = 1 << num_qubits
    observed = np.zeros(dimension)
    for bitstring, count in counts.items():
        observed[int(bitstring.replace(" ", ""), 2)] += count
    total = observed.sum()
    if total == 0:
        return dict(counts)
    observed /= total

    corrected = np.linalg.lstsq(matrix, observed, rcond=None)[0]
    corrected = _project_to_simplex(corrected)

    return {
        format(index, f"0{num_qubits}b"): float(value * total)
        for index, value in enumerate(corrected)
        if value > 0
    }


def _project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Nearest point with non-negative entries summing to one (Euclidean)."""
    n = len(vector)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, n + 1)
    feasible = ordered - (cumulative - 1) / indices > 0
    if not feasible.any():
        return np.full(n, 1.0 / n)
    rho = indices[feasible][-1]
    theta = (cumulative[feasible][-1] - 1) / rho
    return np.maximum(vector - theta, 0.0)


def attach_readout_correction(estimator, matrix: np.ndarray) -> None:
    """Route every counts dict the estimator sees through the correction.

    **The layout has to match.**  ``matrix`` is measured on physical qubits
    ``0..n-1``; readout error differs per qubit, so applying it to counts from
    circuits the transpiler laid out elsewhere corrects with the wrong numbers
    and produces a confidently wrong answer.  Build both the estimator and the
    calibration at ``optimization_level=0``, which keeps the trivial layout on
    each, or the correction is meaningless.
    """
    num_qubits = estimator.ansatz.num_qubits
    original = estimator._group_statistics

    def wrapped(group_index, counts):
        return original(group_index, correct_counts(counts, matrix, num_qubits))

    estimator._group_statistics = wrapped


def readout_mitigated_energy(
    hamiltonian,
    ansatz: QuantumCircuit,
    environment,
    params: Sequence[float],
    total_shots: int,
    calibration_fraction: float = 0.25,
    grouping: str = "qwc",
    calibration: str = "exact",
    **estimator_kwargs,
) -> MitigationResult:
    """Readout-corrected energy, with calibration paid out of the same budget.

    Calibration is ``2^n`` circuits, so it is not a rounding error: at 10 qubits
    it is 1 024 circuits before a single measurement of the observable.  Taking
    it out of ``total_shots`` rather than adding it on top is the whole point --
    otherwise the comparison flatters mitigation by however much it spends.
    """
    from .estimator import ShotEstimator

    num_qubits = ansatz.num_qubits
    calibration_budget = int(total_shots * calibration_fraction)

    # build the estimator first, then calibrate the qubits it actually uses
    estimator = ShotEstimator(
        hamiltonian, ansatz, environment, grouping=grouping, **estimator_kwargs
    )
    physical = measured_physical_qubits(estimator)

    # the same budget buys 2^(n-1) times more shots per calibration point when
    # only two circuits are needed instead of 2^n
    if calibration == "tensored":
        circuits_needed = 2
        per_state = max(1, calibration_budget // circuits_needed)
        matrix = tensored_assignment_matrix(environment, physical, shots=per_state)
    elif calibration == "exact":
        circuits_needed = 1 << num_qubits
        per_state = max(1, calibration_budget // circuits_needed)
        matrix = assignment_matrix_on(environment, physical, shots=per_state)
    else:
        raise ValueError(f"unknown calibration {calibration!r}")

    attach_readout_correction(estimator, matrix)
    result = estimator.estimate(params, total_shots - calibration_budget)

    return MitigationResult(
        value=result.value,
        shots_used=result.shots_used + per_state * circuits_needed,
        circuits_used=result.circuits_used + circuits_needed,
        method=f"readout/{calibration}",
        metadata={
            "calibration_shots": per_state * circuits_needed,
            "calibration_circuits": circuits_needed,
            "shots_per_calibration_point": per_state,
            "shots_on_observable": result.shots_used,
            "physical_qubits": physical,
        },
    )
