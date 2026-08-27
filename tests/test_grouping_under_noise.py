"""The domain boundary of every variance-based result in this project.

RESEARCH_LOG Result 55: the measurement machinery optimises estimator
*variance*, and on a device noise model at realistic depths the error is
*bias* instead -- so the ranking those improvements are built on inverts.  Two
facts hold that conclusion up, and both are cheap enough to keep as tests:

* general-commuting grouping needs entangling gates to reach its measurement
  basis while qubit-wise commuting grouping needs none, which is the mechanism
* on an ideal simulator more shots buy accuracy at the usual `1/sqrt(n)`, and
  under device noise they buy almost nothing

If either stops holding, the domain caveat in the README and in Results 38, 54
and 55 has to be re-derived rather than quietly inherited.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qres.ansatz import hartree_fock_state
from qres.estimator import ShotEstimator
from qres.noise import make_environment
from qres.problems.chemistry import build_molecule

TWO_QUBIT_GATES = {"cx", "cz", "ecr", "cy", "swap", "rzz"}


def two_qubit_count(circuit) -> int:
    return sum(n for name, n in circuit.count_ops().items() if name in TWO_QUBIT_GATES)


def estimator_for(problem, scheme: str, environment: str, seed: int = 0):
    ansatz = hartree_fock_state(problem.hf_bitstring)
    return ShotEstimator(
        problem.hamiltonian,
        ansatz,
        make_environment(environment, seed=seed),
        grouping=scheme,
        allocation="uniform",
    )


def exact_energy(problem) -> float:
    state = Statevector(hartree_fock_state(problem.hf_bitstring))
    return float(np.real(state.expectation_value(problem.hamiltonian)))


def test_qwc_basis_changes_need_no_entangling_gates():
    """The mechanism behind Result 55's inversion.

    Qubit-wise commuting terms share a single-qubit measurement basis, so the
    basis change is a layer of rotations.  If this ever stops being true the
    inversion has a different cause than the one recorded.
    """
    problem = build_molecule("LiH")
    estimator = estimator_for(problem, "qwc", "ideal")
    assert sum(two_qubit_count(c) for c in estimator._isa) == 0


def test_general_commuting_basis_changes_do_need_entangling_gates():
    """The other half: general commuting groups pay a Clifford diagonalisation."""
    problem = build_molecule("LiH")
    estimator = estimator_for(problem, "commuting", "ideal")
    assert max(two_qubit_count(c) for c in estimator._isa) > 0


def test_general_commuting_uses_far_fewer_groups():
    """The advantage it buys with those gates, so the trade is visible here."""
    problem = build_molecule("LiH")
    commuting = estimator_for(problem, "commuting", "ideal")
    qwc = estimator_for(problem, "qwc", "ideal")
    assert len(commuting.groups) < len(qwc.groups) / 3


@pytest.mark.parametrize("scheme", ["commuting", "qwc"])
def test_ideal_error_falls_with_shots(scheme):
    """Shot-noise-limited: 16x the shots must buy a clear improvement."""
    problem = build_molecule("H4")
    exact = exact_energy(problem)

    def median_error(shots: int) -> float:
        errors = []
        for seed in range(8):
            estimator = estimator_for(problem, scheme, "ideal", seed=3000 + seed)
            errors.append(abs(estimator.estimate([], shots).value - exact))
        return float(np.median(errors))

    assert median_error(400_000) < 0.6 * median_error(25_000)


def test_noisy_error_is_bias_and_shots_barely_help():
    """The finding that bounds every variance-based result here.

    Under a device noise model the median error tracks the bias, so more shots
    do almost nothing -- measured at 4.3% for 16x the shots.  Asserted loosely
    (no better than 30%) so ordinary run-to-run variation does not fail it,
    while a return to `1/sqrt(n)` behaviour still would.
    """
    problem = build_molecule("H4")
    exact = exact_energy(problem)

    def measure(shots: int):
        values = [
            estimator_for(problem, "commuting", "heron", seed=4000 + seed)
            .estimate([], shots)
            .value
            for seed in range(8)
        ]
        values = np.array(values)
        return float(np.median(np.abs(values - exact))), float(np.mean(values) - exact)

    small_error, small_bias = measure(25_000)
    large_error, _ = measure(400_000)

    # the error is the bias, not the statistics
    assert abs(small_bias) == pytest.approx(small_error, rel=0.2)
    # and 16x the shots does not rescue it
    assert large_error > 0.7 * small_error


def test_identity_padding_leaves_the_state_exactly_unchanged():
    """Result 60's gate-count knob must move gates and nothing else.

    `CX CX = I`, so the padded circuit prepares exactly the Hartree-Fock state
    at any padding depth -- which is what lets one exact reference serve the
    whole sweep.  The barriers are load-bearing: without them the transpiler
    cancels the pairs and the sweep measures nothing.
    """
    from experiments.exp014_gate_budget import (
        padded_reference,
        two_qubit_count as transpiled_two_qubit_count,
    )

    problem = build_molecule("H4")
    reference = Statevector(hartree_fock_state(problem.hf_bitstring))

    previous = -1
    for pairs in (0, 2, 8, 16):
        circuit = padded_reference(problem, pairs)
        assert abs(reference.inner(Statevector(circuit))) == pytest.approx(1.0, abs=1e-12)

        estimator = ShotEstimator(
            problem.hamiltonian,
            circuit,
            make_environment("heron", seed=0),
            grouping="qwc",
            allocation="uniform",
        )
        gates = transpiled_two_qubit_count(estimator)
        assert gates == 2 * pairs, "transpiler cancelled the padding"
        assert gates > previous
        previous = gates


def test_per_gate_cost_tracks_the_device_error_rate():
    """Result 65's relationship, on two devices a factor of 6 apart.

    The cost of a two-qubit gate *is* the device's own two-qubit error rate
    (ratio 0.64-1.09 across seven devices), which is what turns the wall into a
    hardware number rather than an algorithmic one.  Checked loosely here -- the
    property that must hold is the ordering and rough scale, not the fit.
    """
    from experiments.exp014_gate_budget import padded_reference
    from experiments.exp015_device_generations import device_error_rates
    from qres.mitigation import readout_mitigated_energy
    from qres.noise import device_environment

    problem = build_molecule("H4")
    exact = exact_energy(problem)
    circuit = padded_reference(problem, 16)  # 32 two-qubit gates

    measured = {}
    for device in ("fake_boston", "fake_brisbane"):
        values = [
            readout_mitigated_energy(
                problem.hamiltonian, circuit,
                device_environment(device, seed=18000 + seed),
                params=[], total_shots=60_000, calibration_fraction=0.05,
                calibration="tensored", grouping="qwc", allocation="uniform",
            ).value
            for seed in range(4)
        ]
        measured[device] = float(np.median(np.abs(np.array(values) - exact)))

    better, worse = device_error_rates("fake_boston")[0], device_error_rates("fake_brisbane")[0]
    assert better < worse, "fake_boston should be the better-calibrated device"
    assert measured["fake_boston"] < measured["fake_brisbane"], (
        "the better device must give the smaller error at the same gate count"
    )


def test_code_distance_grows_only_logarithmically_in_the_target():
    """Result 69's load-bearing property, and the reason it reverses the picture.

    Eight orders of magnitude in required logical error must cost only a small
    factor in code distance -- that is what turns a 4.5e8 fidelity gap into a
    4x qubit factor.  If this ever stopped holding, the conclusion that
    error-corrected chemistry is an engineering target rather than an
    impossibility would have to be re-derived.
    """
    from experiments.exp016_error_correction_overhead import (
        logical_error,
        required_distance,
    )

    physical = 0.00127
    loose = required_distance(1e-4, physical)
    tight = required_distance(1e-12, physical)

    assert loose is not None and tight is not None
    # eight orders of magnitude in target error buys at most a 4x distance
    # (measured: 7 -> 25) and hence ~13x the qubits, against 1e8 in fidelity
    assert tight < 4 * loose, f"distance {loose} -> {tight} for 1e-4 -> 1e-12"
    assert (tight / loose) ** 2 < 20, "qubit overhead must stay polynomial"

    # and the model must actually be decreasing in distance, or the search above
    # would return the first value it tried for any target
    assert logical_error(21, physical) < logical_error(11, physical)
    assert logical_error(11, physical) < logical_error(5, physical)


def test_required_distance_reports_failure_rather_than_guessing():
    """A physical error above threshold has no valid distance; say so."""
    from experiments.exp016_error_correction_overhead import required_distance

    assert required_distance(1e-9, 0.02) is None, "above threshold must not converge"
    assert required_distance(1e-9, 0.001) is not None


def test_distillation_overhead_shrinks_with_problem_size():
    """Result 70's counterintuitive property, and why the Result 69 caveat was wrong.

    The factory is a fixed footprint while the data register grows, so magic-state
    overhead falls from a large multiple on tiny problems to ~25% on a drug-sized
    one.  If it ever grew instead, the ~1.3e5 estimate would need re-deriving.
    """
    from experiments.exp016_error_correction_overhead import (
        FACTORY_LOGICAL_QUBITS,
        distillation_rounds,
        t_count,
    )

    physical = 0.00127

    # a small problem and a large one, at their respective code distances
    small = (6, 11, 152)     # H4: logical qubits, distance, rotations
    large = (98, 23, 1_665_043)  # 50 orbitals

    fractions = []
    for logical_qubits, distance, rotations in (small, large):
        gates, _ = t_count(rotations, 1.6e-3)
        rounds = distillation_rounds(physical, 1.6e-3 / gates)
        assert rounds is not None and rounds <= 3, f"expected few rounds, got {rounds}"

        data = logical_qubits * 2 * distance**2
        factory = FACTORY_LOGICAL_QUBITS * 2 * distance**2 * rounds
        fractions.append(factory / (data + factory))

    assert fractions[1] < fractions[0], "overhead should shrink with size"
    assert fractions[1] < 0.4, f"drug-sized factory share {fractions[1]:.2f}"


def test_t_count_is_far_below_the_two_qubit_gate_count():
    """Most of a UCCSD circuit is Clifford, which is why distillation is cheap.

    NH3 has 32,318 two-qubit gates but only 2,340 non-Clifford rotations.  If a
    future ansatz inverted that, the distillation term would dominate instead.
    """
    from experiments.exp016_error_correction_overhead import MOLECULES

    for name, _orbitals, _qubits, gates, rotations in MOLECULES[1:]:
        assert rotations < gates / 5, f"{name}: {rotations} rotations vs {gates} gates"
