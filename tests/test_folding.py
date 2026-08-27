"""Correctness of the HP-lattice folding model and its classical solvers.

The energy function is where this can go quietly wrong.  A folding benchmark
whose energy accepts self-intersecting conformations measures nothing, and the
error is invisible: the numbers stay plausible and only get *better*, because
overlapping residues create spurious contacts.  So the checks below pin the
energy to hand-computed conformations and verify that invalid walks are refused.

The move set is the other risk.  Started from the fully extended chain there are
no corners and no U-turns, so two of the three moves return ``None`` on every
residue and the search cannot fold at all -- measured, it found *zero* H-H
contacts on two of four benchmark sequences before random starting
conformations were added.
"""

from __future__ import annotations

import numpy as np
import pytest

from qres.problems.folding import (
    BENCHMARKS,
    MOVES_2D,
    _corner_flip,
    _straight_walk,
    annealed_fold,
    energy_of,
    exact_fold,
    positions_from_moves,
    random_walk,
)


def test_energy_of_a_hand_computed_fold():
    """A square of four H residues: one non-consecutive contact, so energy -1."""
    positions = ((0, 0), (1, 0), (1, 1), (0, 1))
    assert energy_of("HHHH", positions) == -1.0

    # the same square with polar corners has no H-H contact at all
    assert energy_of("HPPH", positions) == -1.0  # residues 0 and 3 are adjacent
    assert energy_of("HPPP", positions) == 0.0


def test_consecutive_residues_never_count_as_contacts():
    """Sequence neighbours are always lattice neighbours; they are not contacts."""
    straight = tuple(_straight_walk(4))
    assert energy_of("HHHH", straight) == 0.0


def test_self_intersecting_walks_are_refused():
    """An overlapping conformation must never score better than a valid one.

    Without this the search finds "folds" that collapse residues onto each other
    and reports impossibly low energies.
    """
    overlapping = ((0, 0), (1, 0), (0, 0), (1, 0))
    assert energy_of("HHHH", overlapping) == float("inf")


def test_positions_from_moves_walks_the_lattice():
    positions = positions_from_moves([0, 2, 1])
    assert positions == ((0, 0), (1, 0), (1, 1), (0, 1))
    for before, after in zip(positions, positions[1:]):
        assert abs(before[0] - after[0]) + abs(before[1] - after[1]) == 1


@pytest.mark.parametrize("length", [6, 10, 16])
def test_random_walk_is_self_avoiding_and_connected(length):
    rng = np.random.default_rng(0)
    for _ in range(20):
        walk = random_walk(length, rng)
        assert len(walk) == length
        assert len(set(walk)) == length, "self-intersecting start"
        for before, after in zip(walk, walk[1:]):
            assert abs(before[0] - after[0]) + abs(before[1] - after[1]) == 1


def test_corner_flip_returns_none_on_a_straight_chain():
    """The bug that made the annealer useless: nothing to flip on a line."""
    straight = _straight_walk(6)
    assert all(_corner_flip(straight, i) is None for i in range(1, 5))


def test_corner_flip_preserves_self_avoidance_and_connectivity():
    rng = np.random.default_rng(1)
    for _ in range(50):
        walk = random_walk(10, rng)
        for index in range(1, len(walk) - 1):
            moved = _corner_flip(walk, index)
            if moved is None:
                continue
            assert len(set(moved)) == len(moved)
            for before, after in zip(moved, moved[1:]):
                assert abs(before[0] - after[0]) + abs(before[1] - after[1]) == 1


def test_exact_enumeration_matches_a_brute_force_over_all_moves():
    """Two independent routes to the same optimum on a small sequence."""
    sequence = "HPHPHH"
    best = exact_fold(sequence)

    naive = float("inf")
    for encoded in range(len(MOVES_2D) ** (len(sequence) - 1)):
        moves = []
        value = encoded
        for _ in range(len(sequence) - 1):
            moves.append(value % len(MOVES_2D))
            value //= len(MOVES_2D)
        naive = min(naive, energy_of(sequence, positions_from_moves(moves)))

    assert best.energy == naive


def test_annealer_finds_a_valid_fold_on_every_benchmark():
    """It need not be optimal -- but `inf` means it never found a legal fold."""
    for sequence, _ in BENCHMARKS.values():
        fold = annealed_fold(sequence, sweeps=400, restarts=2, seed=0)
        assert fold.is_valid
        assert np.isfinite(fold.energy)


def test_annealer_beats_a_random_valid_conformation():
    """Weak, but it is the property that makes the search a search."""
    sequence, _ = BENCHMARKS["hp20"]
    rng = np.random.default_rng(0)
    random_energies = [
        energy_of(sequence, tuple(random_walk(len(sequence), rng))) for _ in range(50)
    ]
    best_random = min(e for e in random_energies if np.isfinite(e))

    fold = annealed_fold(sequence, sweeps=1500, restarts=4, seed=0)
    assert fold.energy < best_random


def test_turn_encoding_reproduces_every_energy_exactly():
    """The encoding is only worth measuring if it is the same function.

    A hand-derived self-avoidance penalty is where published encodings differ
    from each other and where an error is invisible -- the Hamiltonian still
    looks like a Hamiltonian.  Checked here against the enumerated energy at
    *every* basis state, not a sample.
    """
    from qres.problems.folding import turn_encoding_hamiltonian

    sequence = "HPHPHH"
    hamiltonian, qubits = turn_encoding_hamiltonian(sequence)
    penalty = float(len(sequence))

    for state in range(1 << qubits):
        moves = [0] + [(state >> (2 * bond)) & 3 for bond in range(len(sequence) - 2)]
        expected = energy_of(sequence, positions_from_moves(moves))
        if not np.isfinite(expected):
            expected = penalty

        value = 0.0
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            sign = 1
            for qubit in range(qubits):
                if pauli.z[qubit] and (state >> qubit) & 1:
                    sign = -sign
            value += sign * coeff.real

        assert value == pytest.approx(expected, abs=1e-9), f"state {state}"


def test_turn_encoding_is_dense_and_high_weight():
    """The measured obstruction: 2^n terms at full weight, not a sparse Ising.

    This is what puts folding outside the gate budget (Result 63), so if the
    encoding ever becomes sparse the conclusion has to be re-derived.
    """
    from qres.problems.folding import encoding_cost

    cost = encoding_cost("HPHPHH")
    assert cost["qubits"] == 8
    assert cost["max_weight"] == cost["qubits"], "expected full-weight terms"
    assert cost["terms"] > 0.9 * 2 ** cost["qubits"], "expected a dense Hamiltonian"
    assert cost["two_qubit_gates_per_layer"] > 1000


def test_turn_encoding_refuses_sizes_it_cannot_enumerate():
    from qres.problems.folding import turn_encoding_hamiltonian

    with pytest.raises(ValueError, match="too many to enumerate"):
        turn_encoding_hamiltonian("H" * 15)
