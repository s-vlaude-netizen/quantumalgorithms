"""HP-lattice protein folding, and the classical baseline it has to beat.

Result 61 narrowed what is worth looking for to a single shape: a problem whose
*required relative accuracy* is loose enough to fit a hundred-gate budget, and
whose *classical baseline* is genuinely hard.  MaxCut has the first and fails the
second (Result 51: a fifty-line hill-climber beats QAOA in a millisecond).
Chemistry has the second and fails the first by ~3 000x.

Protein folding is the one candidate the user named that might have both, and it
is the standard quantum-optimisation showcase.  The HP model is its simplest
form: each residue is hydrophobic (H) or polar (P), the chain occupies a
self-avoiding walk on a lattice, and the energy counts H-H pairs that are lattice
neighbours but not sequence neighbours.  Finding the minimum is NP-hard.

**This module deliberately builds the classical side first.**  The lesson from
Results 42 and 50 is that a quantum benchmark on instances a classical method
solves instantly measures nothing, and both times that was discovered *after*
building the quantum machinery.  So: an exact solver, a strong heuristic, and a
measurement of where the heuristic starts to fail.  Only a regime where it does
fail is worth encoding into qubits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

#: 2D square lattice moves, as (dx, dy)
MOVES_2D = ((1, 0), (-1, 0), (0, 1), (0, -1))

#: benchmark sequences from the HP-model literature, with their known optima
#: (Unger & Moult; Lau & Dill).  Used to check the solvers rather than trusted.
BENCHMARKS = {
    "hp20": ("HPHPPHHPHPPHPHHPPHPH", -9),
    "hp24": ("HHPPHPPHPPHPPHPPHPPHPPHH", -9),
    "hp25": ("PPHPPHHPPPPHHPPPPHHPPPPHH", -8),
    "hp36": ("PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", -14),
}


@dataclass
class Fold:
    """A conformation: the lattice positions of every residue."""

    sequence: str
    positions: tuple[tuple[int, int], ...]
    energy: float
    seconds: float = 0.0
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(set(self.positions)) == len(self.positions)


def energy_of(sequence: str, positions) -> float:
    """Minus the number of non-consecutive H-H lattice contacts.

    Returns ``+inf`` for a self-intersecting walk, so an invalid conformation can
    never look better than a valid one -- the failure mode that makes a folding
    benchmark silently meaningless.
    """
    if len(set(positions)) != len(positions):
        return float("inf")

    hydrophobic = [i for i, residue in enumerate(sequence) if residue == "H"]
    index = {position: i for i, position in enumerate(positions)}

    contacts = 0
    for i in hydrophobic:
        x, y = positions[i]
        for dx, dy in MOVES_2D:
            j = index.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -float(contacts)


def positions_from_moves(moves) -> tuple[tuple[int, int], ...]:
    """Walk the chain from the origin along ``moves`` (one per bond)."""
    x = y = 0
    positions = [(0, 0)]
    for move in moves:
        dx, dy = MOVES_2D[move]
        x, y = x + dx, y + dy
        positions.append((x, y))
    return tuple(positions)


def _walks(length: int) -> Iterator[tuple[int, ...]]:
    """Every self-avoiding walk of ``length`` bonds, first bond fixed.

    Fixing the first move removes the four-fold rotational symmetry, which is a
    4x saving and does not change the optimum.
    """
    stack: list[tuple[tuple[int, ...], set, int, int]] = [((0,), {(0, 0), (1, 0)}, 1, 0)]
    while stack:
        moves, occupied, x, y = stack.pop()
        if len(moves) == length:
            yield moves
            continue
        for index, (dx, dy) in enumerate(MOVES_2D):
            nx, ny = x + dx, y + dy
            if (nx, ny) in occupied:
                continue
            stack.append((moves + (index,), occupied | {(nx, ny)}, nx, ny))


def exact_fold(sequence: str) -> Fold:
    """Minimum energy by enumerating every self-avoiding walk.

    Self-avoiding walks grow as ~2.64^n on the square lattice, so this is the
    honest ceiling: usable to about 20 residues and hopeless past it.  Where it
    reaches, it is the only thing that can score a heuristic.
    """
    started = time.perf_counter()
    best_energy, best_positions, walks = float("inf"), None, 0

    for moves in _walks(len(sequence) - 1):
        walks += 1
        positions = positions_from_moves(moves)
        energy = energy_of(sequence, positions)
        if energy < best_energy:
            best_energy, best_positions = energy, positions

    return Fold(
        sequence=sequence,
        positions=best_positions or ((0, 0),),
        energy=best_energy,
        seconds=time.perf_counter() - started,
        method="exact (enumeration)",
        metadata={"walks_enumerated": walks},
    )


def _neighbours(position):
    x, y = position
    return [(x + dx, y + dy) for dx, dy in MOVES_2D]


def _corner_flip(positions: list, index: int) -> list | None:
    """Move residue ``index`` across the corner it sits on.

    Valid only where the chain turns: residues ``index-1`` and ``index+1`` must
    be diagonal to each other, and the opposite corner must be free.
    """
    before, here, after = positions[index - 1], positions[index], positions[index + 1]
    if before[0] == after[0] or before[1] == after[1]:
        return None  # straight segment, no corner to flip
    target = (before[0] + after[0] - here[0], before[1] + after[1] - here[1])
    if target in positions:
        return None
    moved = list(positions)
    moved[index] = target
    return moved


def _crankshaft(positions: list, index: int) -> list | None:
    """Rotate a U-shaped four-residue segment onto the other side."""
    if index + 3 >= len(positions):
        return None
    a, b, c, d = positions[index : index + 4]
    if a[0] != d[0] and a[1] != d[1]:
        return None
    # b and c must sit on the same side; reflect them through the a-d axis
    new_b = (a[0] + (a[0] - b[0]), a[1] + (a[1] - b[1]))
    new_c = (d[0] + (d[0] - c[0]), d[1] + (d[1] - c[1]))
    if abs(new_b[0] - new_c[0]) + abs(new_b[1] - new_c[1]) != 1:
        return None
    occupied = set(positions) - {b, c}
    if new_b in occupied or new_c in occupied:
        return None
    moved = list(positions)
    moved[index + 1], moved[index + 2] = new_b, new_c
    return moved


def _end_move(positions: list, at_start: bool) -> list | None:
    """Swing a terminal residue to any free neighbour of its partner."""
    index = 0 if at_start else len(positions) - 1
    anchor = positions[1] if at_start else positions[-2]
    occupied = set(positions) - {positions[index]}
    free = [n for n in _neighbours(anchor) if n not in occupied]
    if not free:
        return None
    moved = list(positions)
    moved[index] = free[0] if len(free) == 1 else free[hash(tuple(positions)) % len(free)]
    return moved


def _straight_walk(length: int) -> list:
    """The fully extended chain -- valid, and a trap to start from.

    It has no corners and no U-turns, so ``_corner_flip`` and ``_crankshaft``
    both return ``None`` on every residue and the search can only wiggle the two
    ends.  Measured: starting here the annealer found *zero* H-H contacts on two
    of four benchmark sequences.  Kept only as a fallback when random growth
    fails.
    """
    return [(i, 0) for i in range(length)]


def random_walk(length: int, rng, attempts: int = 200) -> list:
    """A random self-avoiding walk, grown with restarts when it traps itself.

    Growth is the standard way to sample a starting conformation: extend one
    step at a time into a free neighbour, and start over if the walk paints
    itself into a corner.  Unlike the extended chain this is full of corners,
    which is what the move set needs to do anything at all.
    """
    for _ in range(attempts):
        positions = [(0, 0)]
        occupied = {(0, 0)}
        for _ in range(length - 1):
            free = [n for n in _neighbours(positions[-1]) if n not in occupied]
            if not free:
                break
            choice = free[int(rng.integers(len(free)))]
            positions.append(choice)
            occupied.add(choice)
        if len(positions) == length:
            return positions
    return _straight_walk(length)


def annealed_fold(
    sequence: str,
    sweeps: int = 4000,
    restarts: int = 8,
    seed: int = 0,
    start_temperature: float = 2.0,
) -> Fold:
    """Simulated annealing over *valid* conformations only.

    This is the reference the quantum side would have to beat, so it has to be
    a real HP-model search rather than a generic one.  The distinction that
    matters: in MaxCut every bitstring is a feasible solution, while here almost
    every bond-direction vector self-intersects.  A search that perturbs
    directions freely spends all its time outside the feasible set -- measured,
    such a search failed to find *any* valid conformation for a 36-residue
    sequence in 30 000 iterations.

    So the moves are the standard lattice-protein set, each of which maps a
    self-avoiding walk to another self-avoiding walk: corner flips, crankshaft
    rotations, and end moves.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(sequence)
    best_energy, best_positions = float("inf"), None

    for restart in range(restarts):
        positions = random_walk(n, rng)
        energy = energy_of(sequence, positions)

        for sweep in range(sweeps):
            temperature = start_temperature * (1 - sweep / sweeps) + 1e-3
            for _ in range(n):
                kind = rng.random()
                if kind < 0.6:
                    index = int(rng.integers(1, n - 1))
                    candidate = _corner_flip(positions, index)
                elif kind < 0.85:
                    candidate = _crankshaft(positions, int(rng.integers(0, max(1, n - 3))))
                else:
                    candidate = _end_move(positions, bool(rng.integers(2)))
                if candidate is None:
                    continue

                candidate_energy = energy_of(sequence, candidate)
                delta = candidate_energy - energy
                if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                    positions, energy = candidate, candidate_energy

            if energy < best_energy:
                best_energy, best_positions = energy, list(positions)

    return Fold(
        sequence=sequence,
        positions=tuple(best_positions or _straight_walk(n)),
        energy=best_energy,
        seconds=time.perf_counter() - started,
        method="simulated annealing",
        metadata={"sweeps": sweeps, "restarts": restarts},
    )


def random_sequence(length: int, hydrophobic_fraction: float = 0.5, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return "".join("H" if rng.random() < hydrophobic_fraction else "P" for _ in range(length))


def turn_encoding_hamiltonian(sequence: str, penalty: float | None = None, threshold: float = 1e-9):
    """The exact Ising Hamiltonian for the turn encoding of ``sequence``.

    Two qubits per bond give the four lattice directions; the first bond is
    fixed to remove the rotational symmetry, so the register is ``2(N-2)``
    qubits.  Self-intersecting conformations are charged ``penalty``.

    Rather than hand-deriving the multi-body self-avoidance terms -- which is
    where published encodings differ from each other and where an error would be
    invisible -- the energy is enumerated over every bitstring and expanded in
    the Pauli-Z basis by a Walsh-Hadamard transform.  That is exact by
    construction: a diagonal function of ``n`` bits has exactly ``2^n``
    Z-coefficients and the transform computes all of them.

    Feasible to about 12 residues (20 qubits, 10^6 states), which is enough to
    measure what the encoding *costs* -- the question Result 62 left open.
    """
    from qiskit.quantum_info import SparsePauliOp

    bonds = len(sequence) - 1
    free_bonds = bonds - 1  # first bond fixed
    n = 2 * free_bonds
    if n > 22:
        raise ValueError(f"{len(sequence)} residues needs {n} qubits; too many to enumerate")

    if penalty is None:
        penalty = float(len(sequence))  # never worth an overlap

    energies = np.empty(1 << n)
    for state in range(1 << n):
        moves = [0]
        for bond in range(free_bonds):
            moves.append((state >> (2 * bond)) & 3)
        value = energy_of(sequence, positions_from_moves(moves))
        energies[state] = penalty if not np.isfinite(value) else value

    # Walsh-Hadamard: coefficients of the Z-basis expansion
    coefficients = energies.copy()
    step = 1
    while step < len(coefficients):
        for start in range(0, len(coefficients), 2 * step):
            for offset in range(start, start + step):
                a, b = coefficients[offset], coefficients[offset + step]
                coefficients[offset], coefficients[offset + step] = a + b, a - b
        step *= 2
    coefficients /= 1 << n

    labels, values = [], []
    for mask in range(1 << n):
        if abs(coefficients[mask]) <= threshold:
            continue
        label = "".join("Z" if (mask >> q) & 1 else "I" for q in reversed(range(n)))
        labels.append(label)
        values.append(coefficients[mask])

    return SparsePauliOp(labels, np.array(values, dtype=complex)), n


def encoding_cost(sequence: str) -> dict:
    """Qubits, Pauli terms by locality, and the QAOA cost-layer gate count.

    A weight-``k`` Z term needs ``2(k-1)`` CX gates per QAOA layer, so locality
    is what decides whether an encoding fits inside a gate budget -- and the
    self-avoidance penalties are exactly the high-weight terms.
    """
    hamiltonian, qubits = turn_encoding_hamiltonian(sequence)

    by_weight: dict[int, int] = {}
    gates = 0
    for pauli in hamiltonian.paulis:
        weight = int(np.count_nonzero(pauli.z))
        by_weight[weight] = by_weight.get(weight, 0) + 1
        if weight >= 2:
            gates += 2 * (weight - 1)

    return {
        "residues": len(sequence),
        "qubits": qubits,
        "terms": len(hamiltonian),
        "terms_by_weight": dict(sorted(by_weight.items())),
        "max_weight": max(by_weight) if by_weight else 0,
        "two_qubit_gates_per_layer": gates,
    }
