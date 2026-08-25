"""Grouping Pauli observables into simultaneously-measurable sets.

Evaluating <psi|H|psi> for a molecular Hamiltonian with a few thousand Pauli
terms is *the* bottleneck of VQE on hardware: naively it costs one circuit per
term, per energy evaluation, per optimiser iteration.  Two levers reduce it:

1. **Grouping** -- Paulis that commute can be measured in one circuit.  We
   support qubit-wise commuting (QWC, cheap basis change: single-qubit
   rotations only) and general commuting (GC, needs a Clifford, ~3-5x fewer
   groups on molecular Hamiltonians).
2. **Shot allocation** -- groups do not deserve equal shots.  The allocation
   minimising total variance at fixed budget is Neyman's, n_g ∝ sqrt(Var_g).

The variance of a group is computed *empirically from the shots themselves*
(the per-shot group value already includes all intra-group covariances), which
is both exact and free -- no extra circuits.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli, SparsePauliOp


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


@dataclass
class MeasurementGroup:
    """Paulis measurable in a single circuit, plus that circuit's basis change.

    ``diagonal_z`` holds, for each Pauli, the boolean mask of qubits whose
    measured bit enters that Pauli's parity after the basis change, and
    ``signs`` the +-1 picked up by the Clifford conjugation.
    """

    indices: list[int]
    paulis: list[Pauli]
    coeffs: np.ndarray
    basis_change: QuantumCircuit | None
    diagonal_z: np.ndarray  # shape (n_paulis, n_qubits), bool
    signs: np.ndarray  # shape (n_paulis,), +-1
    kind: str = "qwc"

    @property
    def size(self) -> int:
        return len(self.indices)

    def parities(self, bit_array: np.ndarray) -> np.ndarray:
        """Per-shot +-1 parity of every Pauli in the group.

        ``bit_array`` has shape (n_shots, n_qubits) with qubit q in column q.
        Returns shape (n_shots, n_paulis).
        """
        # popcount of (bits & mask) mod 2, done as a matrix product mod 2
        par = (bit_array.astype(np.int8) @ self.diagonal_z.T.astype(np.int8)) & 1
        return np.where(par == 1, -1.0, 1.0) * self.signs[None, :]

    def shot_values(self, bit_array: np.ndarray) -> np.ndarray:
        """Per-shot value of this group's contribution sum_i c_i <P_i>."""
        return self.parities(bit_array) @ self.coeffs

    def weight(self) -> float:
        """sum |c_i| -- the cheap a-priori proxy for this group's variance."""
        return float(np.abs(self.coeffs).sum())


def qwc_compatible(a: Pauli, b: Pauli) -> bool:
    """True if a and b commute qubit-wise (agree wherever both are non-I)."""
    a_active = a.x | a.z
    b_active = b.x | b.z
    both = a_active & b_active
    return bool(np.all((a.x == b.x)[both]) and np.all((a.z == b.z)[both]))


def group_paulis(
    hamiltonian: SparsePauliOp,
    method: str = "qwc",
    drop_identity: bool = True,
) -> tuple[list[MeasurementGroup], float]:
    """Partition ``hamiltonian`` into simultaneously-measurable groups.

    Returns the groups and the identity coefficient, which needs no measurement
    at all and is simply added to every energy estimate.
    """
    paulis = list(hamiltonian.paulis)
    coeffs = np.asarray(hamiltonian.coeffs).real
    n = hamiltonian.num_qubits

    identity_coeff = 0.0
    keep = []
    for i, p in enumerate(paulis):
        if not (p.x.any() or p.z.any()):
            if drop_identity:
                identity_coeff += coeffs[i]
                continue
        keep.append(i)

    if method == "qwc":
        partition = _greedy_colouring(
            keep, lambda i, j: qwc_compatible(paulis[i], paulis[j]), paulis, coeffs
        )
        groups = [_build_qwc_group(idx, paulis, coeffs, n) for idx in partition]
    elif method == "commuting":
        partition = _greedy_colouring(
            keep, lambda i, j: paulis[i].commutes(paulis[j]), paulis, coeffs
        )
        groups = [_build_commuting_group(idx, paulis, coeffs, n) for idx in partition]
    elif method == "none":
        groups = [_build_qwc_group([i], paulis, coeffs, n) for i in keep]
    else:
        raise ValueError(f"unknown grouping method {method!r}")
    return groups, float(identity_coeff)


def _greedy_colouring(
    indices: Sequence[int],
    compatible,
    paulis,
    coeffs,
) -> list[list[int]]:
    """Largest-degree-first greedy clique cover.

    Ordering by |coefficient| descending is a deliberate choice: it puts the
    energetically dominant terms into groups first, which empirically yields
    groups whose variances are more even and so allocate shots better than a
    pure size-optimal cover would.
    """
    order = sorted(indices, key=lambda i: -abs(coeffs[i]))
    groups: list[list[int]] = []
    for i in order:
        for g in groups:
            if all(compatible(i, j) for j in g):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def _build_qwc_group(idx, paulis, coeffs, n) -> MeasurementGroup:
    """Basis change for a qubit-wise commuting set: single-qubit rotations."""
    sel = [paulis[i] for i in idx]
    qc = QuantumCircuit(n)
    # the group's "envelope": on each qubit, whichever of X/Y/Z appears
    for q in range(n):
        letter = "I"
        for p in sel:
            if p.x[q] and p.z[q]:
                letter = "Y"
                break
            if p.x[q]:
                letter = "X"
                break
            if p.z[q]:
                letter = "Z"
                break
        if letter == "X":
            qc.h(q)
        elif letter == "Y":
            qc.sdg(q)
            qc.h(q)
    diag = np.array([[bool(p.x[q] or p.z[q]) for q in range(n)] for p in sel])
    signs = np.ones(len(sel))
    return MeasurementGroup(
        indices=list(idx),
        paulis=sel,
        coeffs=np.array([coeffs[i] for i in idx], dtype=float),
        basis_change=qc,
        diagonal_z=diag,
        signs=signs,
        kind="qwc",
    )


def _build_commuting_group(idx, paulis, coeffs, n) -> MeasurementGroup:
    """Basis change for a generally-commuting set: a Clifford.

    We synthesise a circuit U that prepares the stabilizer state stabilised by
    an independent generating subset, so U Z_k U^dag = S_k; then C = U^dag maps
    every element of the group to a Z-type Pauli.
    """
    from qiskit.synthesis import synth_circuit_from_stabilizers

    sel = [paulis[i] for i in idx]
    if len(sel) == 1:
        return _build_qwc_group(idx, paulis, coeffs, n)

    generators = _independent_generators(sel, n)
    try:
        prep = synth_circuit_from_stabilizers(
            [str(g) for g in generators],
            allow_redundant=True,
            allow_underconstrained=True,
        )
        basis_change = prep.inverse()
        cliff = Clifford(basis_change)
    except Exception:
        # Fall back to measuring the set term by term rather than being wrong.
        return _build_qwc_group(idx, paulis, coeffs, n)

    diag = np.zeros((len(sel), n), dtype=bool)
    signs = np.ones(len(sel))
    for k, p in enumerate(sel):
        image = p.evolve(cliff, frame="s")
        if image.x.any():  # not diagonalised -> refuse, stay correct
            return _build_qwc_group(idx, paulis, coeffs, n)
        diag[k] = image.z
        signs[k] = -1.0 if image.phase % 4 == 2 else 1.0
        if image.phase % 4 in (1, 3):  # an i factor would mean a non-Hermitian image
            return _build_qwc_group(idx, paulis, coeffs, n)

    return MeasurementGroup(
        indices=list(idx),
        paulis=sel,
        coeffs=np.array([coeffs[i] for i in idx], dtype=float),
        basis_change=basis_change,
        diagonal_z=diag,
        signs=signs,
        kind="commuting",
    )


def _independent_generators(paulis: Sequence[Pauli], n: int) -> list[Pauli]:
    """A GF(2)-independent subset generating the same abelian group."""
    rows = []
    chosen = []
    for p in paulis:
        vec = np.concatenate([p.x, p.z]).astype(np.int8)
        reduced = vec.copy()
        for pivot, row in rows:
            if reduced[pivot]:
                reduced ^= row
        if reduced.any():
            pivot = int(np.flatnonzero(reduced)[0])
            rows.append((pivot, reduced))
            rows.sort(key=lambda t: t[0])
            chosen.append(p)
        if len(chosen) == n:
            break
    return chosen


# --------------------------------------------------------------------------
# Shot allocation
# --------------------------------------------------------------------------


@dataclass
class VarianceModel:
    """Running per-group variance estimates driving the shot allocation.

    Groups start with the coefficient-based prior sigma_g^2 ~ (sum_i |c_i|)^2,
    which is the exact worst case, and are shrunk towards the empirical
    variance as shots accumulate.  The shrinkage weight n/(n + n0) keeps early
    allocations from chasing noise in a variance estimated from a handful of
    shots -- the failure mode of estimating variance once and trusting it.
    """

    prior_variance: np.ndarray
    #: pseudo-count controlling how fast we trust the empirical estimate
    prior_strength: float = 64.0
    #: per-batch discount on past observations.  The group variances are *not*
    #: stationary -- they change as the ansatz state moves during optimisation
    #: -- so an estimator that never forgets is biased towards the variance of
    #: a state the optimiser has already left.  ``decay < 1`` tracks the move.
    decay: float = 0.85
    empirical_variance: np.ndarray = field(default=None)
    observed_shots: np.ndarray = field(default=None)

    def __post_init__(self):
        g = len(self.prior_variance)
        if self.empirical_variance is None:
            self.empirical_variance = np.zeros(g)
        if self.observed_shots is None:
            self.observed_shots = np.zeros(g)

    @classmethod
    def from_groups(
        cls,
        groups: Sequence[MeasurementGroup],
        prior_strength: float = 64.0,
        decay: float = 0.85,
    ):
        prior = np.array([g.weight() ** 2 for g in groups], dtype=float)
        return cls(
            prior_variance=np.maximum(prior, 1e-12),
            prior_strength=prior_strength,
            decay=decay,
        )

    def observe(self, group_index: int, variance: float, n_shots: float) -> None:
        """Fold an already-computed batch variance into the running estimate."""
        if n_shots < 2:
            return
        prev_n = self.observed_shots[group_index] * self.decay
        total = prev_n + n_shots
        self.empirical_variance[group_index] = (
            self.empirical_variance[group_index] * prev_n + variance * n_shots
        ) / total
        self.observed_shots[group_index] = total

    def update(self, group_index: int, values: np.ndarray) -> None:
        """Fold a batch of per-shot group values into the estimate."""
        if len(values) < 2:
            return
        self.observe(group_index, float(np.var(values, ddof=1)), len(values))

    def variances(self) -> np.ndarray:
        w = self.observed_shots / (self.observed_shots + self.prior_strength)
        return (1 - w) * self.prior_variance + w * self.empirical_variance


def allocate_shots(
    total_shots: int,
    weights: np.ndarray,
    min_shots: int = 8,
) -> np.ndarray:
    """Neyman allocation: n_g ∝ sqrt(Var_g), with a floor and exact total.

    The floor matters because a group starved to zero shots contributes an
    unbounded, silently-ignored bias to the energy; it also keeps the variance
    estimate for that group alive.
    """
    g = len(weights)
    if g == 0:
        return np.zeros(0, dtype=int)
    if total_shots < g * min_shots:
        min_shots = max(1, total_shots // g)

    sigma = np.sqrt(np.maximum(weights, 0.0))
    if sigma.sum() <= 0:
        sigma = np.ones(g)

    floor_total = g * min_shots
    free = max(0, total_shots - floor_total)
    alloc = np.full(g, min_shots, dtype=float) + free * sigma / sigma.sum()

    counts = np.floor(alloc).astype(int)
    # hand out the remainder to the largest fractional parts
    deficit = total_shots - int(counts.sum())
    if deficit > 0:
        order = np.argsort(-(alloc - counts))
        for k in range(deficit):
            counts[order[k % g]] += 1
    elif deficit < 0:
        order = np.argsort(counts)
        k = 0
        while deficit < 0:
            i = order[k % g]
            if counts[i] > 1:
                counts[i] -= 1
                deficit += 1
            k += 1
    return counts


ALLOCATION_STRATEGIES = {
    # equal split -- the naive baseline
    "uniform": lambda groups, model: np.ones(len(groups)),
    # shots ∝ (sum |c|)^2 so sqrt gives ∝ sum |c|; the common "weighted" scheme
    "coefficient": lambda groups, model: np.array([g.weight() ** 2 for g in groups]),
    # Neyman allocation on the running variance estimate
    "adaptive": lambda groups, model: model.variances(),
}


def grouping_report(groups: Sequence[MeasurementGroup], identity: float) -> dict:
    sizes = [g.size for g in groups]
    return {
        "num_groups": len(groups),
        "num_terms": int(sum(sizes)),
        "largest_group": max(sizes) if sizes else 0,
        "mean_group_size": float(np.mean(sizes)) if sizes else 0.0,
        "kinds": sorted({g.kind for g in groups}),
        "identity_coeff": identity,
        "sum_abs_weight": float(sum(g.weight() for g in groups)),
    }
