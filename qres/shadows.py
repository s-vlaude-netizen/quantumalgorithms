"""Classical shadow estimators, scored against grouped measurement.

Two variants, and the difference between them is the whole question:

**Random-Pauli shadows** measure each qubit in a uniformly random X/Y/Z basis.
The estimator's variance for a Pauli of weight ``k`` carries a ``3^k`` factor,
which is fatal for fermionic Hamiltonians: Jordan-Wigner strings produce
high-weight Paulis by construction, and on LiH the weight-9 and weight-10 terms
are 33 of 631 while carrying 67% of the variance (RESEARCH_LOG Result 36).

**Derandomised shadows** (Huang, Chen, Preskill 2021) choose the measurement
settings greedily to cover the *specific* observables at hand, removing the
``3^k`` penalty for terms that get deliberately targeted. That is the version
worth testing, and this module implements it so the comparison is measured
rather than argued.

Note what derandomisation can and cannot do: its settings are **product bases**,
so it can only ever measure qubit-wise-commuting sets together. General
commuting grouping uses a Clifford and is strictly more powerful per setting --
15-20x fewer settings on these Hamiltonians. Whether derandomisation's better
*allocation* makes up for its weaker *grouping* is the empirical question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def pauli_supports(hamiltonian) -> tuple[np.ndarray, np.ndarray]:
    """Per-term single-qubit basis requirement, as integer codes.

    ``0`` means identity (the term does not care what this qubit is measured
    in), ``1/2/3`` mean X/Y/Z.  A product-basis setting measures a term iff it
    matches on every qubit the term is non-identity on.
    """
    n = hamiltonian.num_qubits
    codes = np.zeros((len(hamiltonian), n), dtype=np.int8)
    for i, pauli in enumerate(hamiltonian.paulis):
        for q in range(n):
            x, z = bool(pauli.x[q]), bool(pauli.z[q])
            if x and z:
                codes[i, q] = 2  # Y
            elif x:
                codes[i, q] = 1  # X
            elif z:
                codes[i, q] = 3  # Z
    weights = (codes != 0).sum(axis=1)
    return codes, weights


def random_pauli_shadow_variance(hamiltonian) -> float:
    """Single-snapshot variance bound of the random-Pauli shadow estimator.

    ``sum_i c_i^2 3^{weight_i}`` -- the standard bound, and the one that makes
    the method unusable on fermionic Hamiltonians.
    """
    _, weights = pauli_supports(hamiltonian)
    coeffs = np.asarray(hamiltonian.coeffs).real
    keep = weights > 0
    return float(np.sum(coeffs[keep] ** 2 * 3.0 ** weights[keep]))


@dataclass
class DerandomisedSettings:
    """Product-basis measurement settings chosen for a specific Hamiltonian."""

    settings: np.ndarray  # (n_settings, n_qubits) of codes 1/2/3
    hit_counts: np.ndarray  # (n_terms,) how many settings measure each term

    @property
    def num_settings(self) -> int:
        return len(self.settings)


def derandomise(
    hamiltonian, num_settings: int = 64, decay: float = 0.5, max_extra: int = 4096
) -> DerandomisedSettings:
    """Greedily choose product bases covering the Hamiltonian's terms.

    Follows the derandomisation of Huang, Chen and Preskill: build each setting
    qubit by qubit, at every step picking the basis that most reduces a
    confidence bound over the terms not yet well covered.  The bound is
    ``sum_i c_i^2 exp(-decay * hits_i)``, so a term with zero hits contributes
    its full weight and the greedy is driven to reach it.

    **Coverage is then enforced.**  A term no setting measures makes the
    estimator undefined, not merely imprecise -- a first version using a
    ``1/(1+hits)`` proxy left terms at zero hits and produced an infinite
    variance.  Extra settings are appended until every term is hit at least
    once, and the resulting count is reported rather than the requested one.

    That count has a floor worth stating: any product-basis scheme needs at
    least as many settings as a minimum qubit-wise-commuting clique cover just
    to touch every term.  Derandomisation cannot beat QWC grouping on *setting
    count*; its only possible advantage is in how it allocates repetitions.
    """
    codes, weights = pauli_supports(hamiltonian)
    coeffs = np.asarray(hamiltonian.coeffs).real
    active = weights > 0
    codes, coeffs = codes[active], coeffs[active]
    n_terms, n_qubits = codes.shape

    hits = np.zeros(n_terms, dtype=float)
    settings = np.zeros((num_settings, n_qubits), dtype=np.int8)

    def build_one() -> tuple[np.ndarray, np.ndarray]:
        setting = np.zeros(n_qubits, dtype=np.int8)
        alive = np.ones(n_terms, dtype=bool)
        value = coeffs**2 * np.exp(-decay * hits)
        for q in range(n_qubits):
            best_basis, best_score = 1, -np.inf
            for basis in (1, 2, 3):
                compatible = alive & ((codes[:, q] == 0) | (codes[:, q] == basis))
                score = float(value[compatible].sum())
                if score > best_score:
                    best_basis, best_score = basis, score
            setting[q] = best_basis
            alive &= (codes[:, q] == 0) | (codes[:, q] == best_basis)
        return setting, alive

    chosen = []
    for _ in range(num_settings):
        setting, alive = build_one()
        chosen.append(setting)
        hits += alive

    # enforce coverage: an unmeasured term is an undefined estimator, not an
    # imprecise one
    extra = 0
    while hits.min() == 0 and extra < max_extra:
        setting, alive = build_one()
        if not alive[hits == 0].any():
            # this setting reaches nothing new; force it onto the worst term
            worst = int(np.argmin(hits))
            setting = np.where(codes[worst] == 0, setting, codes[worst]).astype(np.int8)
            alive = np.all((codes == 0) | (codes == setting[None, :]), axis=1)
        chosen.append(setting)
        hits += alive
        extra += 1

    settings = np.array(chosen, dtype=np.int8)
    return DerandomisedSettings(settings=settings, hit_counts=hits)


def derandomised_variance(hamiltonian, chosen: DerandomisedSettings, reference) -> float:
    """Total estimator variance from a derandomised setting list.

    Shots are split evenly across settings, and each term is estimated from the
    settings that measure it.  Scored the same way as the grouped estimator so
    the two numbers are comparable: variance at a fixed total shot budget of 1.
    """
    from .covariance import covariance_matrix

    codes, weights = pauli_supports(hamiltonian)
    coeffs = np.asarray(hamiltonian.coeffs).real
    active = np.flatnonzero(weights > 0)
    covariance = covariance_matrix(hamiltonian, reference)

    n_settings = chosen.num_settings
    total = 0.0
    for local, index in enumerate(active):
        hits = chosen.hit_counts[local]
        if hits <= 0:
            # never measured -- an unbounded contribution, reported as such
            return float("inf")
        # fraction of the budget this term's estimator actually receives
        share = hits / n_settings
        total += coeffs[index] ** 2 * covariance[index, index] / share
    return float(total)


def shadow_report(problem, num_settings: int | None = None) -> dict[str, Any]:
    """Compare both shadow variants against general-commuting grouping."""
    from qiskit.quantum_info import Statevector

    from .ansatz import hartree_fock_state
    from .covariance import covariance_matrix, predicted_total_variance
    from .measurement import group_paulis

    reference = Statevector(hartree_fock_state(problem.hf_bitstring))
    coeffs = np.asarray(problem.hamiltonian.coeffs).real
    covariance = covariance_matrix(problem.hamiltonian, reference)

    groups, _ = group_paulis(problem.hamiltonian, "commuting")
    grouped = predicted_total_variance(groups, covariance, coeffs)

    qwc_groups, _ = group_paulis(problem.hamiltonian, "qwc")
    qwc = predicted_total_variance(qwc_groups, covariance, coeffs)

    settings = num_settings if num_settings is not None else len(qwc_groups)
    chosen = derandomise(problem.hamiltonian, settings)
    derandomised = derandomised_variance(problem.hamiltonian, chosen, reference)

    return {
        "molecule": problem.name,
        "terms": len(problem.hamiltonian),
        "commuting_groups": len(groups),
        "commuting_variance": grouped,
        "qwc_groups": len(qwc_groups),
        "qwc_variance": qwc,
        "derandomised_settings": chosen.num_settings,
        "derandomised_variance": derandomised,
        "random_shadow_variance": random_pauli_shadow_variance(problem.hamiltonian),
        "min_hits": float(chosen.hit_counts.min()),
        "median_hits": float(np.median(chosen.hit_counts)),
    }
