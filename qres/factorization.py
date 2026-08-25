"""Double factorisation of the electronic Hamiltonian.

The reason this module exists is the scaling law measured in RESEARCH_LOG
Result 34: the shots needed to reach a target accuracy go as

    shots  ~  (sum |c|)^2 * n_parameters / eps^2

Every other technique in this package buys a constant factor against that.
``sum |c|`` -- the measurement 1-norm of the qubit Hamiltonian -- is the term
that grows with system size, so reducing *it* is the only lever that changes
what is reachable rather than how fast the unreachable is approached.

Double factorisation rewrites the two-body part

    (1/2) sum_{pqrs} g_pqrs a†_p a†_q a_s a_r

by eigendecomposing the two-electron integral tensor reshaped as a matrix,
``G[(pq),(rs)] = sum_t lambda_t L_t L_t^T``, and then diagonalising each
symmetric factor ``L_t = U_t D_t U_t^†``.  In the orbital basis defined by
``U_t`` the factor is a square of a *number-operator* sum, so it is diagonal:
the whole factor is measurable in one basis after a Givens-rotation network,
and its contribution to the 1-norm is ``|lambda_t| (sum_p |d_tp|)^2`` rather
than the sum of many individual Pauli coefficients.

Whether that is actually smaller is an empirical question, which is what
:func:`factorization_report` answers before any circuit is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DoubleFactorization:
    """A double-factorised two-body Hamiltonian."""

    #: one-body integrals in the original (spatial) basis, after absorbing the
    #: contraction that double factorisation shifts out of the two-body term
    one_body: np.ndarray
    #: eigenvalues of the reshaped two-electron tensor, largest |value| first
    factor_weights: np.ndarray
    #: per-factor orbital rotations, shape (n_factors, n_orbitals, n_orbitals)
    factor_rotations: np.ndarray
    #: per-factor diagonal number-operator coefficients
    factor_diagonals: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_factors(self) -> int:
        return len(self.factor_weights)

    def one_norm(self, spin_factor: int = 2) -> float:
        """Measurement 1-norm of the factorised form.

        One-body part contributes the sum of its eigenvalue magnitudes (it is
        one rotation away from diagonal); each two-body factor contributes
        ``|lambda_t| (sum_p |d_tp|)^2``.

        ``spin_factor`` is load-bearing for any comparison against a *qubit*
        Hamiltonian's 1-norm.  The factorisation is done over **spatial**
        orbitals, but the number-operator sum in each factor runs over
        spin-orbitals, so ``sum_p |d_tp|`` doubles and the squared term
        quadruples.  Comparing a spatial-orbital 1-norm against a spin-orbital
        one understates the former by 4x and would manufacture a result.
        """
        one = spin_factor * float(np.abs(np.linalg.eigvalsh(self.one_body)).sum())
        two = float(
            np.sum(
                np.abs(self.factor_weights)
                * (spin_factor * np.abs(self.factor_diagonals).sum(axis=1)) ** 2
            )
        ) / 2.0
        return one + two

    def truncated(self, threshold: float) -> "DoubleFactorization":
        """Drop factors whose weight is below ``threshold`` times the largest.

        Truncation is where double factorisation earns its reputation: the
        eigenvalue spectrum of the reshaped tensor decays quickly, so most of
        the 1-norm sits in a few factors.
        """
        if self.num_factors == 0:
            return self
        keep = np.abs(self.factor_weights) >= threshold * np.abs(self.factor_weights).max()
        return DoubleFactorization(
            one_body=self.one_body,
            factor_weights=self.factor_weights[keep],
            factor_rotations=self.factor_rotations[keep],
            factor_diagonals=self.factor_diagonals[keep],
            metadata={**self.metadata, "truncation_threshold": threshold},
        )


def double_factorize(
    one_body: np.ndarray,
    two_body: np.ndarray,
    tolerance: float = 1e-10,
) -> DoubleFactorization:
    """Factorise ``two_body`` (chemist notation ``g[p,q,r,s]``) into squared
    one-body operators.

    ``two_body`` is expected in the same convention PySCF's ``ao2mo`` produces
    after transformation to the MO basis: ``g[p,q,r,s] = (pq|rs)``.
    """
    n = one_body.shape[0]
    matrix = two_body.reshape(n * n, n * n)
    # symmetrise: (pq|rs) = (rs|pq) holds exactly for real orbitals, but the
    # numerical transform leaves asymmetry at the 1e-12 level that would give
    # complex eigenvalues
    matrix = 0.5 * (matrix + matrix.T)

    weights, vectors = np.linalg.eigh(matrix)
    order = np.argsort(-np.abs(weights))
    weights, vectors = weights[order], vectors[:, order]
    keep = np.abs(weights) > tolerance
    weights, vectors = weights[keep], vectors[:, keep]

    rotations = np.empty((len(weights), n, n))
    diagonals = np.empty((len(weights), n))
    for t in range(len(weights)):
        factor = vectors[:, t].reshape(n, n)
        factor = 0.5 * (factor + factor.T)
        d, u = np.linalg.eigh(factor)
        diagonals[t] = d
        rotations[t] = u

    return DoubleFactorization(
        one_body=one_body,
        factor_weights=weights,
        factor_rotations=rotations,
        factor_diagonals=diagonals,
        metadata={"num_orbitals": n, "tolerance": tolerance},
    )


def pauli_one_norm(hamiltonian) -> float:
    """sum |c| over the non-identity terms of a qubit Hamiltonian.

    The identity term needs no measurement at all, so including it would
    overstate the cost.
    """
    total = 0.0
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if pauli.x.any() or pauli.z.any():
            total += abs(coeff.real)
    return float(total)


def molecular_integrals(problem) -> tuple[np.ndarray, np.ndarray, float]:
    """Spatial-orbital one- and two-body integrals in the MO basis.

    Taken from PySCF directly rather than through qiskit-nature, which returns
    the two-body integrals **8-fold-symmetry packed** (55 numbers for 4
    orbitals, not a 4-index tensor).  ``ao2mo.restore(1, ...)`` gives the
    unpacked chemist-notation tensor ``g[p,q,r,s] = (pq|rs)`` that
    :func:`double_factorize` expects.
    """
    from pyscf import ao2mo, gto, scf

    mol = gto.M(
        atom=problem.metadata["geometry"],
        basis=problem.metadata["basis"],
        charge=0,
        spin=0,
        verbose=0,
    )
    mean_field = scf.RHF(mol)
    mean_field.kernel()
    coefficients = mean_field.mo_coeff
    n = coefficients.shape[1]

    core = mean_field.get_hcore()
    one_body = coefficients.T @ core @ coefficients
    two_body = ao2mo.restore(1, ao2mo.kernel(mol, coefficients), n)
    return one_body, np.asarray(two_body), float(mol.energy_nuc())


def factorization_report(problem, thresholds=(0.0, 1e-4, 1e-3, 1e-2)) -> dict[str, Any]:
    """Compare the raw Pauli 1-norm against the double-factorised one.

    This is the cheap decisive check: if the factorised 1-norm is not
    substantially smaller, no amount of circuit engineering will make the
    approach worth it, and that is worth knowing before building any of it.
    """
    one_body, two_body, _ = molecular_integrals(problem)
    factorised = double_factorize(one_body, two_body)
    raw = pauli_one_norm(problem.hamiltonian)

    rows = []
    for threshold in thresholds:
        truncated = factorised.truncated(threshold) if threshold else factorised
        rows.append(
            {
                "threshold": threshold,
                "factors": truncated.num_factors,
                "one_norm": truncated.one_norm(),
                "ratio_to_pauli": truncated.one_norm() / raw if raw else float("nan"),
            }
        )

    return {
        "molecule": problem.name,
        "num_qubits": problem.num_qubits,
        "num_pauli_terms": len(problem.hamiltonian),
        "pauli_one_norm": raw,
        "num_orbitals": one_body.shape[0],
        "rows": rows,
    }


def factor_operators(factorisation: DoubleFactorization, problem) -> list:
    """Each two-body factor as a qubit operator, in the original orbital basis.

    ``lambda_t (sum_pq [U_t diag(d_t) U_t^T]_pq a†_p a_q)^2`` -- the rotation is
    folded back in rather than applied to the state, so the factors can be
    scored with the same machinery as the Pauli grouping and the comparison is
    like-for-like.
    """
    from qiskit_nature.second_q.operators import FermionicOp

    from .ansatz import _mapper_for

    mapper = _mapper_for(problem)
    n = factorisation.one_body.shape[0]
    operators = []
    for t in range(factorisation.num_factors):
        rotation = factorisation.factor_rotations[t]
        matrix = rotation @ np.diag(factorisation.factor_diagonals[t]) @ rotation.T
        terms = {}
        for p in range(n):
            for q in range(n):
                value = matrix[p, q]
                if abs(value) < 1e-12:
                    continue
                for spin in (0, 1):
                    terms[f"+_{p + spin * n} -_{q + spin * n}"] = value
        one_body_op = FermionicOp(terms, num_spin_orbitals=2 * n)
        squared = (one_body_op @ one_body_op).simplify()
        qubit_op = mapper.map(squared) * factorisation.factor_weights[t]
        operators.append(qubit_op.simplify(atol=1e-12))
    return operators


def variance_comparison(problem, reference=None) -> dict[str, Any]:
    """Total estimator variance: Pauli grouping vs double factorisation.

    The 1-norm is only a proxy, and a badly asymmetric one -- the qubit
    Hamiltonian's identity term needs no measurement and is excluded, while the
    corresponding constant is still inside the factorised form.  The quantity
    that actually sets the shot cost is the total variance under Neyman
    allocation, ``(sum_g sqrt(Var_g))^2``, and both schemes can be scored on it
    at the same state.
    """
    from qiskit.quantum_info import Statevector

    from .ansatz import hartree_fock_state
    from .covariance import covariance_matrix, predicted_total_variance
    from .measurement import group_paulis

    if reference is None:
        reference = Statevector(hartree_fock_state(problem.hf_bitstring))

    coefficients = np.asarray(problem.hamiltonian.coeffs).real
    covariance = covariance_matrix(problem.hamiltonian, reference)
    groups, _ = group_paulis(problem.hamiltonian, "commuting")
    pauli_variance = predicted_total_variance(groups, covariance, coefficients)

    one_body, two_body, _ = molecular_integrals(problem)
    factorisation = double_factorize(one_body, two_body)
    operators = factor_operators(factorisation, problem)

    roots = 0.0
    for operator in operators:
        mean = float(reference.expectation_value(operator).real)
        second = float(reference.expectation_value((operator @ operator).simplify()).real)
        roots += np.sqrt(max(second - mean**2, 0.0))
    factorised_variance = float(roots**2)

    return {
        "molecule": problem.name,
        "pauli_groups": len(groups),
        "pauli_variance": pauli_variance,
        "factors": factorisation.num_factors,
        "factorised_variance": factorised_variance,
        "ratio": factorised_variance / pauli_variance if pauli_variance else float("nan"),
    }
