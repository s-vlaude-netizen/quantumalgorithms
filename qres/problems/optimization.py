"""Combinatorial optimisation problems as Ising Hamiltonians.

MaxCut is the standard QAOA benchmark; portfolio selection and the
number-partitioning family are included because they are the shapes real users
bring.  All are expressed as diagonal Ising Hamiltonians

    H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

so exact solutions are available by brute force up to ~26 variables, which is
enough to score approximation ratios honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass
class IsingProblem:
    """A diagonal Hamiltonian plus its exact optimum."""

    name: str
    hamiltonian: SparsePauliOp
    #: constant offset dropped from the Hamiltonian (cost = <H> + offset)
    offset: float
    num_variables: int
    #: minimum of <H>; for MaxCut this is -(max cut value) + offset terms
    optimal_value: float
    optimal_bitstrings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return self.hamiltonian.num_qubits

    def cost_of_bitstring(self, bitstring: str) -> float:
        """Energy of a computational-basis assignment (little-endian print)."""
        return _diagonal_value(self.hamiltonian, bitstring) + self.offset

    def approximation_ratio(self, value: float) -> float:
        """Ratio in [0, 1]; 1 means optimal.

        Defined against the *random-guess* baseline so it is comparable across
        instances: (random - achieved) / (random - optimal).
        """
        rnd = self.metadata.get("random_expectation", 0.0)
        denom = rnd - (self.optimal_value + self.offset)
        if abs(denom) < 1e-12:
            return 1.0
        return float((rnd - value) / denom)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_terms": len(self.hamiltonian),
            "optimal_value": self.optimal_value + self.offset,
            **self.metadata,
        }


def _diagonal_value(hamiltonian: SparsePauliOp, bitstring: str) -> float:
    bits = bitstring[::-1]
    total = 0.0
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if np.any(pauli.x):
            raise ValueError("not a diagonal Hamiltonian")
        sign = 1
        for q in range(pauli.num_qubits):
            if pauli.z[q] and bits[q] == "1":
                sign = -sign
        total += sign * coeff.real
    return float(total)


def _brute_force(hamiltonian: SparsePauliOp, offset: float):
    """Exact minimum by enumerating all 2^n assignments (vectorised)."""
    n = hamiltonian.num_qubits
    if n > 26:
        raise ValueError(f"{n} qubits is too many to brute force")
    states = np.arange(2**n, dtype=np.int64)
    # bit b of `states` is the value of qubit b
    bits = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    signs = 1 - 2 * bits  # 0 -> +1, 1 -> -1
    energies = np.zeros(2**n)
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        z = np.asarray(pauli.z, dtype=bool)
        if not z.any():
            energies += coeff.real
            continue
        energies += coeff.real * np.prod(signs[:, z], axis=1)
    best = energies.min()
    winners = np.flatnonzero(energies <= best + 1e-9)[:8]
    strings = [format(int(w), f"0{n}b") for w in winners]
    return float(best), strings, float(energies.mean())


#: largest instance still worth enumerating; above this ``maxcut`` returns a
#: problem with no exact optimum rather than refusing to build one at all.
BRUTE_FORCE_LIMIT = 24


def maxcut(
    edges: list[tuple[int, int]] | list[tuple[int, int, float]],
    num_nodes: int | None = None,
    name: str = "maxcut",
    exact: bool | None = None,
) -> IsingProblem:
    """MaxCut on a weighted graph.

    cut(x) = sum_{(i,j)} w_ij (1 - z_i z_j)/2, maximised.  We minimise
    H = sum w_ij z_i z_j / 2 and carry the constant separately.

    ``exact`` controls whether the optimum is enumerated.  It defaults to
    ``n <= BRUTE_FORCE_LIMIT``, because the interesting region for MaxCut starts
    exactly where enumeration stops (RESEARCH_LOG Result 50: Goemans-Williamson
    is *exactly* optimal on every instance small enough to check).  When it is
    off, ``max_cut`` is ``None`` and ``optimal_value`` is ``nan`` -- a caller
    that wants a score above the limit has to supply its own reference, and gets
    a loud failure rather than a silent zero if it forgets.
    """
    weighted = [(e[0], e[1], e[2] if len(e) > 2 else 1.0) for e in edges]
    n = num_nodes if num_nodes is not None else 1 + max(max(i, j) for i, j, _ in weighted)
    terms, coeffs = [], []
    total_w = 0.0
    for i, j, w in weighted:
        z = ["I"] * n
        z[i] = "Z"
        z[j] = "Z"
        terms.append("".join(reversed(z)))
        coeffs.append(w / 2)
        total_w += w
    ham = SparsePauliOp(terms, np.array(coeffs, dtype=complex)).simplify()
    offset = -total_w / 2

    if exact is None:
        exact = n <= BRUTE_FORCE_LIMIT

    if exact:
        best, strings, mean = _brute_force(ham, offset)
        max_cut = -(best + offset)
        random_expectation = mean + offset
    else:
        best, strings = float("nan"), []
        max_cut = None
        # a uniformly random assignment cuts each edge with probability 1/2
        random_expectation = -total_w / 2

    return IsingProblem(
        name=f"{name}/n{n}/m{len(weighted)}",
        hamiltonian=ham,
        offset=offset,
        num_variables=n,
        optimal_value=best,
        optimal_bitstrings=strings,
        metadata={
            "problem": "maxcut",
            "edges": weighted,
            "max_cut": max_cut,
            "random_expectation": random_expectation,
            "exact": bool(exact),
        },
    )


def random_regular_maxcut(
    n: int, degree: int = 3, seed: int = 0, exact: bool | None = None
) -> IsingProblem:
    """MaxCut on a random d-regular graph -- the canonical QAOA benchmark."""
    rng = np.random.default_rng(seed)
    edges = _random_regular_edges(n, degree, rng)
    p = maxcut(edges, num_nodes=n, name=f"reg{degree}", exact=exact)
    p.metadata["seed"] = seed
    p.metadata["degree"] = degree
    return p


def _random_regular_edges(n: int, d: int, rng, attempts: int = 5000) -> list[tuple[int, int]]:
    """Pairing model with edge-swap repair.

    Plain rejection sampling -- discard the whole draw on the first collision --
    fails often as the degree rises: at ``d = 5, n = 12`` it produced a graph on
    only 3 of 10 seeds, and a study that silently skipped the failures read that
    back as "local search is optimal on 30% of instances" when it had simply
    divided by the wrong denominator.

    Repairing instead: on a collision, swap one endpoint with a random other
    pair, which is the standard fix and succeeds where rejection does not.
    """
    if n * d % 2 != 0:
        raise ValueError(f"n*d must be even (got n={n}, d={d})")
    if d >= n:
        raise ValueError(f"degree {d} needs more than {n} vertices")

    for _ in range(attempts):
        stubs = np.repeat(np.arange(n), d)
        rng.shuffle(stubs)
        pairs = [[int(a), int(b)] for a, b in stubs.reshape(-1, 2)]

        for _ in range(200):
            seen, bad = set(), []
            for index, (a, b) in enumerate(pairs):
                key = (min(a, b), max(a, b))
                if a == b or key in seen:
                    bad.append(index)
                else:
                    seen.add(key)
            if not bad:
                return [(a, b) for a, b in pairs]
            # swap an endpoint of each offending pair with a random other pair
            for index in bad:
                other = int(rng.integers(len(pairs)))
                if other == index:
                    continue
                pairs[index][1], pairs[other][1] = pairs[other][1], pairs[index][1]

    raise RuntimeError(f"failed to sample a simple {d}-regular graph on {n} vertices")


def erdos_renyi_maxcut(
    n: int, p_edge: float = 0.5, seed: int = 0, exact: bool | None = None
) -> IsingProblem:
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p_edge]
    prob = maxcut(edges, num_nodes=n, name=f"er{p_edge:g}", exact=exact)
    prob.metadata["seed"] = seed
    return prob


def qubo_to_ising(
    constant: float, linear: np.ndarray, quadratic: np.ndarray
) -> tuple[SparsePauliOp, float]:
    """Convert  f(x) = constant + c.x + sum_{i<j} Q_ij x_i x_j  to an Ising op.

    Substituting x_i = (1 - z_i)/2 with ``quadratic`` strictly upper-triangular:

        offset = constant + sum_i c_i/2 + sum_{i<j} Q_ij/4
        h_i    = -c_i/2 - sum_{j != i} Q_ij/4
        J_ij   = Q_ij/4
    """
    n = len(linear)
    q = np.triu(quadratic, k=1)
    offset = float(constant + linear.sum() / 2 + q.sum() / 4)
    h = -linear / 2 - (q.sum(axis=1) + q.sum(axis=0)) / 4

    terms, coeffs = [], []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            z = ["I"] * n
            z[i] = "Z"
            terms.append("".join(reversed(z)))
            coeffs.append(h[i])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(q[i, j]) > 1e-12:
                z = ["I"] * n
                z[i] = z[j] = "Z"
                terms.append("".join(reversed(z)))
                coeffs.append(q[i, j] / 4)
    if not terms:  # degenerate, all-constant objective
        terms, coeffs = ["I" * n], [0.0]
    return SparsePauliOp(terms, np.array(coeffs, dtype=complex)).simplify(), offset


def portfolio_optimization(
    n_assets: int = 8,
    budget: int | None = None,
    risk_aversion: float = 0.5,
    penalty: float | None = None,
    seed: int = 0,
) -> IsingProblem:
    """Mean-variance portfolio selection with a cardinality constraint.

    minimise  -mu.x + q x^T Sigma x + A (1.x - B)^2   over x in {0,1}^n

    Using x_i^2 = x_i, the QUBO coefficients are

        constant = A B^2
        c_i      = -mu_i + q Sigma_ii + A (1 - 2B)
        Q_ij     = 2 q Sigma_ij + 2A          (i < j)

    The budget penalty is what makes this hard, and is also what plain QAOA
    handles badly without a constraint-preserving mixer -- a target for later.
    """
    rng = np.random.default_rng(seed)
    n = n_assets
    budget = budget if budget is not None else n // 2
    mu = rng.normal(0.05, 0.03, size=n)
    factor = rng.normal(0, 1, size=(n, 3))
    sigma = factor @ factor.T / 3 + np.diag(rng.uniform(0.05, 0.2, n))
    if penalty is None:
        # large enough that violating the budget by one asset never pays
        penalty = float(np.abs(mu).sum() + risk_aversion * np.abs(sigma).sum())

    constant = penalty * budget**2
    linear = -mu + risk_aversion * np.diag(sigma) + penalty * (1 - 2 * budget)
    quadratic = 2 * risk_aversion * np.triu(sigma, k=1) + 2 * penalty * np.triu(
        np.ones((n, n)), k=1
    )

    ham, offset = qubo_to_ising(constant, linear, quadratic)
    best, strings, mean = _brute_force(ham, offset)
    return IsingProblem(
        name=f"portfolio/n{n}/B{budget}",
        hamiltonian=ham,
        offset=offset,
        num_variables=n,
        optimal_value=best,
        optimal_bitstrings=strings,
        metadata={
            "problem": "portfolio",
            "budget": budget,
            "risk_aversion": risk_aversion,
            "penalty": penalty,
            "seed": seed,
            "expected_returns": mu.tolist(),
            "random_expectation": mean + offset,
        },
    )
