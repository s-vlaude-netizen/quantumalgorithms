"""Covariance-aware Pauli grouping.

Standard grouping minimises the *number* of measurement groups.  That is a
proxy, and not always a good one.  At a fixed shot budget under Neyman
allocation the quantity actually minimised is

    total variance  =  (sum_g sqrt(Var_g))^2 / N

and a group's variance is not the sum of its terms' variances:

    Var_g = sum_{i,j in g} c_i c_j Cov(P_i, P_j)

The cross terms are the point.  Putting two *anticorrelated* observables in the
same group makes that group cheaper to estimate than either term alone would
suggest, while two strongly correlated ones make it more expensive.  A grouping
built to exploit that can beat a count-optimal one at equal shots -- and it
composes with the Neyman allocation, which then spreads shots over the improved
group variances.

The covariances are evaluated at a cheap classical reference state (Hartree-Fock
by default).  That is an approximation: the true covariances are those of the
converged VQE state, which is unknown.  Whether the reference is close enough to
be useful is an empirical question, and the point of measuring it.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

from .measurement import MeasurementGroup, _build_commuting_group, _build_qwc_group

#: Covariances are rounded to this many decimals before any greedy step sees
#: them.  See :func:`covariance_matrix` -- without it the partition depends on
#: numerical noise 12 orders of magnitude below anything physical.
COVARIANCE_DECIMALS = 9


def pauli_expectations(paulis, state: Statevector) -> np.ndarray:
    """<P> for a list of Paulis at ``state``, vectorised over the state."""
    data = state.data
    n = state.num_qubits
    out = np.empty(len(paulis), dtype=float)
    idx = np.arange(len(data), dtype=np.int64)
    for k, p in enumerate(paulis):
        out[k] = _single_expectation(p, data, idx, n)
    return out


def _single_expectation(pauli: Pauli, data: np.ndarray, idx: np.ndarray, n: int) -> float:
    """<psi|P|psi> using the bitmask action of a Pauli on basis states.

    A Pauli is ``(-i)^g Z^z X^x`` for the *group* phase ``g``, so

        P|b> = (-i)^g (-1)^popcount((b XOR x) AND z) |b XOR x>

    and reindexing by ``b' = b XOR x`` turns the expectation into a single
    vectorised gather.

    ``Pauli.phase`` is **not** ``g``: it is measured relative to the
    ``(-i)^(x.z)`` convention, so ``Pauli("Y").phase`` is 0 while its group
    phase is 1.  The two differ by the number of Y factors, and using
    ``.phase`` directly silently mis-signs every term containing a Y.
    """
    x_mask = _mask(pauli.x)
    z_mask = _mask(pauli.z)
    parity = _popcount(idx & z_mask) & 1
    signs = np.where(parity == 1, -1.0, 1.0)
    amp = np.vdot(data, signs * data[idx ^ x_mask])
    group_phase = int(pauli.phase) + int(np.count_nonzero(pauli.x & pauli.z))
    return float(np.real(amp * (-1j) ** group_phase))


def _mask(bits: np.ndarray) -> int:
    m = 0
    for i, b in enumerate(bits):
        if b:
            m |= 1 << i
    return m


def _popcount(a: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a)
    counts = np.zeros_like(a)
    v = a.copy()
    while np.any(v):
        counts += v & 1
        v >>= 1
    return counts


def covariance_matrix(hamiltonian: SparsePauliOp, state: Statevector) -> np.ndarray:
    """Cov(P_i, P_j) = <P_i P_j> - <P_i><P_j> for all pairs, at ``state``.

    Only defined (real) for pairs that commute; for non-commuting pairs the
    symmetrised value ``Re<P_i P_j>`` is used, which is what enters the variance
    of a simultaneously-measured estimator anyway.
    """
    paulis = list(hamiltonian.paulis)
    m = len(paulis)
    means = pauli_expectations(paulis, state)

    data = state.data
    idx = np.arange(len(data), dtype=np.int64)
    n = state.num_qubits

    cov = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            product = paulis[i].compose(paulis[j])
            val = _single_expectation(product, data, idx, n)
            cov[i, j] = cov[j, i] = val - means[i] * means[j]
    # Rounded before anything greedy consumes it.  Two reference states that
    # agree to overlap 1.000000 (max amplitude difference 2.6e-12) produced
    # covariance matrices differing at ~1e-12, which flipped near-ties in the
    # grouping and changed H4's partition from 11 groups to 19.  Covariances
    # here are O(1e-2) to O(1), so discarding everything below 1e-9 is
    # physically meaningless and removes the whole failure mode.
    return np.round(cov, COVARIANCE_DECIMALS)


def group_variance(indices, coeffs: np.ndarray, cov: np.ndarray) -> float:
    """sum_{i,j} c_i c_j Cov_ij for the terms in one group."""
    sel = np.asarray(indices, dtype=int)
    c = coeffs[sel]
    return float(c @ cov[np.ix_(sel, sel)] @ c)


def covariance_grouping(
    hamiltonian: SparsePauliOp,
    reference: Statevector,
    method: str = "commuting",
    max_groups: int | None = None,
) -> tuple[list[MeasurementGroup], float]:
    """Greedy grouping that minimises total sqrt-variance rather than count.

    Terms are placed heaviest-first.  Each term goes into whichever compatible
    existing group gives the smallest increase in ``sqrt(Var_g)`` -- opening a
    new group when that is cheaper, which it is whenever the term would be
    strongly positively correlated with everything already grouped.
    """
    paulis = list(hamiltonian.paulis)
    coeffs = np.asarray(hamiltonian.coeffs).real
    n = hamiltonian.num_qubits

    identity_coeff = 0.0
    keep = []
    for i, p in enumerate(paulis):
        if not (p.x.any() or p.z.any()):
            identity_coeff += coeffs[i]
        else:
            keep.append(i)

    cov = covariance_matrix(hamiltonian, reference)
    compatible = (
        (lambda a, b: paulis[a].commutes(paulis[b]))
        if method == "commuting"
        else (lambda a, b: _qwc(paulis[a], paulis[b]))
    )

    # rounded magnitude, ties broken on the label -- see _greedy_colouring
    order = sorted(keep, key=lambda i: (-round(abs(coeffs[i]), 12), str(paulis[i])))
    groups: list[list[int]] = []
    variances: list[float] = []

    for i in order:
        best_g, best_key = None, None
        # cost of opening a new group for this term alone
        solo = np.sqrt(max(group_variance([i], coeffs, cov), 0.0))
        for gi, members in enumerate(groups):
            if not all(compatible(i, j) for j in members):
                continue
            new_var = group_variance(members + [i], coeffs, cov)
            delta = np.sqrt(max(new_var, 0.0)) - np.sqrt(max(variances[gi], 0.0))
            # Rounded delta with the group index as tie-break.  Raw float
            # comparison here is the second place near-ties made the partition
            # depend on floating-point noise: two candidate groups agreeing to
            # 1e-12 would swap, changing the whole downstream partition.
            key = (round(float(delta), 12), gi)
            if best_key is None or key < best_key:
                best_key, best_g = key, gi
        best_delta = best_key[0] if best_key is not None else np.inf
        allow_new = max_groups is None or len(groups) < max_groups
        if best_g is not None and (best_delta <= solo or not allow_new):
            groups[best_g].append(i)
            variances[best_g] = group_variance(groups[best_g], coeffs, cov)
        else:
            groups.append([i])
            variances.append(group_variance([i], coeffs, cov))

    builder = _build_commuting_group if method == "commuting" else _build_qwc_group
    built = [builder(g, paulis, coeffs, n) for g in groups]
    return built, float(identity_coeff)


def refine_partition(
    partition: list[list[int]],
    coeffs: np.ndarray,
    cov: np.ndarray,
    compatible,
    *,
    max_sweeps: int = 20,
    allow_new_groups: bool = True,
) -> tuple[list[list[int]], int]:
    """Local search: move single terms between groups while total sqrt-var falls.

    Motivated by a measurement, not by taste.  In Result 7 the *free*
    Hartree-Fock covariance reference beat the exact-ground-state oracle
    (0.778 vs 0.844 on LiH).  Perfect information producing a worse partition
    means the greedy placement -- not the covariance -- is what limits the
    method, so there should be headroom below the greedy result reachable
    without any better reference.

    Steepest-descent over single-term moves.  Each sweep evaluates every term
    against every compatible destination and applies the single best improving
    move, repeating until no move helps.  Returns the refined partition and the
    number of moves applied.
    """
    parts = [list(g) for g in partition]
    variances = [group_variance(g, coeffs, cov) for g in parts]
    moves = 0

    for _ in range(max_sweeps):
        best = None  # (gain, term, source, destination)
        for src, members in enumerate(parts):
            if len(members) == 0:
                continue
            src_root = np.sqrt(max(variances[src], 0.0))
            for term in members:
                remainder = [t for t in members if t != term]
                new_src = np.sqrt(max(group_variance(remainder, coeffs, cov), 0.0))
                freed = src_root - new_src

                destinations = [
                    d
                    for d in range(len(parts))
                    if d != src and all(compatible(term, t) for t in parts[d])
                ]
                for dst in destinations:
                    added = (
                        np.sqrt(max(group_variance(parts[dst] + [term], coeffs, cov), 0.0))
                        - np.sqrt(max(variances[dst], 0.0))
                    )
                    gain = freed - added
                    if gain > 1e-12 and (best is None or gain > best[0]):
                        best = (gain, term, src, dst)
                if allow_new_groups and len(members) > 1:
                    solo = np.sqrt(max(group_variance([term], coeffs, cov), 0.0))
                    gain = freed - solo
                    if gain > 1e-12 and (best is None or gain > best[0]):
                        best = (gain, term, src, None)

        if best is None:
            break
        _, term, src, dst = best
        parts[src].remove(term)
        variances[src] = group_variance(parts[src], coeffs, cov)
        if dst is None:
            parts.append([term])
            variances.append(group_variance([term], coeffs, cov))
        else:
            parts[dst].append(term)
            variances[dst] = group_variance(parts[dst], coeffs, cov)
        # Only the two touched groups changed; recomputing every group's
        # variance here costs more than the whole move search on LiH.
        if not parts[src]:
            parts.pop(src)
            variances.pop(src)
        moves += 1

    return parts, moves


def refined_covariance_grouping(
    hamiltonian: SparsePauliOp,
    reference: Statevector,
    method: str = "commuting",
    max_sweeps: int = 20,
) -> tuple[list[MeasurementGroup], float, int]:
    """Covariance grouping followed by local refinement."""
    paulis = list(hamiltonian.paulis)
    coeffs = np.asarray(hamiltonian.coeffs).real
    n = hamiltonian.num_qubits

    groups, identity = covariance_grouping(hamiltonian, reference, method)
    cov = covariance_matrix(hamiltonian, reference)
    compatible = (
        (lambda a, b: paulis[a].commutes(paulis[b]))
        if method == "commuting"
        else (lambda a, b: _qwc(paulis[a], paulis[b]))
    )
    partition, moves = refine_partition(
        [list(g.indices) for g in groups], coeffs, cov, compatible, max_sweeps=max_sweeps
    )
    builder = _build_commuting_group if method == "commuting" else _build_qwc_group
    return [builder(g, paulis, coeffs, n) for g in partition], identity, moves


def _qwc(a: Pauli, b: Pauli) -> bool:
    from .measurement import qwc_compatible

    return qwc_compatible(a, b)


def predicted_total_variance(groups, cov: np.ndarray, coeffs: np.ndarray) -> float:
    """(sum_g sqrt(Var_g))^2 -- the quantity Neyman allocation minimises.

    This is the number to compare groupings on.  Group *count* is only a proxy
    for it, and the whole point of covariance-aware grouping is that the proxy
    and the objective can disagree.
    """
    total = 0.0
    for g in groups:
        total += np.sqrt(max(group_variance(g.indices, coeffs, cov), 0.0))
    return float(total**2)
