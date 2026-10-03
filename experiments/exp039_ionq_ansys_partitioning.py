"""Experiment 039 -- did a quantum computer make LS-DYNA 12% faster than classical computing?

**The claim.** IonQ press release, 20 March 2025: "IonQ and Ansys Achieve Major
Quantum Computing Milestone -- Demonstrating Quantum Outperforming Classical
Computing". LS-DYNA ran "up to 12 percent faster ... compared to classical
computing", "one of the first cases ever where quantum computing is outperforming
key classical methods". The paper behind it is arXiv:2503.13128.

**What the quantum computer did.** LS-DYNA's sparse direct solver orders its
matrix by nested dissection, which needs graph bisections. The paper coarsens
the mesh graph to **10-32 vertices**, bisects the coarse graph with VarQITE on the
QUBO (their Eq. 3)

    C(x) = sum_(i,j) w_ij (x_i + x_j - 2 x_i x_j) + lambda (sum_i v_i x_i - V/2)^2 ,

projects back, refines with Fiduccia-Mattheyses, and for the wall-clock runs uses
"the partition which yielded the lowest WCT amongst the 10 lowest energy
partitions from the solver's optimized distribution". In noiseless simulation
"the optimal solution is found in each case". The baseline is LS-GPart at its
production coarsening. **No classical solver was run on the same coarse graphs.**

**What this measures.**

1. *Exact, for every instance.* A coarse graph of ``n <= 32`` vertices has
   ``2^(n-1)`` bisections up to the ``x <-> 1 - x`` symmetry of ``C``. Enumerating
   all of them gives the optimum **and the 10 lowest-energy partitions** -- the
   whole set the 12% was selected from -- deterministically. The work is
   instance-independent, so its time at ``n = 32`` bounds every instance the paper
   could have used.
2. *Heuristics.* On FEA-type meshes (2-D shell with openings, 3-D hex housing,
   unstructured triangulation) coarsened to 10-32 vertices by heavy-edge matching,
   how often METIS, multi-start Fiduccia-Mattheyses on ``C`` and spectral
   bisection reach the exact optimum, and how fast.
3. *VarQITE as specified* -- ``G theta_dot = D`` with
   ``G_aj = Re<psi|P_a d_j psi>``, ``D_a = -1/2 <{P_a, H - E}>``, RZY
   HeavyNeighbors ansatz, ``|+>^n``, zero initial angles, forward Euler, 2000 shots
   -- simulated noiselessly as they did, at 10-14 qubits: does it sample the
   optimum, and how many circuits does it spend (their count: ``2m + 1`` per step).

If (1) holds, the quantum step's entire output is produced classically, exactly,
in seconds, with no quantum computer. Whatever the 12% measures, it is then not
quantum outperforming classical: it is a comparison between two *pipelines*, and
the better one runs as well with a classical solver in the place of VarQITE.

Needs pymetis for the METIS arm (``pip install pymetis``); without it that arm
is skipped.

Run:  python -m experiments.exp039_ionq_ansys_partitioning
          [--sizes 10,12,...,32] [--seeds 2] [--varqite-sizes 10,12,14]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qres.bench import RESULTS_DIR

#: the paper's balance tolerance: each side at most (1/2 + NU) of the total weight
NU = 0.05
#: the paper's "10 lowest energy partitions"
KEEP = 10
#: shots per readout, as in the paper's noiseless simulations
SHOTS = 2000
MESHES = ("shell", "housing", "unstructured")
SIZES = tuple(range(10, 33, 2))
VARQITE_SIZES = (10, 12, 14)


# ----------------------------------------------------------------- meshes

def _graph_from_elements(elements, nodes):
    """Stiffness-matrix sparsity graph: two nodes adjacent iff they share an element."""
    rows, cols = [], []
    width = elements.shape[1]
    for a in range(width):
        for b in range(a + 1, width):
            rows.append(elements[:, a])
            cols.append(elements[:, b])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    used = np.unique(elements)
    relabel = np.full(nodes, -1)
    relabel[used] = np.arange(len(used))
    rows, cols = relabel[rows], relabel[cols]
    graph = sp.coo_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)),
                          shape=(len(used),) * 2).tocsr()
    graph = graph + graph.T
    graph.data[:] = 1
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def shell_mesh(nx=96, ny=64):
    """A 2-D quadrilateral shell panel with two rectangular openings (a roof)."""
    openings = ((18, 34, 14, 30), (54, 76, 30, 50))
    node = lambda i, j: i * (ny + 1) + j
    quads = [(node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1))
             for i in range(nx) for j in range(ny)
             if not any(x0 <= i < x1 and y0 <= j < y1 for x0, x1, y0, y1 in openings)]
    return _graph_from_elements(np.array(quads), (nx + 1) * (ny + 1))


def housing_mesh(nr=3, nt=56, nz=36):
    """A 3-D hexahedral hollow cylinder, periodic in the angle (a pump housing)."""
    node = lambda r, t, z: (r * nt + t % nt) * (nz + 1) + z
    hexes = [(node(r, t, z), node(r + 1, t, z), node(r + 1, t + 1, z), node(r, t + 1, z),
              node(r, t, z + 1), node(r + 1, t, z + 1), node(r + 1, t + 1, z + 1),
              node(r, t + 1, z + 1))
             for r in range(nr) for t in range(nt) for z in range(nz)]
    return _graph_from_elements(np.array(hexes), (nr + 1) * nt * (nz + 1))


def unstructured_mesh(points=6000, seed=0):
    """Delaunay triangles on a 2 x 1 plate with a circular hole."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    xy = rng.uniform((0, 0), (2, 1), size=(4 * points, 2))
    xy = xy[np.hypot(xy[:, 0] - 0.6, xy[:, 1] - 0.5) > 0.22][:points]
    triangles = Delaunay(xy).simplices
    centre = xy[triangles].mean(axis=1)
    keep = np.hypot(centre[:, 0] - 0.6, centre[:, 1] - 0.5) > 0.22
    return _graph_from_elements(triangles[keep], len(xy))


def mesh(kind):
    return {"shell": shell_mesh, "housing": housing_mesh,
            "unstructured": unstructured_mesh}[kind]()


# ------------------------------------------------------------- coarsening

def coarsen(graph, target, seed=0, return_map=False, return_levels=False):
    """Multilevel heavy-edge matching down to exactly ``target`` vertices.

    METIS-style: visit vertices in random order and contract each with its
    unmatched neighbour of heaviest connecting edge, refusing merges heavier than
    1.5x the final average vertex weight. Vertex weights count fine vertices, edge
    weights count fine edges, so the coarse cut equals the fine cut of the
    projected partition. With ``return_map`` also returns each fine vertex's
    coarse vertex; with ``return_levels``, every level's ``(graph, weights,
    group)``, finest first, for multilevel refinement on the way back.
    """
    rng = np.random.default_rng(seed)
    graph = graph.tocsr().astype(np.int64)
    weights = np.ones(graph.shape[0], dtype=np.int64)
    fine_to_coarse = np.arange(graph.shape[0])
    levels = []
    cap = 1.5 * weights.sum() / target
    while graph.shape[0] > target:
        n = graph.shape[0]
        mate = np.full(n, -1)
        merges = 0
        for u in rng.permutation(n):
            if mate[u] >= 0:
                continue
            start, stop = graph.indptr[u], graph.indptr[u + 1]
            best, best_weight = -1, 0
            for v, w in zip(graph.indices[start:stop], graph.data[start:stop]):
                if mate[v] < 0 and v != u and w > best_weight and weights[u] + weights[v] <= cap:
                    best, best_weight = v, w
            if best >= 0:
                mate[u], mate[best] = best, u
                merges += 1
                if n - merges == target:
                    break
            else:
                mate[u] = u
        if merges == 0:
            cap *= 1.25
            continue
        group = np.full(n, -1)
        count = 0
        for u in range(n):
            if group[u] < 0:
                group[u] = count
                if mate[u] >= 0:
                    group[mate[u]] = count
                count += 1
        levels.append((graph, weights, group))
        project = sp.csr_matrix((np.ones(n, dtype=np.int64), (np.arange(n), group)),
                                shape=(n, count))
        graph = (project.T @ graph @ project).tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()
        weights = project.T @ weights
        fine_to_coarse = group[fine_to_coarse]
    if return_levels:
        return graph.toarray(), weights, levels
    if return_map:
        return graph.toarray(), weights, fine_to_coarse
    return graph.toarray(), weights


# -------------------------------------------------------- the paper's QUBO

def penalty_weight(adjacency, weights, nu=NU):
    """lambda at which a ``nu`` imbalance costs as much as cutting every edge.

    The paper leaves lambda unstated. This is the natural scale: below it the
    optimum may trade balance for cut, far above it the cut barely registers.
    """
    return np.triu(adjacency).sum() / (nu * weights.sum()) ** 2


def energy(x, adjacency, weights, lam):
    """The paper's Eq. 3, directly."""
    x = np.asarray(x, dtype=np.int64)
    cut = (adjacency * (x[:, None] ^ x[None, :])).sum() // 2
    return cut + lam * (weights @ x - weights.sum() / 2) ** 2


def qubo_form(adjacency, weights, lam):
    """``C(x) = const + h.x + x^T J x / 2`` with symmetric, zero-diagonal ``J``."""
    a = adjacency.astype(float)
    v = weights.astype(float)
    total = v.sum()
    h = a.sum(axis=1) + lam * (v ** 2 - total * v)
    coupling = -2 * a + 2 * lam * np.outer(v, v)
    np.fill_diagonal(coupling, 0.0)
    return coupling, h, lam * total ** 2 / 4


def balanced(x, weights, nu=NU):
    return abs(weights @ np.asarray(x) - weights.sum() / 2) <= nu * weights.sum()


# ------------------------------------------------------- exact enumeration

def exhaustive(adjacency, weights, lam, keep=KEEP, nu=NU, low_bits=22):
    """Every bisection of the coarse graph, streamed in chunks.

    Vertex 0 is fixed to side 0, which loses nothing: ``C(x) = C(1 - x)``. The
    remaining ``n - 1`` bits are split into ``l`` low bits, tabulated once, and
    ``h`` high bits, looped over; for a fixed high assignment the cut is the
    low-internal table plus a term linear in the low bits, built by doubling.
    Cuts and side weights are exact integers.

    Returns the ``keep`` lowest energies with their partitions, the number of
    optimal partitions, and the minimum cut among ``nu``-balanced partitions.
    """
    adjacency = np.asarray(adjacency, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)
    n = len(weights)
    total = int(weights.sum())
    l = min(n - 1, low_bits)
    low, high = np.arange(1, 1 + l), np.arange(1 + l, n)
    index = np.arange(2 ** l, dtype=np.int64)

    # cut among vertex 0 and the low vertices, by doubling: adding vertex j on
    # side 1 cuts its edges to earlier vertices on side 0, on side 0 those on side 1
    cut_low = np.zeros(2 ** l, dtype=np.int64)
    for j, vertex in enumerate(low):
        earlier = adjacency[vertex, low[:j]]
        to_side_one = _doubling(earlier)
        cut_low[2 ** j: 2 ** (j + 1)] = (cut_low[: 2 ** j] + adjacency[vertex, 0]
                                          + earlier.sum() - to_side_one)
        cut_low[: 2 ** j] += to_side_one
    inside = np.concatenate(([0], low))
    weight_low = _doubling(weights[low])

    cross = adjacency[np.ix_(high, low)]
    to_inside = adjacency[np.ix_(high, inside)].sum(axis=1)
    within_high = adjacency[np.ix_(high, high)]
    best_balanced = (np.iinfo(np.int64).max, None)
    kept_energy, kept_x = np.empty(0), np.empty((0, n), dtype=np.int8)
    ground, ground_count = np.inf, 0

    for chunk in range(2 ** len(high)):
        xh = (chunk >> np.arange(len(high))) & 1
        signs = 1 - 2 * xh
        linear = _doubling(cross.T @ signs)
        const = int(xh @ to_inside) + int((within_high * (xh[:, None] ^ xh[None, :])).sum() // 2)
        cut = cut_low + linear + const
        deviation = 2 * (weight_low + int(xh @ weights[high])) - total
        e = cut + lam * deviation.astype(float) ** 2 / 4

        lowest = e.min()
        tol = 1e-9 * max(1.0, abs(lowest))
        if lowest < ground - tol:
            ground, ground_count = lowest, int((e <= lowest + tol).sum())
        elif lowest <= ground + tol:
            ground_count += int((e <= ground + tol).sum())

        # only partitions that could enter the running list are sorted
        threshold = kept_energy[-1] if len(kept_energy) == keep else np.inf
        candidates = np.flatnonzero(e <= threshold)
        if len(candidates) > keep:
            candidates = candidates[np.argpartition(e[candidates], keep - 1)[:keep]]
        take = candidates
        xs = np.zeros((len(take), n), dtype=np.int8)
        xs[:, low] = (index[take, None] >> np.arange(l)) & 1
        xs[:, high] = xh
        kept_energy = np.concatenate((kept_energy, e[take]))
        kept_x = np.concatenate((kept_x, xs))
        order = np.argsort(kept_energy, kind="stable")[:keep]
        kept_energy, kept_x = kept_energy[order], kept_x[order]

        ok = np.abs(deviation) <= 2 * nu * total
        if ok.any():
            i = int(np.argmin(np.where(ok, cut, np.iinfo(np.int64).max)))
            if cut[i] < best_balanced[0]:
                x = np.zeros(n, dtype=np.int8)
                x[low] = (index[i] >> np.arange(l)) & 1
                x[high] = xh
                best_balanced = (int(cut[i]), x)

    return {"energies": kept_energy, "partitions": kept_x, "optimum": float(kept_energy[0]),
            "optimal_partitions": ground_count,
            "balanced_cut": best_balanced[0], "balanced_partition": best_balanced[1]}


def _doubling(values):
    """``table[k] = sum_j bit_j(k) values[j]`` for every ``k < 2^len(values)``."""
    table = np.zeros(2 ** len(values), dtype=np.asarray(values).dtype)
    for j, value in enumerate(values):
        table[2 ** j: 2 ** (j + 1)] = table[: 2 ** j] + value
    return table


# --------------------------------------------------------------- heuristics

def fm_pass(x, adjacency, weights, lam):
    """One Fiduccia-Mattheyses pass in the paper's form (their Algorithm 1).

    Gain ``D[v] = sum_u w(v,u) (-1)^[same side]``; the move is the best-gain
    unlocked vertex *on the heavier side*, which is the paper's modification and
    keeps the sequence near balance. Every vertex moves once; the best prefix by
    the paper's objective ``C`` is kept.
    """
    x = x.copy()
    n = len(x)
    sign = 1 - 2 * x
    gain = adjacency @ sign * sign * -1  # sum over cut edges minus uncut edges
    side_weight = weights @ x
    total = weights.sum()
    current = energy(x, adjacency, weights, lam)
    best, best_x = current, x.copy()
    cut = int((adjacency * (x[:, None] ^ x[None, :])).sum() // 2)
    locked = np.zeros(n, bool)
    for _ in range(n):
        heavier = 1 if side_weight > total - side_weight else 0
        candidates = np.where(~locked & (x == heavier))[0]
        if len(candidates) == 0:
            candidates = np.where(~locked)[0]
        v = candidates[np.argmax(gain[candidates])]
        cut -= gain[v]
        side_weight += weights[v] * (1 - 2 * x[v])
        # moving v flips the sign of every edge term at v
        delta = adjacency[v] * np.where(x == x[v], 2, -2)
        gain += delta
        gain[v] = -gain[v]  # v's own gain reverses
        x[v] ^= 1
        locked[v] = True
        current = cut + lam * (side_weight - total / 2) ** 2
        if current < best - 1e-9 * max(1.0, abs(best)):
            best, best_x = current, x.copy()
    return best_x, best


def fm_bisection(adjacency, weights, lam, starts=20, seed=0):
    """Multi-start FM from random balanced splits, passes until no improvement."""
    rng = np.random.default_rng(seed)
    n = len(weights)
    best_e, best_x = np.inf, None
    for _ in range(starts):
        order = rng.permutation(n)
        x = np.zeros(n, dtype=np.int64)
        x[order[np.cumsum(weights[order]) <= weights.sum() / 2]] = 1
        e = energy(x, adjacency, weights, lam)
        while True:
            y, f = fm_pass(x, adjacency, weights, lam)
            if f >= e - 1e-9 * max(1.0, abs(e)):
                break
            x, e = y, f
        if e < best_e:
            best_e, best_x = e, x
    return best_x


def metis_bisection(adjacency, weights, nu=NU, seed=0):
    """METIS 2-way partition with the paper's balance (ufactor = 2 nu x 1000)."""
    import pymetis

    a = sp.csr_matrix(adjacency)
    options = pymetis.Options(ufactor=round(2000 * nu), seed=seed)
    result = pymetis.part_graph(2, adjacency=pymetis.CSRAdjacency(a.indptr, a.indices),
                                vweights=list(map(int, weights)),
                                eweights=list(map(int, a.data)), options=options)
    return np.array(result.vertex_part, dtype=np.int64)


def milp_bisection(adjacency, weights, nu=NU, time_limit=600):
    """Exact minimum cut under the paper's balance constraint, by HiGHS (in SciPy).

    ``min sum w_ij y_ij`` with ``y_ij >= |x_i - x_j|`` and
    ``|sum v_i x_i - V/2| <= nu V``, vertex 0 fixed to side 0. This is the problem
    the penalty in Eq. 3 stands in for; branch and bound proves the optimum
    rather than enumerating it, so it reaches far past 32 vertices.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    n = len(weights)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j]]
    m = len(edges)
    rows = np.repeat(np.arange(2 * m), 3)
    cols, vals = [], []
    for k, (i, j) in enumerate(edges):
        cols += [i, j, n + k, j, i, n + k]
        vals += [1, -1, -1, 1, -1, -1]
    a = sp.csr_matrix((vals, (rows, cols)), shape=(2 * m, n + m))
    a = sp.vstack([a, sp.csr_matrix(np.concatenate([weights, np.zeros(m)]))])
    total = weights.sum()
    lower = np.concatenate([np.full(2 * m, -np.inf), [total / 2 - nu * total]])
    upper = np.concatenate([np.zeros(2 * m), [total / 2 + nu * total]])
    ub = np.ones(n + m)
    ub[0] = 0
    result = milp(np.concatenate([np.zeros(n), [adjacency[e] for e in edges]]),
                  constraints=LinearConstraint(a, lower, upper), bounds=Bounds(0, ub),
                  integrality=np.concatenate([np.ones(n), np.zeros(m)]),
                  options={"time_limit": time_limit})
    if result.status != 0:
        raise RuntimeError(f"MILP did not prove optimality: {result.message}")
    return np.round(result.x[:n]).astype(np.int64)


def spectral_bisection(adjacency, weights):
    """Fiedler vector of the weighted Laplacian, split at the weighted median."""
    a = adjacency.astype(float)
    laplacian = np.diag(a.sum(axis=1)) - a
    fiedler = np.linalg.eigh(laplacian)[1][:, 1]
    order = np.argsort(fiedler)
    x = np.zeros(len(weights), dtype=np.int64)
    x[order[np.cumsum(weights[order]) > weights.sum() / 2]] = 1
    return x


# ------------------------------------------------------------------ VarQITE

def heavy_neighbors_ansatz(adjacency):
    """RZY gate list ``(z_qubit, y_qubit)``, two layers as in the noiseless runs.

    Layer 0: one gate per edge, heaviest first ("the g heaviest edges", all of
    them in simulation). Layer 1: every edge of the induced radius-1 ego graph of
    the vertex whose ego graph is heaviest. The paper fixes neither ``g`` nor
    which qubit carries ``Z``; layer 1 takes the opposite orientation, so both
    appear.
    """
    n = len(adjacency)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j]]
    edges.sort(key=lambda e: -adjacency[e])
    gates = list(edges)
    ego_weight = []
    for s in range(n):
        ego = [s] + [u for u in range(n) if adjacency[s, u]]
        ego_weight.append(adjacency[np.ix_(ego, ego)].sum())
    s = int(np.argmax(ego_weight))
    ego = set([s] + [u for u in range(n) if adjacency[s, u]])
    gates += [(j, i) for i, j in edges if i in ego and j in ego]
    return gates


class RZYCircuit:
    """Statevector of ``prod_k exp(-i theta_k/2 Z_a Y_b) |+>^n``, with its Jacobian.

    ``-i Z Y`` is real, so every amplitude stays real. Qubit ``q`` is bit ``q`` of
    the basis index, as in :func:`all_energies`.
    """

    def __init__(self, n, gates):
        index = np.arange(2 ** n)
        self.n, self.gates = n, gates
        self.partner = [index ^ (1 << y) for _, y in gates]
        self.sign = [np.where((index >> z) & 1, -1.0, 1.0) * np.where((index >> y) & 1, 1.0, -1.0)
                     for z, y in gates]

    def state(self, theta, jacobian=False):
        psi = np.full(2 ** self.n, 2.0 ** (-self.n / 2))
        tangent = np.zeros((2 ** self.n, len(theta))) if jacobian else None
        for k, angle in enumerate(theta):
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            partner, sign = self.partner[k], self.sign[k]
            if jacobian:
                if k:
                    tangent[:, :k] = c * tangent[:, :k] + s * sign[:, None] * tangent[partner, :k]
                tangent[:, k] = 0.5 * (-s * psi + c * sign * psi[partner])
            psi = c * psi + s * sign * psi[partner]
        return (psi, tangent) if jacobian else psi


def all_energies(adjacency, weights, lam):
    """``C(x)`` for every basis state, bit ``q`` of the index being ``x_q``."""
    n = len(weights)
    index = np.arange(2 ** n, dtype=np.int64)
    cut = np.zeros(2 ** n, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                cut += adjacency[i, j] * (((index >> i) ^ (index >> j)) & 1)
    side = _doubling(np.asarray(weights, dtype=np.int64))
    return cut + lam * (side - weights.sum() / 2) ** 2


def pauli_z_table(n):
    """The commuting ``P_a``: every ``Z_i`` and ``Z_i Z_j`` -- the terms of ``C``."""
    index = np.arange(2 ** n)
    z = [1.0 - 2.0 * ((index >> q) & 1) for q in range(n)]
    return np.array(z + [z[i] * z[j] for i in range(n) for j in range(i + 1, n)])


def smallest_balanced_penalty(adjacency, weights, lam):
    """Halve ``lam`` while the QUBO optimum stays ``nu``-balanced.

    The paper leaves lambda unstated; a smaller penalty leaves the cut a larger
    share of the energy scale, which is what VarQITE has to resolve. Choosing it
    with the exact answer in hand favours VarQITE, deliberately.
    """
    n = len(weights)
    while True:
        k = int(np.argmin(all_energies(adjacency, weights, lam / 2)))
        if not balanced((k >> np.arange(n)) & 1, weights):
            return lam
        lam /= 2


def varqite(adjacency, weights, lam, steps=1000, eta=0.1, ridge=1e-4, shots=SHOTS, seed=0,
            target=0.9):
    """The paper's VarQITE, noiseless, with exact ``G`` and ``D``.

    ``G_aj = Re<psi|P_a d_j psi>`` and ``D_a = -<P_a (H - E)>`` (the paper's
    ``-1/2 <{P_a, H - E}>`` for diagonal ``P_a``). ``G theta_dot = D`` is solved
    with a small ridge, then forward Euler with the step capped so no angle moves
    more than ``eta`` radians: the paper gives neither step nor regulariser, and
    a fixed step was unstable here. Energies are scaled by their standard deviation
    in ``|+>^n``. Exact ``G`` and ``D`` are the best case for the algorithm:
    hardware estimates both from shots.

    Records the paper's readout -- the optimum "sampled at any given iteration"
    in ``shots`` shots -- next to the probability that ``shots`` *uniform* samples
    contain it, which is what that criterion is worth at step 0.
    """
    n = len(weights)
    energies = all_energies(adjacency, weights, lam)
    scaled = energies / energies.std()
    optimum = energies.min()
    optimal = energies <= optimum + 1e-9 * max(1.0, optimum)
    paulis = pauli_z_table(n)
    gates = heavy_neighbors_ansatz(adjacency)
    circuit = RZYCircuit(n, gates)
    rng = np.random.default_rng(seed)
    theta = np.zeros(len(gates))
    first_sampled, best_probability = None, 0.0
    trace = []
    for step in range(steps + 1):
        psi, tangent = circuit.state(theta, jacobian=True)
        p = psi ** 2
        p_optimal = float(p[optimal].sum())
        best_probability = max(best_probability, p_optimal)
        mean = float(p @ energies)
        sample = rng.choice(2 ** n, size=shots, p=p / p.sum())
        if first_sampled is None and optimal[sample].any():
            first_sampled = step
        trace.append((step, mean, p_optimal))
        if p_optimal >= target or step == steps:
            break
        g = paulis @ (psi[:, None] * tangent)
        d = -paulis @ (p * (scaled - p @ scaled))
        rate = np.linalg.solve(g.T @ g + ridge * np.eye(len(theta)), g.T @ d)
        theta = theta + eta / max(np.abs(rate).max(), 1e-12) * rate
    uniform = float(optimal.sum()) / 2 ** n
    return {
        "qubits": n, "parameters": len(gates), "steps": step, "lambda": lam,
        "optimum": float(optimum), "final_mean_energy": mean,
        "final_optimal_probability": p_optimal, "best_optimal_probability": best_probability,
        "reached_target": bool(p_optimal >= target),
        "first_step_optimum_sampled": first_sampled,
        "uniform_shots_contain_optimum": 1 - (1 - uniform) ** shots,
        "circuits_per_step": 2 * len(gates) + 1,
        "circuits_total": (2 * len(gates) + 1) * step,
        "trace": trace[:: max(1, len(trace) // 30)] + [trace[-1]],
    }


# --------------------------------------------------------------- the runs

def classical_instance(kind, size, seed, fine):
    adjacency, weights = coarsen(fine, size, seed)
    lam = penalty_weight(adjacency, weights)
    row = {"mesh": kind, "vertices": size, "seed": seed, "fine_vertices": fine.shape[0],
           "coarse_edges": int(np.count_nonzero(np.triu(adjacency))), "lambda": lam}

    started = time.perf_counter()
    exact = exhaustive(adjacency, weights, lam)
    row["exact_seconds"] = time.perf_counter() - started
    optimum = exact["optimum"]
    row.update(optimum=optimum, optimal_partitions=exact["optimal_partitions"],
               ten_lowest=[float(e) for e in exact["energies"]],
               balanced_cut=exact["balanced_cut"],
               optimum_is_balanced=bool(balanced(exact["partitions"][0], weights)))

    methods = {"milp": lambda: milp_bisection(adjacency, weights),
               "fm": lambda: fm_bisection(adjacency, weights, lam, seed=seed),
               "spectral": lambda: spectral_bisection(adjacency, weights)}
    try:
        import pymetis  # noqa: F401
        methods["metis"] = lambda: metis_bisection(adjacency, weights, seed=seed)
    except ImportError:
        pass
    for name, method in methods.items():
        started = time.perf_counter()
        x = method()
        seconds = time.perf_counter() - started
        e = energy(x, adjacency, weights, lam)
        cut = int((adjacency * (x[:, None] ^ x[None, :])).sum() // 2)
        row[name] = {"seconds": seconds, "energy": float(e),
                     "approximation_ratio": float(e / optimum),
                     "optimal": bool(e <= optimum + 1e-9 * max(1.0, optimum)),
                     "cut": cut, "balanced": bool(balanced(x, weights)),
                     "balanced_and_cut_optimal": bool(balanced(x, weights)
                                                      and cut == exact["balanced_cut"])}
    return row


def run_milp_scaling(sizes, seeds=1):
    """Exact balanced bisection past the quantum step's 32 vertices."""
    fines = {kind: mesh(kind) for kind in MESHES}
    rows = []
    for size in sizes:
        for kind in MESHES:
            for seed in range(seeds):
                adjacency, weights = coarsen(fines[kind], size, seed)
                started = time.perf_counter()
                x = milp_bisection(adjacency, weights)
                seconds = time.perf_counter() - started
                cut = int((adjacency * (x[:, None] ^ x[None, :])).sum() // 2)
                rows.append({"mesh": kind, "vertices": size, "seed": seed, "seconds": seconds,
                             "balanced_cut": cut})
                print(f"  n={size:>3} {kind:>12} s{seed}  proved optimal in {seconds:6.2f}s"
                      f"  cut {cut}", flush=True)
    return rows


def run_classical(sizes, seeds):
    fines = {kind: mesh(kind) for kind in MESHES}
    rows = []
    for size in sizes:
        for kind in MESHES:
            for seed in range(seeds):
                row = classical_instance(kind, size, seed, fines[kind])
                rows.append(row)
                arms = "  ".join(f"{m} {row[m]['approximation_ratio']:.4f}"
                                 for m in ("milp", "fm", "metis", "spectral") if m in row)
                print(f"  n={size:>2} {kind:>12} s{seed}  exact {row['exact_seconds']:7.2f}s "
                      f"({row['optimal_partitions']} optimal)  {arms}", flush=True)
    return rows


def run_varqite(sizes, steps):
    fines = {kind: mesh(kind) for kind in MESHES}
    rows = []
    for size in sizes:
        for kind in MESHES:
            adjacency, weights = coarsen(fines[kind], size, 0)
            lam = smallest_balanced_penalty(adjacency, weights,
                                            penalty_weight(adjacency, weights))
            started = time.perf_counter()
            row = varqite(adjacency, weights, lam, steps=steps)
            row.update(mesh=kind, seconds=time.perf_counter() - started)
            rows.append(row)
            print(f"  n={size:>2} {kind:>12}  m={row['parameters']:>3}  steps {row['steps']:>4}  "
                  f"p(opt) {row['final_optimal_probability']:.3f}  first sampled at step "
                  f"{row['first_step_optimum_sampled']} (uniform shots: "
                  f"{row['uniform_shots_contain_optimum']:.2f})  circuits {row['circuits_total']}  "
                  f"({row['seconds']:.0f}s)", flush=True)
    return rows


def summarise(classical, quantum, scaling=()):
    out = {}
    if scaling:
        out["milp_seconds_by_size"] = {}
        for r in scaling:
            seen = out["milp_seconds_by_size"].get(r["vertices"], 0.0)
            out["milp_seconds_by_size"][r["vertices"]] = max(seen, r["seconds"])
    if classical:
        by_size = {}
        for r in classical:
            by_size.setdefault(r["vertices"], []).append(r)
        out["exact_seconds_by_size"] = {n: max(r["exact_seconds"] for r in rs)
                                        for n, rs in sorted(by_size.items())}
        for arm in ("milp", "fm", "metis", "spectral"):
            hits = [r[arm]["optimal"] for r in classical if arm in r]
            if hits:
                out[arm] = {"instances": len(hits), "optimal": int(sum(hits)),
                            "balanced_cut_optimal": sum(r[arm]["balanced_and_cut_optimal"]
                                                        for r in classical if arm in r),
                            "worst_ratio": max(r[arm]["approximation_ratio"]
                                               for r in classical if arm in r),
                            "max_seconds": max(r[arm]["seconds"] for r in classical if arm in r)}
    if quantum:
        out["varqite"] = {"instances": len(quantum),
                          "reached_probability_0.9": sum(r["reached_target"] for r in quantum),
                          "optimum_ever_sampled": sum(r["first_step_optimum_sampled"] is not None
                                                      for r in quantum),
                          "sampled_at_step_0": sum(r["first_step_optimum_sampled"] == 0
                                                   for r in quantum)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--varqite-sizes", default=",".join(map(str, VARQITE_SIZES)))
    ap.add_argument("--milp-sizes", default="64,128,256")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--skip-varqite", action="store_true")
    ap.add_argument("--output", default="exp039_ionq_ansys_partitioning.json")
    args = ap.parse_args()

    print("=== experiment 039 :: IonQ + Ansys coarse bisection, quantum against exact ===")
    classical = quantum = scaling = []
    if not args.skip_classical:
        print("\n-- exact enumeration and classical heuristics --")
        classical = run_classical([int(s) for s in args.sizes.split(",")], args.seeds)
        print("\n-- exact balanced bisection past 32 vertices --")
        scaling = run_milp_scaling([int(s) for s in args.milp_sizes.split(",") if s])
    if not args.skip_varqite:
        print("\n-- VarQITE, noiseless, as specified --")
        quantum = run_varqite([int(s) for s in args.varqite_sizes.split(",") if s], args.steps)
    summary = summarise(classical, quantum, scaling)
    print(json.dumps(summary, indent=2))
    path = RESULTS_DIR / args.output
    with open(path, "w") as fh:
        json.dump({"nu": NU, "keep": KEEP, "shots": SHOTS, "summary": summary,
                   "classical": classical, "milp_scaling": scaling, "varqite": quantum},
                  fh, indent=1, default=_plain)
    print(f"saved -> {path}")
    return 0


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


if __name__ == "__main__":
    raise SystemExit(main())
