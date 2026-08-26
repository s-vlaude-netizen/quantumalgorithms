"""Exact QAOA energies on graphs far too large to simulate.

Result 51 measured that a QAOA number only means something at ``n >= 60``, since
below ``n = 40`` two independent classical methods are still exactly optimal.
That is also where statevector simulation stops: 2^60 amplitudes do not exist.

The way through is QAOA's own structure.  For a cost Hamiltonian
``C = sum_(u,v) w_uv (1 - Z_u Z_v) / 2``, the depth-``p`` expectation

    <gamma, beta| Z_u Z_v |gamma, beta>

depends **only** on the vertices within graph distance ``p`` of the edge
``(u, v)``.  Every gate outside that light cone either commutes through or
cancels against its inverse.  So the energy of a 100-vertex instance is a sum of
independent small simulations -- on a 3-regular graph the ``p = 1`` cone holds at
most 6 vertices and the ``p = 2`` cone at most 14, whatever ``n`` is.

Two things make it fast enough to optimise against:

* **Isomorphic cones are computed once.**  A 3-regular graph has very few
  distinct local neighbourhoods, so a canonical form of each cone keys a cache.
  On a 100-vertex 3-regular graph that turns 150 subgraph simulations per energy
  into a handful.
* **The cones do not grow with n.**  Cost is ``O(|E|)`` in the graph size and
  exponential only in the cone, which is bounded by degree and depth.

This is exact, not approximate: verified against full statevector simulation on
graphs small enough for both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class LightCone:
    """One edge's neighbourhood, relabelled to a compact qubit range."""

    #: edges of the induced subgraph, as (local_u, local_v, weight)
    edges: tuple[tuple[int, int, float], ...]
    num_qubits: int
    #: the target edge, in local labels
    target: tuple[int, int]
    #: weight of the target edge in the full problem
    weight: float


def _adjacency(edges, num_nodes: int) -> list[set[int]]:
    neighbours: list[set[int]] = [set() for _ in range(num_nodes)]
    for u, v, _ in edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
    return neighbours


def build_light_cone(edges, num_nodes: int, target, depth: int, neighbours=None) -> LightCone:
    """Induced subgraph on everything within ``depth`` hops of ``target``.

    Gates acting outside this set cannot affect ``<Z_u Z_v>``: each layer of the
    QAOA circuit spreads correlations by exactly one hop, so after ``p`` layers
    nothing further away has reached the target edge.
    """
    u, v, weight = target
    if neighbours is None:
        neighbours = _adjacency(edges, num_nodes)

    frontier = {u, v}
    included = {u, v}
    for _ in range(depth):
        frontier = {w for x in frontier for w in neighbours[x]} - included
        if not frontier:
            break
        included |= frontier

    label = {vertex: i for i, vertex in enumerate(sorted(included))}
    sub = tuple(
        sorted(
            (label[a], label[b], w)
            for a, b, w in edges
            if a in included and b in included
        )
    )
    return LightCone(
        edges=sub,
        num_qubits=len(included),
        target=(label[u], label[v]),
        weight=weight,
    )


def _ahu_code(adjacency, root: int, parent: int) -> str:
    """Canonical encoding of the subtree below ``root`` (AHU / tree hashing).

    Two rooted trees have the same code exactly when they are isomorphic, so
    this is a *complete* invariant on the case that dominates here.
    """
    children = sorted(
        _ahu_code(adjacency, child, root) + f":{weight:.12g}"
        for child, weight in adjacency[root]
        if child != parent
    )
    return "(" + "".join(children) + ")"


def _tree_cone_key(cone: LightCone) -> tuple | None:
    """Exact canonical form when the cone is a tree, else ``None``.

    Worth the special case because it is not a special case in practice: a
    random 3-regular graph is locally tree-like, so almost every depth-2
    neighbourhood *is* a tree.  Degree-based refinement cannot see that -- it
    left 1 433 of 1 500 cones looking distinct on a 1 000-vertex graph, so
    essentially every edge paid for its own 14-qubit simulation.
    """
    if len(cone.edges) != cone.num_qubits - 1:
        return None  # has a cycle; the tree encoding does not apply

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(cone.num_qubits)]
    for a, b, w in cone.edges:
        adjacency[a].append((b, w))
        adjacency[b].append((a, w))

    u, v = cone.target
    target_weight = next(
        (w for a, b, w in cone.edges if {a, b} == {u, v}), None
    )
    if target_weight is None:
        return None

    # root at the target edge: the two halves hang off u and v with that edge
    # removed.  <Z_u Z_v> is symmetric in u and v, so the halves are sorted.
    halves = tuple(sorted((
        _ahu_code(adjacency, u, v),
        _ahu_code(adjacency, v, u),
    )))
    return ("tree", cone.num_qubits, halves, round(target_weight, 12))


def _refine_colours(cone: LightCone) -> list[tuple]:
    """Colour refinement (1-WL) on the cone, seeded by the target edge.

    Repeatedly recolour each vertex by its own colour plus the multiset of its
    neighbours' colours and edge weights, until the partition stops changing.
    Vertices in the same graph orbit keep the same colour, so ordering by colour
    puts isomorphic cones into the same labelling -- which the plain degree
    ordering did not, leaving every cyclic cone looking unique.
    """
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(cone.num_qubits)]
    for a, b, w in cone.edges:
        adjacency[a].append((b, round(w, 12)))
        adjacency[b].append((a, round(w, 12)))

    # the two target endpoints start distinguished from everything else, and
    # from each other only as a pair, since <Z_u Z_v> is symmetric in u and v
    colours: list[tuple] = [
        (0,) if vertex in cone.target else (1,) for vertex in range(cone.num_qubits)
    ]

    for _ in range(cone.num_qubits):
        signatures = [
            (colours[vertex], tuple(sorted((colours[n], w) for n, w in adjacency[vertex])))
            for vertex in range(cone.num_qubits)
        ]
        distinct = sorted(set(signatures))
        refined = [(distinct.index(s),) for s in signatures]
        if refined == colours:
            break
        colours = refined
    return colours


def _cone_key(cone: LightCone) -> tuple:
    """Canonical form, so isomorphic cones share one simulation.

    Trees get an exact canonical form (above).  Everything else falls back to
    refinement by degree sequence and neighbour degrees, which is not a complete
    invariant -- but it does not have to be.  The fallback key embeds the full
    *relabelled* edge set, so two cones collide only if some bijection makes
    their edge sets literally identical, which means they are isomorphic. A weak
    ordering therefore costs a missed cache hit, never a wrong energy.
    """
    tree = _tree_cone_key(cone)
    if tree is not None:
        return tree

    colours = _refine_colours(cone)
    order = sorted(range(cone.num_qubits), key=lambda x: (colours[x], x))
    remap = {old: new for new, old in enumerate(order)}
    edges = tuple(sorted(
        (min(remap[a], remap[b]), max(remap[a], remap[b]), round(w, 12))
        for a, b, w in cone.edges
    ))
    target = (min(remap[cone.target[0]], remap[cone.target[1]]),
              max(remap[cone.target[0]], remap[cone.target[1]]))
    return (cone.num_qubits, edges, target)


def _cone_circuit(cone: LightCone, gammas, betas) -> QuantumCircuit:
    """Standard QAOA on the cone: |+>^n, then alternating cost and mixer.

    Kept for verification against Qiskit; the hot path uses ``_evolve`` instead.
    """
    circuit = QuantumCircuit(cone.num_qubits)
    circuit.h(range(cone.num_qubits))
    for gamma, beta in zip(gammas, betas):
        for a, b, w in cone.edges:
            # exp(-i gamma w Z_a Z_b / 2) from the (1 - Z Z)/2 cost convention
            circuit.rzz(gamma * w, a, b)
        circuit.rx(2 * beta, range(cone.num_qubits))
    return circuit


class _ConePlan:
    """Precomputed diagonals for one cone, reused across every evaluation.

    Building a ``QuantumCircuit`` and calling ``Statevector`` per evaluation cost
    21 ms for a 14-qubit cone -- essentially all of it circuit construction and
    gate dispatch, against a state vector of only 16 384 amplitudes.  An
    optimiser makes thousands of these calls, so the overhead *was* the runtime.

    Everything that does not depend on the angles is hoisted here: the cost
    layer is diagonal, so it is one precomputed vector of ``sum w_ab s_a s_b``
    and the layer becomes an elementwise multiply.  The mixer is a 2x2 rotation
    applied per qubit by reshaping.
    """

    __slots__ = ("num_qubits", "dimension", "zz_diagonal", "target_signs")

    def __init__(self, cone: LightCone):
        n = cone.num_qubits
        self.num_qubits = n
        self.dimension = 1 << n

        indices = np.arange(self.dimension, dtype=np.int64)
        # little-endian, matching qiskit: bit q of the index is qubit q
        signs = np.empty((n, self.dimension), dtype=np.int8)
        for qubit in range(n):
            signs[qubit] = 1 - 2 * ((indices >> qubit) & 1)

        diagonal = np.zeros(self.dimension)
        for a, b, w in cone.edges:
            diagonal += w * signs[a] * signs[b]
        self.zz_diagonal = diagonal

        u, v = cone.target
        self.target_signs = (signs[u] * signs[v]).astype(np.float64)


def _evolve(plan: _ConePlan, gammas, betas) -> np.ndarray:
    """QAOA state on the cone, by direct vector operations."""
    n, dimension = plan.num_qubits, plan.dimension
    state = np.full(dimension, 1.0 / np.sqrt(dimension), dtype=np.complex128)

    for gamma, beta in zip(gammas, betas):
        # cost layer: diagonal, so one elementwise phase
        state *= np.exp(-0.5j * gamma * plan.zz_diagonal)

        # mixer layer: rx(2 beta) on every qubit
        cos, sin = np.cos(beta), np.sin(beta)
        for qubit in range(n):
            block = state.reshape(dimension >> (qubit + 1), 2, 1 << qubit)
            lower = block[:, 0, :].copy()
            upper = block[:, 1, :]
            block[:, 0, :] = cos * lower - 1j * sin * upper
            block[:, 1, :] = cos * upper - 1j * sin * lower
    return state


def _zz_expectation(cone: LightCone, gammas, betas, plan: _ConePlan | None = None) -> float:
    plan = plan if plan is not None else _ConePlan(cone)
    state = _evolve(plan, gammas, betas)
    probabilities = state.real**2 + state.imag**2
    return float(probabilities @ plan.target_signs)


class LightConeEnergy:
    """Callable exact QAOA energy for one problem at one depth.

    Holds the cones and the cache, so a full optimisation over ``(gamma, beta)``
    pays the subgraph extraction once.
    """

    #: refuse rather than hang.  Cone size grows like ``d^p``, so on a 3-regular
    #: graph p=1 needs 6 qubits and p=2 needs 14, but p=3 reaches ~30 -- 8 GB of
    #: amplitudes, which does not fail fast, it just stops responding.
    MAX_CONE_QUBITS = 24

    def __init__(self, problem, depth: int, max_cone_qubits: int | None = None):
        edges = [(int(a), int(b), float(w)) for a, b, w in problem.metadata["edges"]]
        n = problem.num_variables
        neighbours = _adjacency(edges, n)

        self.depth = depth
        self.problem = problem
        self.total_weight = sum(w for _, _, w in edges)
        self.cones = [
            build_light_cone(edges, n, edge, depth, neighbours) for edge in edges
        ]
        self.keys = [_cone_key(cone) for cone in self.cones]
        self.max_cone = max((c.num_qubits for c in self.cones), default=0)
        self.distinct_cones = len(set(self.keys))
        self.simulations = 0

        limit = self.MAX_CONE_QUBITS if max_cone_qubits is None else max_cone_qubits
        if self.max_cone > limit:
            raise ValueError(
                f"light cone reaches {self.max_cone} qubits at depth {depth} on "
                f"{problem.name} (limit {limit}). The cone grows as degree^depth, "
                f"independently of n -- lower the depth or the degree."
            )

        # one plan per *distinct* cone, built once and reused for every
        # evaluation: an optimiser calls this thousands of times
        self._plans: dict[tuple, _ConePlan] = {}
        for cone, key in zip(self.cones, self.keys):
            if key not in self._plans:
                self._plans[key] = _ConePlan(cone)

        # collapse the per-edge sum to one term per distinct cone, so a 1 500
        # edge graph does a handful of simulations and a handful of multiplies
        self._weights: dict[tuple, float] = {}
        for cone, key in zip(self.cones, self.keys):
            self._weights[key] = self._weights.get(key, 0.0) + cone.weight

    def expected_cut(self, params) -> float:
        """<C> for the parameters, in cut units (higher is better)."""
        params = np.asarray(params, dtype=float)
        gammas, betas = params[: self.depth], params[self.depth :]

        total = 0.0
        for key, plan in self._plans.items():
            zz = _zz_expectation(None, gammas, betas, plan)
            self.simulations += 1
            total += self._weights[key] * (1 - zz) / 2
        return float(total)

    def __call__(self, params) -> float:
        """Negated, for a minimiser."""
        return -self.expected_cut(params)
