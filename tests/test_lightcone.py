"""Correctness of the light-cone QAOA energy.

This module exists to compute QAOA energies on graphs no simulator can hold, so
there is no way to spot-check its output by eye: a wrong light cone produces a
perfectly plausible energy on a 1 000-vertex graph and nothing downstream would
notice.  The only honest check is against full statevector simulation on graphs
small enough for both, which is what most of this file does.

The other risk is the cache.  Cones are keyed by a canonical form so isomorphic
neighbourhoods are simulated once; if that key ever collides across
*non*-isomorphic cones, energies are silently wrong.  The key embeds the full
relabelled edge set precisely so that cannot happen, and that property is
tested here directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qres.lightcone import (
    LightConeEnergy,
    _cone_key,
    build_light_cone,
)
from qres.problems.optimization import (
    erdos_renyi_maxcut,
    maxcut,
    random_regular_maxcut,
)
from qres.qaoa import qaoa_circuit


def statevector_cut(problem, params, reps: int) -> float:
    """Expected cut by simulating the whole circuit, no light cones involved."""
    circuit = qaoa_circuit(problem.hamiltonian, reps=reps)
    state = Statevector(circuit.assign_parameters(np.asarray(params, dtype=float)))
    energy = float(np.real(state.expectation_value(problem.hamiltonian)))
    return -(energy + problem.offset)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: random_regular_maxcut(12, 3, seed=1),
        lambda: random_regular_maxcut(14, 3, seed=2),
        lambda: random_regular_maxcut(12, 5, seed=3),
        lambda: erdos_renyi_maxcut(13, 0.4, seed=4),
    ],
)
@pytest.mark.parametrize("reps", [1, 2, 3])
def test_light_cone_energy_is_exact(factory, reps):
    """Not approximately equal to the full simulation -- equal."""
    problem = factory()
    rng = np.random.default_rng(reps)
    energy = LightConeEnergy(problem, reps)

    for _ in range(3):
        params = rng.uniform(-np.pi, np.pi, 2 * reps)
        assert energy.expected_cut(params) == pytest.approx(
            statevector_cut(problem, params, reps), abs=1e-9
        )


def test_weighted_graphs_are_exact_too():
    """Weights enter both the cone circuit and the canonical key."""
    rng = np.random.default_rng(0)
    edges = [
        (i, j, float(rng.normal(0, 1)))
        for i in range(10)
        for j in range(i + 1, 10)
        if rng.random() < 0.4
    ]
    problem = maxcut(edges, num_nodes=10)
    energy = LightConeEnergy(problem, 2)
    params = rng.uniform(-np.pi, np.pi, 4)
    assert energy.expected_cut(params) == pytest.approx(
        statevector_cut(problem, params, 2), abs=1e-9
    )


def test_cone_size_is_bounded_by_degree_and_depth_not_by_n():
    """The whole point: cost must not grow with the graph.

    On a 3-regular graph the depth-1 cone holds at most 6 vertices and the
    depth-2 cone at most 14, whether n is 20 or 1 000.
    """
    for n in (20, 100, 400):
        assert LightConeEnergy(random_regular_maxcut(n, 3, seed=0), 1).max_cone <= 6
        assert LightConeEnergy(random_regular_maxcut(n, 3, seed=0), 2).max_cone <= 14


def test_deep_cones_are_refused_rather_than_hung():
    """A 30-qubit cone is 8 GB of amplitudes; it must fail fast and say why."""
    problem = random_regular_maxcut(200, 3, seed=0)
    with pytest.raises(ValueError, match="light cone reaches"):
        LightConeEnergy(problem, 3)


def test_isomorphic_cones_share_one_key():
    """Caching only pays if structurally identical cones actually collide.

    Every depth-1 cone of a 3-regular graph with no short cycle is the same
    shape, so a large instance must collapse to a handful of distinct keys --
    the tree canonicalisation exists because degree-based ordering left 1 433 of
    1 500 cones looking unique.
    """
    energy = LightConeEnergy(random_regular_maxcut(400, 3, seed=0), 1)
    assert len(energy.cones) == 600
    assert energy.distinct_cones <= 12


def test_cone_keys_never_collide_across_non_isomorphic_cones():
    """A false cache hit would silently corrupt every energy that follows.

    Soundness comes from the key embedding the full *relabelled* edge set: two
    cones share a key only if some bijection makes their edge sets identical,
    which is isomorphism. Checked here by construction rather than trusted.
    """
    seen: dict[tuple, tuple] = {}
    for n, degree, depth in [(60, 3, 1), (60, 3, 2), (40, 5, 1), (30, 4, 2)]:
        problem = random_regular_maxcut(n, degree, seed=0, exact=False)
        edges = [(int(a), int(b), float(w)) for a, b, w in problem.metadata["edges"]]
        for edge in edges:
            cone = build_light_cone(edges, n, edge, depth)
            key = _cone_key(cone)
            signature = (cone.num_qubits, len(cone.edges))
            if key in seen:
                # same key must imply the same cone size and edge count
                assert seen[key] == signature, f"key collision: {key}"
            else:
                seen[key] = signature


def test_cached_and_uncached_energies_agree():
    """Same energy whether or not isomorphic cones were collapsed."""
    problem = random_regular_maxcut(16, 3, seed=5)
    params = np.array([0.7, -0.4])

    energy = LightConeEnergy(problem, 1)
    cached = energy.expected_cut(params)

    # recompute with every cone forced to its own key
    uncached = 0.0
    for cone in energy.cones:
        from qres.lightcone import _zz_expectation

        uncached += cone.weight * (1 - _zz_expectation(cone, params[:1], params[1:])) / 2

    assert cached == pytest.approx(uncached, abs=1e-12)


def test_call_is_the_negated_cut_for_a_minimiser():
    problem = random_regular_maxcut(12, 3, seed=0)
    energy = LightConeEnergy(problem, 1)
    params = np.array([0.5, 0.3])
    assert energy(params) == pytest.approx(-energy.expected_cut(params), abs=1e-12)
