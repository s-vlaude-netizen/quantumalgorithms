"""Frank-Kasper foams for the Kelvin problem (exp043).

The structures are generated from space-group generators and must reproduce
the Wyckoff multiplicities and the Frank-Kasper face census; one cheap
relaxation pins the pipeline to its recorded value.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp043_kelvin_search import (
    STRUCTURES,
    build,
    census,
    space_group,
    structure_foam,
)

GROUP_ORDERS = {"A15": 48, "sigma": 16, "Z": 24, "mu": 36, "P": 8, "delta": 4,
                "C15": 192, "C14": 24, "C36": 24, "chi": 48}

# faces per cell of the Voronoi foam of each structure (Frank-Kasper: Z12,
# Z14, Z15, Z16), and the primitive-cell size
CENSUS = {
    "A15": {12: 2, 14: 6},
    "sigma": {12: 10, 14: 16, 15: 4},
    "Z": {12: 3, 14: 2, 15: 2},
    "mu": {12: 7, 14: 2, 15: 2, 16: 2},
    "P": {12: 24, 14: 20, 15: 8, 16: 4},
    "delta": {12: 24, 14: 20, 15: 8, 16: 4},
    "C15": {12: 4, 16: 2},
    "C14": {12: 8, 16: 4},
    "C36": {12: 16, 16: 8},
}


@pytest.mark.parametrize("name", sorted(GROUP_ORDERS))
def test_space_groups_have_their_orders(name):
    # operations modulo the primitive lattice, centring translations included
    assert len(space_group(STRUCTURES[name]["generators"])) == GROUP_ORDERS[name]


@pytest.mark.parametrize("name", sorted(CENSUS))
def test_frank_kasper_census(name):
    foam = structure_foam(name)
    c = census(foam)
    assert c["faces_per_cell"] == CENSUS[name]
    assert set(c["vertex_cells"]) == {4}                    # Plateau: four cells per vertex
    assert {len(face["polygon"]) for face in foam["faces"]} == {5, 6}
    box, points = build(name)
    assert abs(np.linalg.det(box)) == pytest.approx(len(points))     # unit volume per cell


def test_alpha_manganese_is_not_frank_kasper():
    foam = structure_foam("chi")
    sizes = {len(face["polygon"]) for face in foam["faces"]}
    assert 4 in sizes and set(census(foam)["vertex_cells"]) == {4}


def test_p_phase_needs_the_papers_cell_order():
    spec = dict(STRUCTURES["P"])
    try:
        STRUCTURES["P"] = dict(spec, cell=(16.983, 4.752, 9.07, 90, 90, 90))   # COD's order
        faces = census(structure_foam("P"))["faces_per_cell"]
    finally:
        STRUCTURES["P"] = spec
    assert max(faces) > 16                   # no Frank-Kasper cell has more than 16 faces


def test_z_phase_relaxation_is_pinned():
    from experiments.exp042_kelvin_problem import relax, triangulate

    # the free-box optimiser stalls at volume errors of ~1e-6, and the stopping
    # point depends on the order of the cells by a few 1e-6 in cost (measured:
    # 5.3070766 vs 5.3070808 for two orderings), so pin to that scatter
    result = relax(triangulate(structure_foam("Z"), 1), free_box=True)
    assert result["volume_error"] < 1e-5
    assert result["cost"] == pytest.approx(5.307079, abs=1e-5)
