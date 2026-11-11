"""Voronoi cells of the classical lattices, computed numerically (exp049).

Known values pin the routine; the volume of the computed cell must equal the lattice's
covolume, which is what shows the candidate vectors included every relevant one.
"""

from __future__ import annotations

import pytest

from experiments import exp047_an_star_local_optimality as e
from experiments import exp049_known_lattices_voronoi as k


@pytest.fixture(scope="module")
def table():
    return k.lattices()


@pytest.mark.parametrize("name, expected", [
    ("A3*", 5.314739699972),            # BCC: the truncated octahedron, 3 (1 + 2 sqrt 3) / 4^(2/3)
    ("D3", 3 * 2 ** (5 / 6)),           # FCC: the rhombic dodecahedron
    ("Z3", 6.0),
    ("D4", 2 ** (11 / 4)),              # the 24-cell
    ("Z4", 8.0),
])
def test_three_and_four_dimensional_cells_match_their_exact_values(table, name, expected):
    quotient, volume_ratio, _ = k.voronoi_quotient(table[name])
    assert volume_ratio == pytest.approx(1.0, abs=1e-9)
    assert quotient == pytest.approx(expected, rel=1e-9)


def test_dual_root_lattice_is_the_permutohedron(table):
    quotient, volume_ratio, _ = k.voronoi_quotient(table["A4*"])
    assert volume_ratio == pytest.approx(1.0, abs=1e-9)
    assert quotient == pytest.approx(float(e.permutohedron_iso(4)), rel=1e-9)


def test_e7_cell_is_much_rounder_than_any_first_kind_lattice(table):
    quotient, volume_ratio, _ = k.voronoi_quotient(table["E7"])
    assert volume_ratio == pytest.approx(1.0, abs=1e-9)
    assert quotient == pytest.approx(10.401960, abs=2e-6)
    assert quotient < float(e.permutohedron_iso(7)) - 0.7          # A_7* is 11.1346
