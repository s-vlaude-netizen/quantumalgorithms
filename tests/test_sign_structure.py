"""What Result 85's thresholds were measuring (Result 88).

The load-bearing claims are structural, so they are tested on matrices whose
answer is known by construction before being trusted on molecules: a balanced
sign graph has zero sign gap and zero annihilation, a frustrated one has
neither, and both are invariant under the diagonal sign gauge that a different
qubit mapping would apply.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from experiments.exp030_stochastic_chemistry import fciqmc, prepare, sector
from experiments.exp033_sign_structure import balanced, sector_matrix, sign_gap
from qres.problems.chemistry import build_molecule


def triangle(signs, diagonal=(0.0, 0.3, 0.7)):
    """Three determinants all coupled to each other, off-diagonal signs given."""
    matrix = np.diag(diagonal)
    for (i, j), sign in zip(((0, 1), (1, 2), (0, 2)), signs):
        matrix[i, j] = matrix[j, i] = 0.2 * sign
    return sparse.csr_matrix(matrix)


@pytest.mark.parametrize("signs, expected", [
    ((-1, -1, -1), True),   # already stoquastic
    ((+1, +1, -1), True),   # stoquastic after flipping determinant 1
    ((+1, +1, +1), False),  # odd number of wrong signs round the cycle
    ((-1, -1, +1), False),
])
def test_balance_on_a_cycle(signs, expected):
    is_balanced, edges, violated = balanced(triangle(signs))
    assert is_balanced is expected
    assert edges == 3 and (violated == 0) is expected


def test_a_frustrated_cycle_in_a_second_component_is_found():
    """The gauge must be grown in every component, not only from determinant 0.

    A single breadth-first tree left the second block's gauge at zero, which
    makes every edge in it look satisfied: it reported H8 balanced beside a
    2.3 Hartree sign gap.
    """
    good = triangle((-1, -1, -1)).toarray()
    bad = triangle((+1, +1, +1)).toarray()
    block = sparse.csr_matrix(np.block([[good, np.zeros((3, 3))],
                                        [np.zeros((3, 3)), bad]]))
    is_balanced, edges, violated = balanced(block)
    assert not is_balanced and edges == 6 and violated == 1


@pytest.mark.parametrize("signs", [(-1, -1, -1), (+1, +1, -1)])
def test_balanced_means_zero_sign_gap(signs):
    assert sign_gap(triangle(signs))[2] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("signs", [(+1, +1, +1), (-1, -1, +1)])
def test_frustrated_means_positive_sign_gap(signs):
    assert sign_gap(triangle(signs))[2] > 1e-3


def test_the_gap_is_gauge_invariant():
    """A different qubit mapping flips determinant signs; it must not matter."""
    rng = np.random.default_rng(0)
    dense = rng.normal(size=(12, 12))
    dense = dense + dense.T
    flip = np.diag(rng.choice([-1.0, 1.0], size=12))
    before = sign_gap(sparse.csr_matrix(dense))[2]
    after = sign_gap(sparse.csr_matrix(flip @ dense @ flip))[2]
    assert after == pytest.approx(before, rel=1e-12)


def test_noise_couplings_do_not_fake_frustration():
    """Integral noise at 1e-20 carries an arbitrary sign; it is not an edge."""
    matrix = triangle((-1, -1, -1)).toarray()
    matrix[0, 2] = matrix[2, 0] = 1e-20
    assert balanced(sparse.csr_matrix(matrix))[0]


@pytest.mark.parametrize("bond", [0.75, 2.0])
def test_h2_is_sign_problem_free(bond):
    block, members = sector_matrix(build_molecule("H2", bond_length=bond))
    assert len(members) == 2
    assert balanced(block)[0]
    assert sign_gap(block)[2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bond", [0.75, 2.0])
def test_h4_is_frustrated_at_both_geometries(bond):
    block, _ = sector_matrix(build_molecule("H4", bond_length=bond))
    assert not balanced(block)[0]
    assert sign_gap(block)[2] > 1e-2


def test_no_annihilation_without_frustration():
    """The consequence the H2 reading rests on, on the code Result 85 ran."""
    problem = build_molecule("H2", bond_length=2.0)
    diagonal, offdiag = prepare(problem.hamiltonian)
    reference = int(problem.hf_bitstring, 2)
    assert len(sector(offdiag, reference)[0]) == 2
    run = fciqmc(diagonal, offdiag, reference, 200, np.random.default_rng(1),
                 steps=2000, equilibration=500)
    assert run["gross_spawns"] > 0
    assert run["annihilation"] == 0.0
