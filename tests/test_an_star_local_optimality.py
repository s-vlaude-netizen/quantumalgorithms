"""A_n* as a local minimum of the isoperimetric quotient among lattices (exp047).

The quotient formula is checked against two independent computations (the zonotope
determinant and cofactor formulas, and a numerical Voronoi cell), the symmetry
reduction against the full sum over subsets, and then the sign pattern in n.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from experiments import exp047_an_star_local_optimality as e
from experiments.exp044_hales_song_certificate import (
    graphical_generators,
    zonotope_measures,
)
from experiments.exp045_four_dimensional_cells import voronoi_cell


def random_symmetric(n, rng, scale):
    t = rng.normal(scale=scale, size=(n, n))
    t = (t + t.T) / 2
    return t - np.trace(t) / n * np.eye(n)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_formula_matches_the_zonotope_determinant_and_cofactor_sums(n):
    rng = np.random.default_rng(n)
    cell = e.Cell(n)
    for _ in range(2):
        x = rng.normal(scale=0.3, size=cell.m)
        t = random_symmetric(n, rng, 0.15)
        brute = zonotope_measures(graphical_generators(n, np.exp(x), expm(-t)))[0]
        assert cell.quotient(x, t) == pytest.approx(brute, rel=1e-12)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_lattice_slice_is_the_voronoi_cell_of_the_lattice(n):
    # the lattice exp(T) A_n*, its Voronoi cell computed numerically from the lattice
    rng = np.random.default_rng(10 + n)
    cell = e.Cell(n)
    t = random_symmetric(n, rng, 0.05)
    basis = (expm(t) @ cell.f[1:].T).T
    assert cell.lattice_quotient(t) == pytest.approx(voronoi_cell(basis)[0], rel=1e-11)


def test_the_base_point_is_the_permutohedron():
    for n in (3, 4, 6):
        cell = e.Cell(n)
        assert cell.quotient(np.zeros(cell.m), np.zeros((n, n))) == pytest.approx(
            e.permutohedron_iso(n), rel=1e-12)


@pytest.mark.parametrize("n", [6, 14])
def test_symmetry_reduced_sum_equals_the_full_sum_over_subsets(n):
    full = e.block_curvatures(n)
    for kind in e.PIECES:
        reduced = e.ReducedCell(n, kind).curvatures(1e-3)
        for key in ("lattice", "xx", "TT", "xT", "schur"):
            # two double-precision finite differences with h = 1e-3: rounding noise ~1e-6
            assert float(reduced[key]) == pytest.approx(full[kind][key], rel=1e-4, abs=5e-6)


def test_a4_star_is_a_critical_point_with_the_two_predicted_eigenvalues():
    # full Hessian over all symmetric traceless T at A_4*: eigenvalues of the standard
    # piece (x4) and the (3, 2) piece (x5), no gradient
    n = 4
    cell = e.Cell(n)
    rows = []
    for i in range(n):
        for j in range(i, n):
            m = np.zeros((n, n))
            m[i, j] = m[j, i] = 1.0
            rows.append(m.ravel())
    flat = np.array(rows)
    trace = np.eye(n).ravel() / np.sqrt(n)
    flat = flat - np.outer(flat @ trace, trace)
    _, s, vt = np.linalg.svd(flat, full_matrices=False)
    ortho = vt[s > 1e-9].reshape(-1, n, n)
    d, h = len(ortho), 2e-3

    def f(vec):
        return cell.lattice_quotient(np.tensordot(vec, ortho, axes=1))

    eye = np.eye(d)
    gradient = [(f(h * eye[i]) - f(-h * eye[i])) / (2 * h) for i in range(d)]
    hessian = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            a, b = eye[i] * h, eye[j] * h
            hessian[i, j] = hessian[j, i] = (f(a + b) - f(a - b) - f(-a + b) + f(-a - b)) / (4 * h * h)
    assert np.abs(gradient).max() < 1e-4
    eigenvalues = np.linalg.eigvalsh(hessian)
    expected = e.block_curvatures(n)
    assert eigenvalues[:5] == pytest.approx([expected["(n-1,2)"]["lattice"]] * 5, rel=1e-3)
    assert eigenvalues[5:] == pytest.approx([expected["standard"]["lattice"]] * 4, rel=1e-3)


def test_the_lattice_curvature_is_positive_up_to_n_14_and_negative_from_n_15():
    for n in range(3, 15):
        for kind in e.PIECES:
            assert e.ReducedCell(n, kind).curvatures(1e-3)["lattice"] > 0
    for n in range(15, 31):
        standard = e.ReducedCell(n, e.PIECES[0]).curvatures(1e-3)["lattice"]
        rectangle = e.ReducedCell(n, e.PIECES[1]).curvatures(1e-3)["lattice"]
        assert standard > 0 > rectangle


def test_the_affine_class_loses_its_minimum_one_dimension_earlier():
    assert e.ReducedCell(13, e.PIECES[1]).curvatures(1e-3)["schur"] > 0
    assert e.ReducedCell(14, e.PIECES[1]).curvatures(1e-3)["schur"] < 0


def test_high_precision_curvatures_around_the_sign_change():
    # independent of the step and of the precision to many digits
    fourteen = e.reduced_lattice_curvature(14, e.PIECES[1], h="1e-8", dps=30)
    fifteen = e.reduced_lattice_curvature(15, e.PIECES[1], h="1e-8", dps=30)
    assert float(fourteen) == pytest.approx(0.0675181203517822, abs=1e-9)
    assert float(fifteen) == pytest.approx(-0.0013516260188360, abs=1e-9)


def test_an_explicit_lattice_beats_a_15_star_in_the_full_subset_sum():
    # not the symmetric reduction: the full sum over all 2^15 bipartitions
    n = 15
    cell = e.Cell(n)
    direction = cell.matrix_of(cell.edge_vectors()["(n-1,2)"])
    direction /= np.linalg.norm(direction)
    eps = 0.02
    assert (np.exp(cell.lattice_weights(eps * direction)) > 0).all()       # first kind
    assert cell.lattice_quotient(eps * direction) < cell.lattice_quotient(0 * direction) - 4e-7


def test_witnesses_are_first_kind_and_lower_than_the_regular_permutohedron():
    for item in e.witnesses(plan=((15, (0.02,)), (16, (0.05,)))):
        assert item["change"] < 0
        assert item["min_selling_over_mean"] > 0.3


# ---------------------------------------------- the Selling-parameter form and its certificate

def test_selling_formula_matches_the_frame_pipeline():
    rng = np.random.default_rng(4)
    for n in (4, 6):
        cell = e.Cell(n)
        ones = np.ones((n + 1, n + 1)) - np.eye(n + 1)
        base = e.selling_value(n, ones)
        for _ in range(2):
            t = random_symmetric(n, rng, 0.04)
            rho = -(cell.f @ expm(2 * t) @ cell.f.T)
            np.fill_diagonal(rho, 0)
            assert (rho[~np.eye(n + 1, dtype=bool)] > 0).all()
            assert e.selling_value(n, rho) / base == pytest.approx(
                cell.lattice_quotient(t) / cell.lattice_quotient(0 * t), rel=1e-12)


def test_cut_weights_and_tree_counts_obey_the_volume_identity():
    # sum over bipartitions of tau(S) tau(S^c) r(S) = n d: the cone-volume formula
    n = 9
    rng = np.random.default_rng(2)
    rho = rng.uniform(0.5, 1.5, size=(n + 1, n + 1))
    rho = np.triu(rho, 1)
    rho = rho + rho.T
    cell = e.Cell(n)
    total = 0.0
    for inside, outside, _ in cell.batches:
        gamma = cell._tau(rho, inside) * cell._tau(rho, outside)
        cut = np.array([rho[np.ix_(a, b)].sum() for a, b in zip(inside, outside)])
        total += float((gamma * cut).sum())
    d = cell._tau(rho, np.arange(n + 1)[None, :])[0]
    assert total / 2 == pytest.approx(n * d, rel=1e-10)


@pytest.mark.parametrize("n, delta", [(8, "1/20"), (15, "1/20")])
def test_closed_form_class_sums_equal_the_generic_exact_sums(n, delta):
    from fractions import Fraction

    slow = e.exact_terms(n, e.rectangle_selling(n, Fraction(delta)))
    fast = e.exact_terms_fast(n, Fraction(delta))
    assert slow[0] == fast[0]
    assert sorted(slow[1]) == sorted(fast[1])


def test_exact_certificates_exist_from_n_15_and_not_up_to_n_14():
    from fractions import Fraction

    for n in (15, 16, 20, 40, 100):
        assert e.certificate(n, Fraction(1, 2000))["smaller"]
    for n in (4, 8, 14):
        for delta in ("1/2000", "1/50", "1/10"):
            assert not e.certificate(n, Fraction(delta))["smaller"]


def test_certificate_agrees_with_the_floating_point_quotient():
    from fractions import Fraction

    n = 15
    for delta, expected_smaller in (("1/20", True), ("1/2", False)):
        rho = np.array(e.rectangle_selling(n, Fraction(delta)), dtype=float)
        np.fill_diagonal(rho, 0)
        ones = np.ones((n + 1, n + 1)) - np.eye(n + 1)
        assert (e.selling_value(n, rho) < e.selling_value(n, ones)) == expected_smaller
        assert e.certificate(n, Fraction(delta))["smaller"] == expected_smaller


def test_the_hales_song_bound_itself_loses_its_minimum_at_n_10():
    from experiments.exp044_hales_song_certificate import phi_psi_general

    def bound(n, x):
        phi, psi = phi_psi_general(n, np.exp(x))
        return 2 * n * (phi ** 3 / psi) ** (1 / (2 * n))

    def smallest_piece(n):
        w = e.Cell(n).edge_vectors()["(n-1,2)"]
        w = w / np.linalg.norm(w)
        h = 2e-3
        return (bound(n, h * w) - 2 * bound(n, 0 * w) + bound(n, -h * w)) / h ** 2

    assert smallest_piece(9) == pytest.approx(0.000189, abs=2e-5)
    assert smallest_piece(10) < 0
