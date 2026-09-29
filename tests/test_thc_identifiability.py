"""Where the THC gauge stops being the only flat direction (Result 89).

The proof has two load-bearing parts and the tests pin both: that the closed-form
Jacobian the sweep uses is the model's Jacobian (checked against SymPy's own
differentiation, entry by entry), and that the modular rank the sweep reports is
the rational rank where that is cheap enough to compute.
"""

from __future__ import annotations

import pytest
import sympy as sp

from experiments.exp034_thc_identifiability import (
    certified_rank,
    exact_jacobian,
    gauge_generators,
    generic_count,
    identifiable_up_to,
    integer_point,
    one_norm_along_flat_directions,
    one_norm_gradient,
    predicted_rank,
    symbolic_jacobian,
)


@pytest.mark.parametrize("orbitals, rank", [(2, 3), (3, 2), (3, 4)])
def test_closed_form_jacobian_is_the_models_jacobian(orbitals, rank):
    """The sweep's Jacobian, against SymPy differentiating the model itself."""
    symbolic, parameters = symbolic_jacobian(orbitals, rank)
    chi, coupling = integer_point(orbitals, rank, seed=5)
    values = [v for row in chi for v in row] + [v for row in coupling for v in row]
    evaluated = symbolic.subs(dict(zip(parameters, values)))
    assert (evaluated - sp.Matrix(exact_jacobian(chi, coupling))).is_zero_matrix


@pytest.mark.parametrize("orbitals, rank", [(2, 3), (3, 4)])
def test_one_norm_gradient_matches_symbolic_differentiation(orbitals, rank):
    chi, coupling = integer_point(orbitals, rank, seed=5)
    symbols = [sp.Symbol(f"c{p}_{m}") for p in range(orbitals) for m in range(rank)]
    symbols += [sp.Symbol(f"z{m}_{n}") for m in range(rank) for n in range(rank)]
    weight = [sum(sp.Symbol(f"c{p}_{m}")**2 for p in range(orbitals)) for m in range(rank)]
    one_norm = sp.Rational(1, 2) * sum(
        sp.sign(coupling[m][n]) * sp.Symbol(f"z{m}_{n}") * weight[m] * weight[n]
        for m in range(rank) for n in range(rank))
    values = [v for row in chi for v in row] + [v for row in coupling for v in row]
    point = dict(zip(symbols, values))
    assert [sp.diff(one_norm, s).subs(point) for s in symbols] == \
        one_norm_gradient(chi, coupling)


@pytest.mark.parametrize("orbitals, rank", [(3, 4), (3, 5), (4, 3)])
def test_modular_rank_is_the_rational_rank(orbitals, rank):
    """A lower bound in general; here it is checked to be the exact value."""
    rows = exact_jacobian(*integer_point(orbitals, rank, seed=1))
    assert certified_rank(rows) == certified_rank(rows, domain=sp.QQ)


def test_gauge_generators_are_in_the_kernel_exactly():
    chi, coupling = integer_point(4, 5, seed=2)
    jacobian = sp.Matrix(exact_jacobian(chi, coupling))
    assert (jacobian * sp.Matrix(gauge_generators(chi, coupling)).T).is_zero_matrix


@pytest.mark.parametrize("orbitals, rank", [
    (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 4), (3, 5), (3, 6), (3, 7),
    (4, 7), (4, 8), (4, 9), (4, 10),
])
def test_rank_formula(orbitals, rank):
    """Every regime: gauge only, sliding columns, saturation."""
    row = generic_count(orbitals, rank)
    assert row["jacobian_rank"] == predicted_rank(orbitals, rank)
    assert row["gauge_only"] == (rank <= identifiable_up_to(orbitals))


def test_the_rank_is_not_a_property_of_the_point():
    """A different integer point gives the same generic count."""
    assert generic_count(4, 8, seed=1)["jacobian_rank"] == \
        generic_count(4, 8, seed=7)["jacobian_rank"] == 80


def test_boundary_values():
    assert [identifiable_up_to(n) for n in (2, 4, 6)] == [2, 7, 16]


def test_lambda_is_constant_on_the_gauge_but_not_on_the_slides():
    """The consequence: past the boundary lambda moves along exact minimisers."""
    inside = one_norm_along_flat_directions(4, 7)
    assert inside["flat_directions"] == 7
    assert inside["lambda_constant_along_gauge"]
    assert inside["lambda_constant_along_all_flat_directions"]

    past = one_norm_along_flat_directions(4, 8)
    assert past["flat_directions"] == 16
    assert past["gauge_in_kernel"] and past["lambda_constant_along_gauge"]
    assert not past["lambda_constant_along_all_flat_directions"]
