"""The block-encoding trade (Result 76), and the claim that is still extrapolated.

The load-bearing facts are uncomfortable ones and each is pinned here, because
the experiment's headline number (135x at 50 orbitals) is an extrapolation while
every *measured* point says the opposite.

* the double-factorised 1-norm is **larger** than the Pauli one -- DF pays a
  penalty before it earns anything
* at every molecule this repository can build, DF **loses** overall
* the crossover is predicted at N = 9, one orbital past the largest measured
  case, which is what makes it worth testing rather than asserting
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp020_block_encoding import (
    CHEMICAL_ACCURACY,
    double_factorised_cost,
    naive_lcu_cost,
)


def test_lcu_cost_is_linear_in_terms_and_in_one_norm():
    """Both factors enter once; a quadratic here would flatter one side."""
    base = naive_lcu_cost(10.0, 100, CHEMICAL_ACCURACY)["t_gates"]
    assert naive_lcu_cost(20.0, 100, CHEMICAL_ACCURACY)["t_gates"] == pytest.approx(2 * base)
    assert naive_lcu_cost(10.0, 200, CHEMICAL_ACCURACY)["t_gates"] == pytest.approx(2 * base)


def test_double_factorised_walk_cost_is_quadratic_in_orbitals_not_quartic():
    """The whole case for DF: rotations grow as N^2 where LCU terms grow as N^4."""
    costs = np.array([
        double_factorised_cost(10.0, n, 10, CHEMICAL_ACCURACY)["t_per_walk"]
        for n in (10, 20, 40)
    ])
    # doubling N should roughly quadruple the per-walk cost, not 16x it
    assert 3.0 < costs[1] / costs[0] < 4.5
    assert 3.0 < costs[2] / costs[1] < 4.5


def test_the_measured_molecules_all_favour_the_naive_encoding():
    """The honest headline: DF loses everywhere it has actually been measured.

    Guards against the experiment's extrapolated 135x being read as a measured
    result. If a future block encoding makes DF win at these sizes, this test
    should be updated with the new numbers -- not deleted.
    """
    import json
    from pathlib import Path

    path = Path("results/exp020_block_encoding.json")
    if not path.exists():
        pytest.skip("run experiments.exp020_block_encoding first")

    data = json.loads(path.read_text())
    measured = [r for r in data["rows"] if r["orbitals"] <= 8]
    assert measured, "no measured rows to check"

    for row in measured:
        assert row["t_gate_speedup"] < 1.0, (
            f"{row['molecule']}: DF now wins at N={row['orbitals']} "
            f"({row['t_gate_speedup']:.2f}) -- the extrapolation claim needs redoing"
        )
        assert row["one_norm_ratio"] > 1.0, (
            f"{row['molecule']}: DF 1-norm is no longer the larger one"
        )


def test_the_crossover_is_past_every_molecule_measured():
    """Which is exactly why it is a prediction and not a finding."""
    import json
    from pathlib import Path

    path = Path("results/exp020_block_encoding.json")
    if not path.exists():
        pytest.skip("run experiments.exp020_block_encoding first")

    data = json.loads(path.read_text())
    largest = max(r["orbitals"] for r in data["rows"])

    # reconstruct the crossover from the saved fits
    norm_exponent = data["df_one_norm_exponent"]
    factor_exponent = data["df_factor_exponent"]
    rows = data["rows"]
    anchor = np.log(np.array([r["orbitals"] for r in rows], dtype=float))
    norms = np.log(np.array([r["df_one_norm"] for r in rows], dtype=float))
    factors = np.log(np.array([r["df_factors"] for r in rows], dtype=float))
    norm_fit = np.polyfit(anchor, norms, 1)
    factor_fit = np.polyfit(anchor, factors, 1)
    assert norm_fit[0] == pytest.approx(norm_exponent, rel=1e-6)
    assert factor_fit[0] == pytest.approx(factor_exponent, rel=1e-6)

    crossover = None
    for candidate in range(2, 200):
        lam = float(np.exp(np.polyval(norm_fit, np.log(candidate))))
        fac = float(np.exp(np.polyval(factor_fit, np.log(candidate))))
        pauli = 8.47 * (candidate / 4) ** 2.86
        terms = 165 * (candidate / 4) ** 4
        if (naive_lcu_cost(pauli, terms, CHEMICAL_ACCURACY)["t_gates"]
                > double_factorised_cost(lam, candidate, fac, CHEMICAL_ACCURACY)["t_gates"]):
            crossover = candidate
            break

    assert crossover is not None
    assert crossover > largest, (
        f"crossover at N={crossover} is within the measured range (max N={largest}); "
        "it should be reported as measured rather than predicted"
    )


def test_a_bigger_one_norm_always_costs_more_walks():
    """Sanity, and the mechanism behind DF's penalty."""
    small = double_factorised_cost(10.0, 8, 20, CHEMICAL_ACCURACY)["walks"]
    large = double_factorised_cost(40.0, 8, 20, CHEMICAL_ACCURACY)["walks"]
    assert large == pytest.approx(4 * small)
