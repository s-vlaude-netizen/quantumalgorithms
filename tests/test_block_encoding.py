"""The block-encoding trade (Result 76), and the crossover that was measured.

Double factorisation pays a *larger* 1-norm than a Pauli LCU -- 3.6-10x -- and
buys cheaper walks in exchange. Whether that trade pays depends on size, so the
load-bearing claim is a **crossover**, and each of its parts is pinned here:

* the 1-norm penalty is real and DF loses at small N (H2 through H8, 0.01-0.93)
* the penalty **shrinks** with size, because the Pauli norm grows as N^2.59
  against DF's N^1.93 -- without this the curves never cross
* the model put the crossover at N = 9, and **H10 was added to test it rather
  than quote it**: measured gain 1.56, so the prediction was confirmed by
  observation

The last test guards the distinction that matters most: if a future change
pushes the crossover past the largest molecule measured, the headline silently
reverts to extrapolation, and the test should fail rather than let it.
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


def test_a_bigger_one_norm_always_costs_more_walks():
    """Sanity, and the mechanism behind DF's penalty."""
    small = double_factorised_cost(10.0, 8, 20, CHEMICAL_ACCURACY)["walks"]
    large = double_factorised_cost(40.0, 8, 20, CHEMICAL_ACCURACY)["walks"]
    assert large == pytest.approx(4 * small)


def test_the_crossover_is_observed_rather_than_predicted():
    """H10 was added to turn the load-bearing claim into a measurement.

    The model put the crossover at N = 9. H10 (N = 10) is the first molecule
    past it, and it wins -- so the prediction was tested rather than quoted. If
    a future change moves the crossover back outside the measured range, this
    test should fail loudly: the headline would go back to being extrapolation.
    """
    import json
    from pathlib import Path

    path = Path("results/exp020_block_encoding.json")
    if not path.exists():
        pytest.skip("run experiments.exp020_block_encoding first")

    data = json.loads(path.read_text())
    rows = data["rows"]
    largest = max(r["orbitals"] for r in rows)
    crossover = data.get("crossover_orbitals")

    assert crossover is not None, "the experiment must report a crossover"
    assert crossover <= largest, (
        f"crossover N={crossover} is beyond the largest measured N={largest}; "
        "the claim has become extrapolation again"
    )

    # and the measurement must actually agree with the model at that point
    beyond = [r for r in rows if r["orbitals"] >= crossover]
    assert beyond, "no measured molecule at or past the crossover"
    assert all(r["t_gate_speedup"] > 1.0 for r in beyond), (
        "a molecule past the predicted crossover does not favour DF: "
        + ", ".join(f"{r['molecule']}={r['t_gate_speedup']:.2f}" for r in beyond)
    )

    below = [r for r in rows if r["orbitals"] < crossover]
    assert all(r["t_gate_speedup"] <= 1.0 for r in below), (
        "a molecule below the predicted crossover already favours DF: "
        + ", ".join(f"{r['molecule']}={r['t_gate_speedup']:.2f}" for r in below)
    )


def test_the_one_norm_penalty_shrinks_with_size():
    """The mechanism that makes a crossover exist at all.

    DF pays a larger 1-norm, but the Pauli norm grows faster (N^2.59 against
    N^1.93 measured), so the penalty is transient. Without this the two curves
    would never cross and the whole approach would be a constant-factor loss.
    """
    import json
    from pathlib import Path

    path = Path("results/exp020_block_encoding.json")
    if not path.exists():
        pytest.skip("run experiments.exp020_block_encoding first")

    data = json.loads(path.read_text())
    exponents = data["chain_exponents"]
    assert exponents["pauli_one_norm"] > exponents["df_one_norm"], (
        "the DF 1-norm penalty no longer shrinks with size"
    )

    chain = sorted(
        (r for r in data["rows"] if r["molecule"].startswith("H") and r["molecule"][1:].isdigit()),
        key=lambda r: r["orbitals"],
    )
    assert chain[0]["one_norm_ratio"] > chain[-1]["one_norm_ratio"], (
        "the measured ratio should fall along the series"
    )
