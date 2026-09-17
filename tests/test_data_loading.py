"""Data loading cost (Result 84), the barrier the whole ML answer rests on.

The first version of this experiment measured sparsity with qiskit's dense
`StatePreparation` and would have concluded that sparsity buys nothing -- a fact
about the library, not the problem. That failure is pinned here, because it is
the one that would have turned an honest "not settled" into a false negative.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp028_data_loading import compressed_loading_cost, loading_cost


def test_loading_a_general_vector_is_linear_in_its_dimension():
    """The barrier itself: gates track d, so loading costs what reading costs."""
    ratios = {}
    for qubits in (6, 7, 8):
        dimension = 2**qubits
        rng = np.random.default_rng(qubits)
        ratios[dimension] = loading_cost(rng.normal(size=dimension)) / dimension

    # and the ratio approaches one from below rather than drifting
    assert all(0.5 < r < 1.2 for r in ratios.values()), ratios
    assert ratios[256] > ratios[64], "gates/d should rise towards 1, not fall"


def test_a_basis_state_is_free():
    """|0...0> needs no entangling gates; a nonzero count would mean the
    transpiler is being charged for something that is not loading."""
    vector = np.zeros(16)
    vector[0] = 1.0
    assert loading_cost(vector) == 0


def test_a_product_state_needs_no_entanglement():
    """|+>^n factorises, so a correct count charges zero two-qubit gates."""
    vector = np.ones(16) / 4.0
    assert loading_cost(vector) == 0


def test_the_sparse_arm_actually_depends_on_the_nonzeros():
    """The bug that was caught: this must not return the dense cost every time.

    Qiskit's StatePreparation never inspects the zeros, so passing it a sparse
    vector returns the dense gate count. If this assertion ever collapses to a
    constant, the loophole row is measuring the library again.
    """
    rng = np.random.default_rng(0)
    counts = [compressed_loading_cost(s, rng) for s in (32, 128, 512)]
    assert counts[0] < counts[1] < counts[2], counts
    # roughly proportional to the nonzeros, not to any fixed dimension
    assert counts[2] / counts[0] > 4


def test_a_dense_vector_with_zeros_is_not_automatically_cheap():
    """Documents the library's actual behaviour, so the distinction stays visible.

    A sparse vector handed to the dense routine costs the dense price. This is
    why `compressed_loading_cost` exists as a separate function rather than the
    experiment simply zeroing entries and re-measuring.
    """
    rng = np.random.default_rng(1)
    dimension = 256
    sparse = np.zeros(dimension)
    positions = rng.choice(dimension, size=8, replace=False)
    sparse[positions] = rng.normal(size=8)

    dense = loading_cost(rng.normal(size=dimension))
    assert loading_cost(sparse) > dense / 2, (
        "the dense routine appears to exploit sparsity now; if so the experiment's "
        "second arm should be re-derived rather than this test relaxed"
    )


def test_the_recorded_run_extrapolates_with_the_ratio_not_the_exponent():
    """Using the fitted slope on a 70B model would inflate it on curvature alone."""
    import json
    from pathlib import Path

    path = Path("results/exp028_data_loading.json")
    if not path.exists():
        pytest.skip("run experiments.exp028_data_loading first")
    data = json.loads(path.read_text())

    ratio = data["two_qubit_gates"][-1] / data["dimensions"][-1]
    for row in data["implications"]:
        assert row["predicted_gates"] == pytest.approx(
            row["dimension"] * ratio, rel=1e-9
        )
    # the asymptotic exponent must be close to 1, or "linear" is the wrong word
    assert 0.95 < data["exponent"] < 1.15, data["exponent"]
