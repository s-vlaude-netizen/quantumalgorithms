"""Tests for exp050 (numpy-only search) and exp051 (exact certificate) -- first-kind lattices, Result 102."""
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp050_first_kind_search_numpy import Problem, bipartite_rho  # noqa: E402
from experiments.exp051_certify_first_kind import certify, multipartite  # noqa: E402


def test_gradient_matches_finite_differences():
    prob = Problem(5)
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.7, len(prob.pairs))
    f, g = prob.value_grad(x)
    for k in range(len(x)):
        h = 1e-6
        e = np.zeros_like(x); e[k] = h
        num = (prob.value_grad(x + e)[0] - prob.value_grad(x - e)[0]) / (2 * h)
        assert abs(num - g[k]) < 1e-7


def test_an_star_is_critical():
    prob = Problem(6)
    _, g = prob.value_grad(np.zeros(len(prob.pairs)))
    assert np.max(np.abs(g)) < 1e-12


def test_certified_values_of_result_100():
    # K(4,4) at n=7 and K(5,5) at n=9 (Result 100), floating point and exact
    for n, parts, expected in ((7, (4, 4), -0.6193e-3), (9, (5, 5), -1.4879e-3)):
        prob = Problem(n)
        f = prob.value_grad(np.log(bipartite_rho(prob, parts, 1e-9)))[0] - prob.f_an()
        assert abs(f - expected) < 2e-7
        assert abs(float(certify(n, multipartite(n, parts))) - expected) < 2e-7


def test_float_and_exact_agree_on_rational_weights():
    n = 6
    prob = Problem(n)
    rng = np.random.default_rng(3)
    rho = {}
    vec = []
    for (i, j) in prob.pairs:
        w = Fraction(int(rng.integers(1, 9)), int(rng.integers(1, 5)))
        rho[(i, j)] = w; vec.append(float(w))
    f_float = prob.value_grad(np.log(np.array(vec)))[0] - prob.f_an()
    assert abs(f_float - float(certify(n, rho))) < 1e-12


def test_circulant_beats_previous_best_at_n_10():
    """Result 102: weights 1 on the 11-cycle and 4/5 on the difference-4 cycle of Z_11 give log ratio -1.43889e-3,
    below the cone K(1,5,5) (-1.2546e-3) that Result 100 certified as the best at n = 10."""
    n = 10
    rho = {}
    for i in range(11):
        for d, w in ((1, Fraction(1)), (4, Fraction(4, 5))):
            j = (i + d) % 11
            rho[(min(i, j), max(i, j))] = w
    v = float(certify(n, rho))
    assert abs(v - (-1.43889e-3)) < 2e-8
    cone = float(certify(n, multipartite(n, (1, 5, 5), apex_weight=Fraction(21, 40))))
    assert v < cone - 1.5e-4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
