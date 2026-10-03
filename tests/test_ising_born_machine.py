"""Ising Born machine against classical generative models (Result 92).

The comparison is only as good as the quantum model's implementation, so the
load-bearing test checks it against Qiskit's own statevector, gate by gate. The
rest pins what makes the comparison fair: every model normalised, and the
parameter counts the matching relies on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.exp037_ising_born_machine import (  # noqa: E402
    Enumeration,
    born_log_probs,
    digits_bits,
    fit,
    fvsbn_log_probs,
    gibbs_log_probs,
    independent_log_probs,
    indices,
    parameter_count,
    rbm_log_probs,
    split,
)


def random_ising(n, seed):
    rng = np.random.default_rng(seed)
    return (torch.tensor(rng.normal(size=n * (n - 1) // 2)), torch.tensor(rng.normal(size=n)),
            torch.tensor(rng.normal(size=n)))


@pytest.mark.parametrize("n, seed", [(3, 0), (4, 1)])
def test_born_probabilities_match_qiskit(n, seed):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    pairs, fields, angles = random_ising(n, seed)
    ours = torch.exp(born_log_probs(Enumeration(n), pairs, fields, angles)).numpy()

    circuit = QuantumCircuit(n)
    qubit = lambda k: n - 1 - k  # noqa: E731 -- our bit 0 is the most significant
    circuit.h(range(n))
    rows, cols = np.triu_indices(n, 1)
    for weight, i, j in zip(pairs.numpy(), rows, cols):
        circuit.rzz(-2 * weight, qubit(i), qubit(j))      # e^{i J z_i z_j}
    for k, weight in enumerate(fields.numpy()):
        circuit.rz(-2 * weight, qubit(k))                 # e^{i b z}
    for k, angle in enumerate(angles.numpy()):
        circuit.ry(angle, qubit(k))
    np.testing.assert_allclose(ours, Statevector(circuit).probabilities(), atol=1e-12)


def test_without_the_final_layer_the_model_is_uniform():
    """The diagonal phases alone represent nothing."""
    pairs, fields, _ = random_ising(5, 2)
    probabilities = torch.exp(born_log_probs(Enumeration(5), pairs, fields, torch.zeros(5)))
    np.testing.assert_allclose(probabilities.numpy(), 1 / 32, atol=1e-14)


def test_every_model_is_normalised():
    n = 5
    enum = Enumeration(n)
    pairs, fields, angles = random_ising(n, 3)
    g = torch.Generator().manual_seed(0)
    draw = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)  # noqa: E731
    for log_probs in (born_log_probs(enum, pairs, fields, angles),
                      gibbs_log_probs(enum, pairs, fields),
                      rbm_log_probs(enum, draw(n), draw(3), draw(n, 3)),
                      fvsbn_log_probs(enum, draw(n * (n - 1) // 2), draw(n)),
                      independent_log_probs(enum, draw(n))):
        assert float(torch.exp(log_probs).sum()) == pytest.approx(1.0, abs=1e-12)


def test_parameter_counts_that_the_matching_relies_on():
    assert parameter_count("Ising Born machine") == 152
    assert parameter_count("RBM, 8 hidden") == 152
    assert parameter_count("Boltzmann machine") == 136
    assert parameter_count("Ising Born machine, IQP") == 136
    assert parameter_count("FVSBN") == 136


def test_indices_agree_with_the_enumeration():
    enum = Enumeration(6)
    bits = enum.bits.numpy().astype(int)
    np.testing.assert_array_equal(indices(bits), np.arange(64))


def test_the_dataset_is_fixed_and_real():
    data = digits_bits()
    assert data.shape == (1797, 16)
    assert set(np.unique(data)) <= {0, 1}
    assert len(np.unique(indices(data))) == 235


def test_split_is_a_partition():
    train, test = split(1797, seed=0)
    assert len(test) == 359 and len(train) == 1438
    assert not set(train) & set(test)


def test_fit_lowers_the_likelihood_and_the_gibbs_fit_is_the_mle():
    """On a tiny problem the convex Gibbs fit must match the empirical moments."""
    n = 4
    enum = Enumeration(n)
    rng = np.random.default_rng(0)
    data_index = rng.integers(0, 2**n, size=200)
    nll, parameters, log_probs = fit("Boltzmann machine", enum, data_index, seed=0)
    probabilities = torch.exp(log_probs(*parameters))
    model_mean = (probabilities[:, None] * enum.spins).sum(0)
    data_mean = enum.spins[torch.tensor(data_index)].mean(0)
    np.testing.assert_allclose(model_mean.numpy(), data_mean.numpy(), atol=1e-5)
    assert nll < n * math.log(2)
