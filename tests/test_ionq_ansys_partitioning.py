"""IonQ + Ansys coarse bisection against exact classical solvers (Result 93).

The claim rests on the exact solvers being exact, so they are checked against
plain enumeration of every partition, and against each other.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments.exp039_ionq_ansys_partitioning import (
    RZYCircuit,
    all_energies,
    balanced,
    coarsen,
    energy,
    exhaustive,
    fm_pass,
    heavy_neighbors_ansatz,
    mesh,
    milp_bisection,
    pauli_z_table,
    penalty_weight,
    qubo_form,
)


def random_graph(n, seed):
    rng = np.random.default_rng(seed)
    a = np.triu(rng.integers(1, 5, (n, n)) * (rng.random((n, n)) < 0.5), 1)
    a = a + a.T
    return a, rng.integers(1, 7, n)


def every_partition(n):
    return np.array(list(itertools.product((0, 1), repeat=n)))


# ----------------------------------------------------------- the objective

def test_energy_is_symmetric_under_swapping_sides():
    a, v = random_graph(9, 0)
    lam = penalty_weight(a, v)
    for x in every_partition(9)[::17]:
        assert energy(x, a, v, lam) == pytest.approx(energy(1 - x, a, v, lam))


def test_qubo_form_is_the_paper_objective():
    a, v = random_graph(8, 1)
    lam = penalty_weight(a, v)
    coupling, h, const = qubo_form(a, v, lam)
    for x in every_partition(8):
        assert const + h @ x + x @ coupling @ x / 2 == pytest.approx(energy(x, a, v, lam))


def test_all_energies_uses_bit_q_for_vertex_q():
    a, v = random_graph(8, 2)
    lam = penalty_weight(a, v)
    table = all_energies(a, v, lam)
    for k in range(0, 256, 7):
        assert table[k] == pytest.approx(energy((k >> np.arange(8)) & 1, a, v, lam))


# ------------------------------------------------------ the exact solvers

@pytest.mark.parametrize("n, seed", [(6, 3), (9, 4), (11, 5)])
@pytest.mark.parametrize("chunked", [False, True])
def test_exhaustive_matches_plain_enumeration(n, seed, chunked):
    a, v = random_graph(n, seed)
    lam = penalty_weight(a, v)
    xs = every_partition(n)
    energies = np.array([energy(x, a, v, lam) for x in xs])
    half = energies[xs[:, 0] == 0]
    result = exhaustive(a, v, lam, low_bits=3 if chunked else n - 1)

    np.testing.assert_allclose(result["energies"], np.sort(half)[:10])
    assert result["optimal_partitions"] == int((half <= half.min() + 1e-9).sum())
    for e, x in zip(result["energies"], result["partitions"]):
        assert energy(x, a, v, lam) == pytest.approx(e)
    cuts = np.array([(a * (x[:, None] ^ x[None, :])).sum() // 2 for x in xs])
    ok = np.array([balanced(x, v) for x in xs])
    assert result["balanced_cut"] == cuts[ok].min()


@pytest.mark.parametrize("n, seed", [(12, 6), (16, 7)])
def test_milp_and_enumeration_agree_on_the_balanced_optimum(n, seed):
    """Two independent exact methods, one enumerating and one proving by bounds."""
    a, v = random_graph(n, seed)
    x = milp_bisection(a, v)
    assert balanced(x, v)
    cut = (a * (x[:, None] ^ x[None, :])).sum() // 2
    assert cut == exhaustive(a, v, penalty_weight(a, v))["balanced_cut"]


def test_fm_pass_reports_the_energy_of_the_partition_it_returns():
    a, v = random_graph(14, 8)
    lam = penalty_weight(a, v)
    x0 = np.random.default_rng(0).integers(0, 2, 14)
    x, e = fm_pass(x0, a, v, lam)
    assert e == pytest.approx(energy(x, a, v, lam))
    assert e <= energy(x0, a, v, lam) + 1e-9


# ------------------------------------------------------------- coarsening

@pytest.mark.parametrize("kind", ["shell", "housing", "unstructured"])
def test_coarsening_hits_the_target_and_preserves_cuts(kind):
    fine = mesh(kind)
    a, v, group = coarsen(fine, 20, seed=0, return_map=True)
    assert a.shape == (20, 20) and len(v) == 20
    assert v.sum() == fine.shape[0]
    np.testing.assert_array_equal(np.bincount(group), v)
    x = np.random.default_rng(1).integers(0, 2, 20)
    xf = x[group]
    rows, cols = fine.nonzero()
    fine_cut = int((xf[rows] != xf[cols]).sum() // 2)
    assert fine_cut == (a * (x[:, None] ^ x[None, :])).sum() // 2


# ---------------------------------------------------------------- VarQITE

def test_rzy_gate_is_exp_of_minus_i_theta_half_z_y():
    from scipy.linalg import expm

    z, y, ident = np.diag([1.0, -1.0]), np.array([[0, -1j], [1j, 0]]), np.eye(2)
    gates, theta = [(0, 1), (2, 0), (1, 2)], np.array([0.3, -1.1, 2.0])
    state = np.full(8, 8 ** -0.5, dtype=complex)
    for (a, b), angle in zip(gates, theta):
        ops = [ident] * 3
        ops[a], ops[b] = z, y
        # qubit q is bit q of the index, so qubit 0 is the rightmost Kronecker factor
        generator = np.kron(np.kron(ops[2], ops[1]), ops[0])
        state = expm(-0.5j * angle * generator) @ state
    np.testing.assert_allclose(RZYCircuit(3, gates).state(theta), state, atol=1e-12)


def test_jacobian_matches_finite_differences():
    gates = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    circuit = RZYCircuit(4, gates)
    theta = np.random.default_rng(3).normal(size=5)
    _, tangent = circuit.state(theta, jacobian=True)
    for j in range(5):
        step = np.zeros(5)
        step[j] = 1e-6
        numeric = (circuit.state(theta + step) - circuit.state(theta - step)) / 2e-6
        np.testing.assert_allclose(tangent[:, j], numeric, atol=1e-8)


def test_g_and_d_are_the_papers_quantities():
    """G_aj = (1/2) d_j <P_a>, and G theta_dot = D reproduces exact imaginary time.

    Exact normalised imaginary-time evolution gives d<P>/dtau = -2 Cov(P, H); with
    d<P>/dtau = 2 G theta_dot that is D = -Cov(P, H) = -1/2 <{P, H - E}>.
    """
    a, v = random_graph(5, 9)
    lam = penalty_weight(a, v)
    h = all_energies(a, v, lam)
    paulis = pauli_z_table(5)
    circuit = RZYCircuit(5, heavy_neighbors_ansatz(a))
    theta = np.random.default_rng(4).normal(size=len(circuit.gates))
    psi, tangent = circuit.state(theta, jacobian=True)
    g = paulis @ (psi[:, None] * tangent)
    for j in range(0, len(theta), 3):
        step = np.zeros(len(theta))
        step[j] = 1e-6
        plus, minus = circuit.state(theta + step) ** 2, circuit.state(theta - step) ** 2
        np.testing.assert_allclose(g[:, j], 0.5 * paulis @ (plus - minus) / 2e-6, atol=1e-7)

    p = psi ** 2
    d = -paulis @ (p * (h - p @ h))

    def moments(tau):
        q = p * np.exp(-2 * tau * (h - h.mean()) / h.std())
        return paulis @ (q / q.sum())

    rate = (moments(1e-6) - moments(-1e-6)) / 2e-6
    np.testing.assert_allclose(rate, 2 * d / h.std(), atol=1e-6)


def test_ansatz_layer_zero_has_one_gate_per_edge_heaviest_first():
    a, _ = random_graph(10, 10)
    gates = heavy_neighbors_ansatz(a)
    edges = int(np.count_nonzero(np.triu(a)))
    layer0 = gates[:edges]
    assert {tuple(sorted(g)) for g in layer0} == {
        (i, j) for i in range(10) for j in range(i + 1, 10) if a[i, j]}
    weights = [a[g] for g in layer0]
    assert weights == sorted(weights, reverse=True)
    assert all(a[g] for g in gates)


def test_metis_returns_a_bisection():
    pytest.importorskip("pymetis")
    from experiments.exp039_ionq_ansys_partitioning import metis_bisection

    a, v = coarsen(mesh("shell"), 24, seed=0)
    x = metis_bisection(a, v)
    assert set(np.unique(x)) <= {0, 1} and 0 < x.sum() < 24
