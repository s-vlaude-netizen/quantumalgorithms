"""kmlite: the simulator, the adders, lookups and shifts, each checked exhaustively
or against a dense state-vector simulation that branches over measurement outcomes."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qres import kmarith as ka
from qres import kmlite as km

# ------------------------------------------------------------- dense reference

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def apply_hmr(psi, a, nqubits):
    """Hadamard then Z-measurement then reset on qubit ``a``: [(p, state, outcome)]."""
    index = np.arange(1 << nqubits)
    flipped = psi[index ^ (1 << a)]
    after_h = np.where((index >> a) & 1, flipped - psi, psi + flipped) / np.sqrt(2)
    out = []
    for outcome in (0, 1):
        keep = ((index >> a) & 1) == outcome
        branch = np.where(keep, after_h, 0)
        p = float(np.vdot(branch, branch).real)
        if p < 1e-12:
            continue
        branch = branch / np.sqrt(p)
        if outcome:                                    # reset |1> -> |0>
            branch = branch[index ^ (1 << a)]
        out.append((p, branch, outcome))
    return out


def run_dense(circuit: km.Circuit, psi0: np.ndarray, nqubits: int):
    index = np.arange(1 << nqubits)
    branches = [(1.0, psi0.astype(complex), {})]
    for i in range(len(circuit)):
        op, a, b, c, k = (int(circuit.op[i]), int(circuit.qa[i]), int(circuit.qb[i]),
                          int(circuit.qc[i]), int(circuit.cond[i]))
        nxt = []
        for prob, psi, bits in branches:
            if k >= 0 and not bits.get(k, 0):
                nxt.append((prob, psi, bits))
                continue
            bit = lambda q: (index >> q) & 1
            if op == km.HMR:
                for p, branch, outcome in apply_hmr(psi, a, nqubits):
                    nxt.append((prob * p, branch, {**bits, c: outcome}))
                continue
            if op == km.X:
                psi = psi[index ^ (1 << a)]
            elif op == km.CX:
                psi = psi[np.where(bit(a) == 1, index ^ (1 << b), index)]
            elif op == km.CCX:
                psi = psi[np.where((bit(a) & bit(b)) == 1, index ^ (1 << c), index)]
            elif op == km.SWAP:
                psi = psi[(index & ~((1 << a) | (1 << b))) | (bit(a) << b) | (bit(b) << a)]
            elif op == km.CSWAP:
                swapped = (index & ~((1 << b) | (1 << c))) | (bit(b) << c) | (bit(c) << b)
                psi = psi[np.where(bit(a) == 1, swapped, index)]
            elif op == km.Z:
                psi = psi * (1 - 2 * bit(a))
            elif op == km.CZ:
                psi = psi * (1 - 2 * (bit(a) & bit(b)))
            elif op == km.CCZ:
                psi = psi * (1 - 2 * (bit(a) & bit(b) & bit(c)))
            elif op == km.NEG:
                psi = -psi
            elif op == km.BIT_SET0:
                bits = {**bits, a: 0}
            elif op == km.BIT_SET1:
                bits = {**bits, a: 1}
            elif op == km.BIT_INV:
                bits = {**bits, a: bits.get(a, 0) ^ 1}
            else:
                raise AssertionError(op)
            nxt.append((prob, psi, bits))
        branches = nxt
    return branches


def basis_superposition(rng, nqubits, data_qubits):
    """Random state on ``data_qubits`` with every other qubit |0>."""
    psi = np.zeros(1 << nqubits, dtype=complex)
    for value in range(1 << len(data_qubits)):
        index = sum(((value >> j) & 1) << q for j, q in enumerate(data_qubits))
        psi[index] = rng.normal() + 1j * rng.normal()
    return psi / np.linalg.norm(psi)


def test_and_with_measurement_uncompute_is_exact_on_both_outcomes():
    b = km.Builder()
    x = b.register("x", 2)
    t = b.and_(x[0], x[1])
    b.z(t)                                    # use the AND value so the phase matters
    b.unand(t, x[0], x[1])
    c = b.finish()
    psi = basis_superposition(np.random.default_rng(0), b.n_qubits, x)
    expected = psi * np.array([1 - 2 * ((i & 1) & ((i >> 1) & 1)) for i in range(len(psi))])
    branches = run_dense(c, psi, b.n_qubits)
    assert len(branches) == 2
    for prob, state, _ in branches:
        assert prob == pytest.approx(0.5)
        np.testing.assert_allclose(state, expected, atol=1e-12)
    # an AND erased WITHOUT the fix-up must be caught by the classical model
    bad = km.Builder()
    xb = bad.register("x", 2)
    tb = bad.and_(xb[0], xb[1])
    bad.hmr(tb)
    sim = km.Simulator(bad.finish(), 256, seed=3)
    sim.write(xb, np.arange(256) % 4)
    sim.run()
    assert sim.sign_flipped().any()


@pytest.mark.parametrize("n, m", [(2, 2), (3, 2), (3, 3)])
def test_adder_matches_dense_simulation_on_superpositions(n, m):
    b = km.Builder()
    tgt, src = b.register("t", n), b.register("s", m)
    ka.add(b, tgt, src)
    c = b.finish()
    rng = np.random.default_rng(n * 10 + m)
    psi = basis_superposition(rng, b.n_qubits, tgt + src)
    ideal = np.zeros_like(psi)
    for value in range(1 << (n + m)):
        bv, av = value & ((1 << n) - 1), value >> n
        out = ((bv + av) % (1 << n)) | (av << n)
        ideal[out] = psi[value]
    for prob, state, _ in run_dense(c, psi, b.n_qubits):
        assert prob > 0
        np.testing.assert_allclose(state, ideal, atol=1e-10)


def test_lookup_and_unlookup_match_dense_simulation():
    table = [3, 1, 2, 0]
    b = km.Builder()
    addr, out = b.register("a", 2), b.register("o", 2)
    ka.lookup(b, addr, table, out)
    ka.unlookup(b, addr, table, out)
    c = b.finish()
    psi = basis_superposition(np.random.default_rng(5), b.n_qubits, addr)
    branches = run_dense(c, psi, b.n_qubits)
    assert len(branches) >= 4
    for _, state, _ in branches:
        np.testing.assert_allclose(state, psi, atol=1e-10)


# ------------------------------------------------------------ classical model

def simulate(builder, regs, values, seed=0):
    c = builder.finish()
    shots = len(next(iter(values.values())))
    sim = km.Simulator(c, shots, seed=seed)
    for name, v in values.items():
        sim.write(regs[name], v)
    sim.run()
    return c, sim


@pytest.mark.parametrize("n, m", [(1, 1), (2, 2), (3, 3), (4, 4), (5, 3), (6, 2), (5, 5)])
def test_adder_and_subtractor_exhaustively(n, m):
    for subtract in (False, True):
        b = km.Builder()
        tgt, src = b.register("t", n), b.register("s", m)
        (ka.sub if subtract else ka.add)(b, tgt, src)
        grid = np.array(list(itertools.product(range(1 << n), range(1 << m))), dtype=np.uint64)
        c, sim = simulate(b, {"t": tgt, "s": src}, {"t": grid[:, 0], "s": grid[:, 1]}, seed=n)
        sign = -1 if subtract else 1
        np.testing.assert_array_equal(sim.read(tgt), (grid[:, 0].astype(np.int64)
                                      + sign * grid[:, 1].astype(np.int64)) % (1 << n))
        np.testing.assert_array_equal(sim.read(src), grid[:, 1])
        assert not sim.dirty_qubits(tgt + src).any() and not sim.sign_flipped().any()
        assert c.stats["toffoli"] == n - 1


def test_controlled_add_and_add_const_exhaustively():
    n, m = 4, 3
    b = km.Builder()
    ctrl, tgt, src = b.register("c", 1), b.register("t", n), b.register("s", m)
    ka.cadd(b, ctrl[0], tgt, src)
    grid = np.array(list(itertools.product(range(2), range(1 << n), range(1 << m))), dtype=np.uint64)
    c, sim = simulate(b, {"c": ctrl, "t": tgt, "s": src},
                      {"c": grid[:, 0], "t": grid[:, 1], "s": grid[:, 2]})
    np.testing.assert_array_equal(sim.read(tgt), (grid[:, 1] + grid[:, 0] * grid[:, 2]) % (1 << n))
    assert not sim.dirty_qubits(ctrl + tgt + src).any() and not sim.sign_flipped().any()
    assert c.stats["toffoli"] == m + n - 1
    for k in (0, 1, 5, 9, 17):
        b = km.Builder()
        tgt = b.register("t", 5)
        ka.add_const(b, tgt, k)
        vals = np.arange(32, dtype=np.uint64)
        _, sim = simulate(b, {"t": tgt}, {"t": vals})
        np.testing.assert_array_equal(sim.read(tgt), (vals + k) % 32)
        assert not sim.dirty_qubits(tgt).any() and not sim.sign_flipped().any()


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
def test_lookup_then_unlookup(k):
    rng = np.random.default_rng(k)
    width = 7
    table = [int(v) for v in rng.integers(0, 1 << width, size=1 << k)]
    b = km.Builder()
    addr, out = b.register("a", k), b.register("o", width)
    ka.lookup(b, addr, table, out)
    forward = b.toffoli
    assert forward == (1 << k) - 2
    mid = b.finish()
    shots = 1 << min(k + 3, 12)
    values = np.arange(shots, dtype=np.uint64) % (1 << k)
    sim = km.Simulator(mid, shots, seed=1)
    sim.write(addr, values)
    sim.run()
    np.testing.assert_array_equal(sim.read(out), np.array(table, dtype=np.uint64)[values])
    assert not sim.dirty_qubits(addr + out).any()
    # now the measurement-based erase, from a fresh builder
    b = km.Builder()
    addr, out = b.register("a", k), b.register("o", width)
    ka.lookup(b, addr, table, out)
    ka.unlookup(b, addr, table, out)
    k_lo = max(1, k // 2)
    expected = (1 << k) - 2 + (1 << k_lo) - 2 + (1 << (k - k_lo)) - 2 + (1 << k_lo) - 2
    assert b.toffoli == expected
    for seed in range(4):
        sim = km.Simulator(b.finish(), shots, seed=seed)
        sim.write(addr, values)
        sim.run()
        assert not sim.dirty_qubits(addr).any()
        assert not sim.sign_flipped().any(), "unlookup left phase kickback"
        np.testing.assert_array_equal(sim.read(addr), values)


def test_unlookup_without_fixup_is_detected():
    k, table = 4, list(range(16))
    b = km.Builder()
    addr, out = b.register("a", k), b.register("o", 4)
    ka.lookup(b, addr, table, out)
    for q in out:
        b.hmr(q)
    sim = km.Simulator(b.finish(), 1024, seed=2)
    sim.write(addr, np.arange(1024) % 16)
    sim.run()
    assert sim.sign_flipped().any()


@pytest.mark.parametrize("n, shift", [(5, 1), (6, 2), (6, 3), (7, 3), (8, 4), (9, 6)])
def test_controlled_rotate(n, shift):
    b = km.Builder()
    ctrl, reg = b.register("c", 1), b.register("r", n)
    ka.controlled_rotate(b, ctrl[0], reg, shift)
    values = np.arange(1 << n, dtype=np.uint64)
    grid_c = np.concatenate([np.zeros(1 << n, dtype=np.uint64), np.ones(1 << n, dtype=np.uint64)])
    grid_r = np.concatenate([values, values])
    c, sim = simulate(b, {"c": ctrl, "r": reg}, {"c": grid_c, "r": grid_r})
    mask = (1 << n) - 1
    rotated = ((grid_r << np.uint64(shift)) | (grid_r >> np.uint64(n - shift))) & np.uint64(mask)
    np.testing.assert_array_equal(sim.read(reg), np.where(grid_c == 1, rotated, grid_r))
    from math import gcd
    assert c.stats["toffoli"] == n - gcd(n, shift)


def test_zpow_accumulates_exact_phases_and_costs():
    b = km.Builder()
    q = b.register("q", 2)
    b.zpow(q[0], km.Fraction(1, 8))             # T
    b.zpow(q[0], km.Fraction(1, 4))             # S (Clifford)
    b.zpow(q[1], km.Fraction(1, 2 ** 20))        # a fine rotation
    b.neg()
    c = b.finish()
    assert b.rotations == {"clifford": 1, "t": 1, "fine": 1}
    sim = km.Simulator(c, 4)
    sim.write(q, np.arange(4))
    sim.run()
    units = sim.phase_units()
    half = 1 << 63
    expect = [half, half + (1 << 61) + (1 << 62), half + (1 << 44), half + (1 << 61) + (1 << 62) + (1 << 44)]
    assert [int(u) for u in units] == [e % (1 << 64) for e in expect]
