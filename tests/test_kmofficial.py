"""The translation to Google's kickmix format, and my simulator against Google's.

Skipped when the optional ``kickmix`` package is missing (``pip install kickmix``).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from qres import kmarith as ka
from qres import kmlite as km
from qres import kmofficial as ko

kx = pytest.importorskip("kickmix")


def test_zpow_exponent_is_exact_and_matches_the_official_parser():
    assert Fraction(ko.zpow_exponent(1)) == Fraction(1, 2 ** 63)             # 2^-64 turns
    assert Fraction(ko.zpow_exponent(km.UNIT // 8)) == Fraction(1, 4)        # T gate: exp(i pi / 4)
    assert Fraction(ko.zpow_exponent(km.UNIT // 2)) == 1                     # half turn = Z
    assert Fraction(ko.zpow_exponent(3 * km.UNIT // 4)) == Fraction(3, 2)
    sim = kx.Simulator(batch_size=1, debug_print=False)
    sim.write_across_shots(kx.q(0), 1)
    for units in (1, 12345678901234567, km.UNIT - 1):
        sim.do(kx.Circuit(f"Z_POW q0 {ko.zpow_exponent(units)}"))
    expected = Fraction(1 + 12345678901234567 + km.UNIT - 1, 2 ** 63)       # half turns
    assert sim.read_shot_phase(0) == expected % 2


def test_every_opcode_translates_and_the_counts_agree():
    bld = km.Builder()
    x = bld.register("x", 4)
    t = bld.and_(x[0], x[1])
    bld.ccz(x[0], x[1], x[2])
    bld.cswap(x[3], x[0], x[1])
    bld.swap(x[0], x[1])
    bld.zpow(x[2], Fraction(1, 16))
    bld.zpow(x[3], Fraction(1, 8))
    bld.zpow(x[3], Fraction(1, 4))
    bld.neg()
    bld.unand(t, x[0], x[1])
    (spare,) = bld.alloc(1)
    bld.assert0(spare)
    circuit = bld.finish()
    official = ko.to_official(circuit)
    stats = official.operation_statistics()
    assert official.num_qubits == circuit.n_qubits
    assert stats.num_toffoli_like == bld.toffoli == 3
    assert stats.num_t_like == bld.rotations["t"] == 1
    assert stats.num_non_clifford - stats.num_toffoli_like - stats.num_t_like == bld.rotations["fine"]


def _both(circuit, xs, shots=4, seed=3):
    """Run one circuit on every ``x`` in both simulators; compare everything."""
    official = ko.to_official(circuit)
    reg = circuit.registers["x"]
    n = len(reg)
    xs = np.repeat(np.asarray(xs, dtype=np.int64), shots)
    mine = km.Simulator(circuit, len(xs), seed=seed)
    mine.write(reg, xs)
    mine.run()
    theirs = kx.Simulator(batch_size=len(xs), debug_print=False, seed=seed)
    theirs.use_same_registers_as(official)
    theirs.write_across_shots("x", ((xs[None, :] >> np.arange(n)[:, None]) & 1).astype(bool))
    theirs.do(official)
    assert (np.asarray(theirs.read_across_shots("x")) == ((xs[None, :] >> np.arange(n)[:, None]) & 1)).all()
    scratch = [kx.q(i) for i in range(official.num_qubits) if i not in set(reg)]
    assert not np.asarray(theirs.read_across_shots(scratch)).any()
    assert int(mine.dirty_qubits(reg).sum()) == 0 and int(mine.failed().sum()) == 0
    units = mine.phase_units()
    for i in range(len(xs)):
        half_turns = theirs.read_shot_phase(i)
        official_units = Fraction(half_turns) * 2 ** 63
        assert official_units.denominator == 1 and int(official_units) % km.UNIT == int(units[i])
    return int(units.astype(bool).sum())


def test_adder_lookup_and_unlookup_agree_with_the_official_simulator():
    # an addition, a table lookup erased by measurement-based uncomputation, and a
    # phase from every bit of the sum: exercises HMR, conditioned CZ and Z_POW
    bld = km.Builder()
    x = bld.register("x", 9)
    lo, hi, ctrl = x[:4], x[4:8], x[8]
    out = bld.alloc(5)
    table = [(7 * i * i + 3 * i + 1) % 32 for i in range(16)]
    ka.lookup(bld, lo, table, out)
    ka.cadd(bld, ctrl, hi, lo)
    for p, q in enumerate(hi):
        bld.zpow(q, Fraction(1, 2 ** (p + 4)))
    for p, q in enumerate(out):
        bld.zpow(q, Fraction(1, 2 ** (p + 3)))
    ka.cadd(bld, ctrl, hi, lo, subtract=True)
    ka.unlookup(bld, lo, table, out)
    bld.free(out)
    assert _both(bld.finish(), range(512)) > 0


def test_a_deliberately_broken_circuit_is_caught_by_both():
    bld = km.Builder()
    x = bld.register("x", 3)
    t = bld.and_(x[0], x[1])
    k = bld.hmr(t)                 # forgets the phase correction CZ(x0, x1) if k
    bld.free_bit(k)
    bld.free(t)
    circuit = bld.finish()
    official = ko.to_official(circuit)
    xs = np.repeat(np.array([0, 1, 2, 3, 7]), 64)
    mine = km.Simulator(circuit, len(xs), seed=1)
    mine.write(circuit.registers["x"], xs)
    mine.run()
    theirs = kx.Simulator(batch_size=len(xs), debug_print=False, seed=1)
    theirs.use_same_registers_as(official)
    theirs.write_across_shots("x", ((xs[None, :] >> np.arange(3)[:, None]) & 1).astype(bool))
    theirs.do(official)
    # the two simulators draw different random HMR outcomes, so the flipped shots differ,
    # but both flip about half of the shots where AND(x0, x1) = 1 and none elsewhere
    for flipped in (np.asarray(theirs.read_phase_flipped_across_shots()), mine.phase_units() != 0):
        per_input = np.asarray(flipped).reshape(5, 64).sum(axis=1)
        assert per_input[:3].tolist() == [0, 0, 0]
        assert all(15 <= n <= 49 for n in per_input[3:])
