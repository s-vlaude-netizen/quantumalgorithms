"""kmlite -- a small kickmix-style circuit builder and bit-sliced simulator.

Google Quantum AI's *kickmix* circuits (Khattar et al., arXiv:2610.23832, Sec. II)
use only reversible classical gates, diagonal phase gates and measurement-based
uncomputation. Every instruction maps a computational basis state to one basis
state times a phase, so a classical computer simulates a circuit exactly on any
basis input, one bit per qubit and one phase per shot. This module implements
that gate set from the published instruction documentation, independently of
Google's package (which the sandbox this was written in would not install).

Gate set: ``X CX CCX SWAP CSWAP`` (reversible), ``Z CZ CCZ Z_POW NEG`` (phases),
``HMR R`` (X-basis demolition measurement and reset), and classical bit ops. Any
instruction can be conditioned on a classical bit. ``ASSERT0`` is a debugging
instruction that is not part of kickmix and costs nothing: it records a failure
for every shot in which the qubit is 1.

The simulator packs 64 shots into each ``uint64`` word, so a Toffoli on 2^15
shots is 512 word operations. ``Z_POW`` phases accumulate per shot in units of
2^-64 turns.

Costs follow the paper's convention: a Toffoli-like gate is ``CCX``, ``CCZ`` or
``CSWAP`` (one Toffoli each); ``Z_POW`` by a multiple of a quarter turn is
Clifford, by an odd multiple of an eighth turn is one T gate (a quarter
Toffoli), and any finer angle is a rotation priced separately.
"""

from __future__ import annotations

from array import array
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction

import numba as nb
import numpy as np

(X, CX, CCX, SWAP, CSWAP, Z, CZ, CCZ, ZPOW, NEG, HMR, R, ASSERT0,
 BIT_INV, BIT_SET0, BIT_SET1) = range(16)
OP_NAMES = ["X", "CX", "CCX", "SWAP", "CSWAP", "Z", "CZ", "CCZ", "Z_POW", "NEG", "HMR", "R",
            "ASSERT0", "BIT_INVERT", "BIT_STORE0", "BIT_STORE1"]
TOFFOLI_LIKE = (CCX, CCZ, CSWAP)
UNIT = 1 << 64                      # one full turn in Z_POW accumulator units


def turns_to_units(turns) -> int:
    """A phase of ``turns`` full turns as an integer number of 2^-64 turns."""
    if isinstance(turns, int):
        return (turns * UNIT) % UNIT
    return int(Fraction(turns) * UNIT) % UNIT


def rotation_class(units: int) -> str:
    """'clifford' (multiple of 1/4 turn), 't' (odd multiple of 1/8), or 'fine'."""
    if units % (UNIT // 4) == 0:
        return "clifford"
    if units % (UNIT // 8) == 0:
        return "t"
    return "fine"


@dataclass
class Circuit:
    """An immutable instruction list. Arrays share one length."""

    op: np.ndarray
    qa: np.ndarray
    qb: np.ndarray
    qc: np.ndarray
    cond: np.ndarray
    ang: np.ndarray
    n_qubits: int
    n_bits: int
    registers: dict
    stats: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.op)

    @property
    def has_zpow(self) -> bool:
        return bool((self.op == ZPOW).any())

    def counts(self) -> dict:
        names = {}
        for code, name in enumerate(OP_NAMES):
            n = int((self.op == code).sum())
            if n:
                names[name] = n
        return names


class Builder:
    """Records instructions, recycles qubits, and counts Toffolis and rotations."""

    def __init__(self, track_sections: bool = True):
        self._op, self._qa, self._qb = array("b"), array("i"), array("i")
        self._qc, self._cond, self._ang = array("i"), array("i"), array("Q")
        self.n_qubits = 0
        self.n_bits = 0
        self._free_qubits: list[int] = []
        self._free_bits: list[int] = []
        self.registers: dict[str, list[int]] = {}
        self.toffoli = 0
        self.rotations = {"clifford": 0, "t": 0, "fine": 0}
        self.live = 0
        self.peak_live = 0
        self._track = track_sections
        self._stack: list[str] = []
        self.section_costs: dict[tuple, int] = {}

    # ------------------------------------------------------------ allocation
    def alloc(self, n: int = 1) -> list[int]:
        out = []
        for _ in range(n):
            if self._free_qubits:
                out.append(self._free_qubits.pop())
            else:
                out.append(self.n_qubits)
                self.n_qubits += 1
        self.live += n
        self.peak_live = max(self.peak_live, self.live)
        return out

    def free(self, qubits) -> None:
        """Return qubits (which must be |0>) to the pool."""
        qubits = [qubits] if isinstance(qubits, int) else list(qubits)
        self._free_qubits.extend(reversed(qubits))
        self.live -= len(qubits)

    def register(self, name: str, n: int) -> list[int]:
        qs = self.alloc(n)
        self.registers[name] = qs
        return qs

    def alloc_bit(self) -> int:
        if self._free_bits:
            return self._free_bits.pop()
        self.n_bits += 1
        return self.n_bits - 1

    def free_bit(self, bit: int) -> None:
        self._free_bits.append(bit)

    # -------------------------------------------------------------- sections
    @contextmanager
    def section(self, name: str):
        self._stack.append(name)
        try:
            yield
        finally:
            self._stack.pop()

    def _emit(self, op, a=0, b=0, c=0, cond=None, ang=0):
        self._op.append(op)
        self._qa.append(a)
        self._qb.append(b)
        self._qc.append(c)
        self._cond.append(-1 if cond is None else cond)
        self._ang.append(ang)
        if op in TOFFOLI_LIKE:
            self.toffoli += 1
            if self._track:
                key = tuple(self._stack)
                self.section_costs[key] = self.section_costs.get(key, 0) + 1

    # ----------------------------------------------------------------- gates
    def x(self, q, cond=None):
        self._emit(X, q, cond=cond)

    def cx(self, c, t, cond=None):
        self._emit(CX, c, t, cond=cond)

    def ccx(self, a, b, t, cond=None):
        self._emit(CCX, a, b, t, cond)

    def swap(self, a, b, cond=None):
        self._emit(SWAP, a, b, cond=cond)

    def cswap(self, c, a, b, cond=None):
        self._emit(CSWAP, c, a, b, cond)

    def z(self, q, cond=None):
        self._emit(Z, q, cond=cond)

    def cz(self, a, b, cond=None):
        self._emit(CZ, a, b, cond=cond)

    def ccz(self, a, b, c, cond=None):
        self._emit(CCZ, a, b, c, cond)

    def neg(self, cond=None):
        self._emit(NEG, cond=cond)

    def zpow(self, q, turns, cond=None):
        """Phase |1> of ``q`` by ``turns`` full turns (Fraction, int or 2^-64 units
        wrapped as ``('units', n)``)."""
        units = turns[1] if isinstance(turns, tuple) else turns_to_units(turns)
        self.rotations[rotation_class(units)] += 1
        self._emit(ZPOW, q, cond=cond, ang=units)

    def hmr(self, q) -> int:
        """X-basis measure-and-reset; returns the classical bit holding the outcome."""
        bit = self.alloc_bit()
        self._emit(HMR, q, 0, bit)
        return bit

    def reset(self, q):
        self._emit(R, q)

    def assert0(self, q):
        self._emit(ASSERT0, q)

    def bit_xor(self, target_bit, source_bit):
        """target_bit ^= source_bit (classical, free)."""
        self._emit(BIT_INV, target_bit, cond=source_bit)

    def bit_flip(self, target_bit):
        self._emit(BIT_INV, target_bit)

    # ------------------------------------------------- measurement-based AND
    def and_(self, a, b) -> int:
        """Fresh qubit holding a AND b (one Toffoli)."""
        (t,) = self.alloc(1)
        self.ccx(a, b, t)
        return t

    def unand(self, t, a, b) -> None:
        """Erase t = a AND b by X-basis measurement and a classically controlled CZ
        (no Toffoli) and free it. ``a`` and ``b`` must still hold their values."""
        k = self.hmr(t)
        self.cz(a, b, cond=k)
        self.free_bit(k)
        self.free(t)

    # ------------------------------------------------------------------ finish
    def finish(self) -> Circuit:
        return Circuit(
            op=np.frombuffer(self._op, dtype=np.int8).copy(),
            qa=np.frombuffer(self._qa, dtype=np.int32).copy(),
            qb=np.frombuffer(self._qb, dtype=np.int32).copy(),
            qc=np.frombuffer(self._qc, dtype=np.int32).copy(),
            cond=np.frombuffer(self._cond, dtype=np.int32).copy(),
            ang=np.frombuffer(self._ang, dtype=np.uint64).copy(),
            n_qubits=self.n_qubits, n_bits=self.n_bits,
            registers={k: list(v) for k, v in self.registers.items()},
            stats={"toffoli": self.toffoli, "rotations": dict(self.rotations),
                   "section_costs": dict(self.section_costs)})


# ====================================================================== simulator

@nb.njit(cache=True)
def _run(op, qa, qb, qc, cond, ang, S, Cb, P, ACC, FAIL, ones, rng):
    nops = op.shape[0]
    W = S.shape[1]
    state = rng[0]
    inc = np.uint64(0x9E3779B97F4A7C15)
    m1 = np.uint64(0xBF58476D1CE4E5B9)
    m2 = np.uint64(0x94D049BB133111EB)
    s30, s27, s31 = np.uint64(30), np.uint64(27), np.uint64(31)
    one = np.uint64(1)
    for i in range(nops):
        o = op[i]
        a = qa[i]
        b = qb[i]
        c = qc[i]
        k = cond[i]
        if k >= 0:
            m = Cb[k]
        else:
            m = ones
        if o == 0:                                   # X
            for w in range(W):
                S[a, w] ^= m[w]
        elif o == 1:                                 # CX a -> b
            for w in range(W):
                S[b, w] ^= S[a, w] & m[w]
        elif o == 2:                                 # CCX a, b -> c
            for w in range(W):
                S[c, w] ^= S[a, w] & S[b, w] & m[w]
        elif o == 3:                                 # SWAP a, b
            for w in range(W):
                d = (S[a, w] ^ S[b, w]) & m[w]
                S[a, w] ^= d
                S[b, w] ^= d
        elif o == 4:                                 # CSWAP a ? b, c
            for w in range(W):
                d = (S[b, w] ^ S[c, w]) & S[a, w] & m[w]
                S[b, w] ^= d
                S[c, w] ^= d
        elif o == 5:                                 # Z a
            for w in range(W):
                P[w] ^= S[a, w] & m[w]
        elif o == 6:                                 # CZ a, b
            for w in range(W):
                P[w] ^= S[a, w] & S[b, w] & m[w]
        elif o == 7:                                 # CCZ a, b, c
            for w in range(W):
                P[w] ^= S[a, w] & S[b, w] & S[c, w] & m[w]
        elif o == 8:                                 # Z_POW a by ang
            theta = ang[i]
            for w in range(W):
                bits = S[a, w] & m[w]
                if bits != 0:
                    for j in range(64):
                        if (bits >> np.uint64(j)) & one:
                            ACC[w * 64 + j] += theta
        elif o == 9:                                 # NEG
            for w in range(W):
                P[w] ^= m[w]
        elif o == 10 or o == 11:                     # HMR a -> bit c   /   R a
            for w in range(W):
                state += inc
                z = state
                z = (z ^ (z >> s30)) * m1
                z = (z ^ (z >> s27)) * m2
                r = z ^ (z >> s31)
                v = S[a, w]
                P[w] ^= r & v & m[w]
                if o == 10:
                    Cb[c, w] = (Cb[c, w] & ~m[w]) | (r & m[w])
                S[a, w] = v & ~m[w]
        elif o == 12:                                # ASSERT0 a
            for w in range(W):
                FAIL[w] |= S[a, w] & m[w]
        elif o == 13:                                # BIT_INVERT a (if cond)
            for w in range(W):
                Cb[a, w] ^= m[w]
        elif o == 14:                                # BIT_STORE0 a
            for w in range(W):
                Cb[a, w] &= ~m[w]
        elif o == 15:                                # BIT_STORE1 a
            for w in range(W):
                Cb[a, w] |= m[w]
    rng[0] = state


def pack_bits(bits: np.ndarray, words: int) -> np.ndarray:
    """(nshots,) array of 0/1 -> (words,) uint64, shot s in bit s % 64 of word s // 64."""
    padded = np.zeros(words * 64, dtype=np.uint8)
    padded[:len(bits)] = bits
    return np.packbits(padded, bitorder="little").view(np.uint64)


def unpack_bits(row: np.ndarray, nshots: int) -> np.ndarray:
    return np.unpackbits(row.view(np.uint8), bitorder="little")[:nshots]


class Simulator:
    """Run a circuit on ``nshots`` classical inputs at once."""

    def __init__(self, circuit: Circuit, nshots: int, seed: int = 0):
        self.circuit = circuit
        self.nshots = nshots
        self.words = (nshots + 63) // 64
        self.S = np.zeros((max(circuit.n_qubits, 1), self.words), dtype=np.uint64)
        self.Cb = np.zeros((max(circuit.n_bits, 1), self.words), dtype=np.uint64)
        self.P = np.zeros(self.words, dtype=np.uint64)
        self.FAIL = np.zeros(self.words, dtype=np.uint64)
        self.ACC = np.zeros(self.words * 64 if circuit.has_zpow else 1, dtype=np.uint64)
        self.ones = np.full(self.words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        self.rng = np.array([np.uint64(seed) * np.uint64(0x2545F4914F6CDD1D) + np.uint64(12345)],
                            dtype=np.uint64)

    def reset_state(self, seed: int | None = None) -> None:
        self.S[:] = 0
        self.Cb[:] = 0
        self.P[:] = 0
        self.FAIL[:] = 0
        self.ACC[:] = 0
        if seed is not None:
            self.rng[0] = np.uint64(seed) * np.uint64(0x2545F4914F6CDD1D) + np.uint64(12345)

    def write(self, qubits, values) -> None:
        """Set a register (LSB first) to one integer per shot (< 2**63)."""
        values = np.asarray(values, dtype=np.uint64)
        for j, q in enumerate(qubits):
            self.S[q] = pack_bits(((values >> np.uint64(j)) & np.uint64(1)).astype(np.uint8),
                                  self.words)

    def write_bits(self, qubit, bits) -> None:
        self.S[qubit] = pack_bits(np.asarray(bits, dtype=np.uint8), self.words)

    def run(self) -> None:
        c = self.circuit
        _run(c.op, c.qa, c.qb, c.qc, c.cond, c.ang, self.S, self.Cb, self.P, self.ACC,
             self.FAIL, self.ones, self.rng)

    def read(self, qubits) -> np.ndarray:
        """A register (LSB first, at most 64 bits) as one integer per shot."""
        out = np.zeros(self.nshots, dtype=np.uint64)
        for j, q in enumerate(qubits):
            out |= unpack_bits(self.S[q], self.nshots).astype(np.uint64) << np.uint64(j)
        return out

    def dirty_qubits(self, keep) -> np.ndarray:
        """Per-shot flag: some qubit outside ``keep`` is 1."""
        mask = np.ones(self.S.shape[0], dtype=bool)
        mask[list(keep)] = False
        acc = np.zeros(self.words, dtype=np.uint64)
        for q in np.nonzero(mask[:self.circuit.n_qubits])[0]:
            acc |= self.S[q]
        return unpack_bits(acc, self.nshots).astype(bool)

    def failed(self) -> np.ndarray:
        return unpack_bits(self.FAIL, self.nshots).astype(bool)

    def sign_flipped(self) -> np.ndarray:
        """Shots whose accumulated +-1 phase is -1."""
        return unpack_bits(self.P, self.nshots).astype(bool)

    def phase_units(self) -> np.ndarray:
        """Accumulated phase per shot in 2^-64 turns (``Z_POW`` plus sign flips)."""
        acc = self.ACC[:self.nshots].copy() if self.circuit.has_zpow else np.zeros(
            self.nshots, dtype=np.uint64)
        acc += self.sign_flipped().astype(np.uint64) << np.uint64(63)
        return acc
