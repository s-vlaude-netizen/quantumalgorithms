"""Arithmetic building blocks for kmlite circuits.

Registers are lists of qubit ids, least significant bit first. Every routine
returns its ancillas to |0> and frees them; Toffoli costs are exact and are
quoted per function. Carries and AND ancillas are erased by measurement-based
uncomputation, which costs no Toffolis.
"""

from __future__ import annotations

import itertools

from .kmlite import Builder

# ------------------------------------------------------------------- adders

def add(bld: Builder, tgt: list[int], src: list[int]) -> None:
    """``tgt += src`` modulo ``2**len(tgt)``; ``src`` may be shorter. Gidney's
    ripple-carry adder: ``len(tgt) - 1`` Toffolis."""
    n, m = len(tgt), len(src)
    assert 1 <= m <= n
    carry: list[int | None] = [None] * (n + 1)
    for i in range(n - 1):
        ci = carry[i]
        if i < m:
            ai = src[i]
            if ci is None:
                carry[i + 1] = bld.and_(ai, tgt[i])
            else:
                bld.cx(ci, ai)
                bld.cx(ci, tgt[i])
                c = bld.and_(ai, tgt[i])
                bld.cx(ci, c)
                carry[i + 1] = c
        else:                                   # source bit is 0: the carry just ripples
            carry[i + 1] = bld.and_(tgt[i], ci)
    top = n - 1
    if top < m:
        bld.cx(src[top], tgt[top])
    if carry[top] is not None:
        bld.cx(carry[top], tgt[top])
    for i in range(n - 2, -1, -1):
        ci, c = carry[i], carry[i + 1]
        if i < m:
            ai = src[i]
            if ci is None:
                bld.unand(c, ai, tgt[i])
                bld.cx(ai, tgt[i])
            else:
                bld.cx(ci, c)
                bld.unand(c, ai, tgt[i])
                bld.cx(ai, tgt[i])
                bld.cx(ci, tgt[i])
                bld.cx(ci, ai)
        else:
            bld.unand(c, tgt[i], ci)
            bld.cx(ci, tgt[i])


def sub(bld: Builder, tgt: list[int], src: list[int]) -> None:
    """``tgt -= src`` modulo ``2**len(tgt)``; ``len(tgt) - 1`` Toffolis."""
    for q in tgt:
        bld.x(q)
    add(bld, tgt, src)
    for q in tgt:
        bld.x(q)


def cadd(bld: Builder, ctrl: int, tgt: list[int], src: list[int], subtract: bool = False) -> None:
    """``tgt +=/-= src`` if ``ctrl``: ``len(src)`` ANDs plus the adder, i.e.
    ``len(src) + len(tgt) - 1`` Toffolis."""
    temps = [bld.and_(ctrl, s) for s in src]
    (sub if subtract else add)(bld, tgt, temps)
    for t, s in zip(reversed(temps), reversed(src)):
        bld.unand(t, ctrl, s)


def add_const(bld: Builder, tgt: list[int], value: int) -> None:
    """``tgt += value`` (a classical constant) modulo ``2**len(tgt)``."""
    value %= 1 << len(tgt)
    if value == 0:
        return
    bits = value.bit_length()
    scratch = bld.alloc(bits)
    for j in range(bits):
        if (value >> j) & 1:
            bld.x(scratch[j])
    add(bld, tgt, scratch)
    for j in range(bits):
        if (value >> j) & 1:
            bld.x(scratch[j])
    bld.free(scratch)


# ------------------------------------------------------------ table lookups

def unary_iteration(bld: Builder, addr: list[int], visit, ctrl: int | None = None,
                    used=None) -> None:
    """Call ``visit(index, control)`` for the indices in ``range(2**len(addr))``,
    with ``control`` a qubit that is |1> exactly when the address (and ``ctrl``)
    select ``index``. Without ``ctrl`` and with every index visited this costs
    ``2**len(addr) - 2`` Toffolis (Babbush et al., 2018). If ``used`` (a set of
    indices) is given, subtrees without any used index are skipped entirely."""
    k = len(addr)
    if used is None:
        def any_used(base, size):
            return True
    else:
        prefix = [0] * ((1 << k) + 1)
        for i in range(1 << k):
            prefix[i + 1] = prefix[i] + (1 if i in used else 0)

        def any_used(base, size):
            return prefix[base + size] > prefix[base]

    def rec(level: int, c: int | None, base: int) -> None:
        if level < 0:
            visit(base, c)
            return
        half = 1 << level
        a = addr[level]
        left, right = any_used(base, half), any_used(base + half, half)
        if c is None:
            if left:
                bld.x(a)
                rec(level - 1, a, base)
                bld.x(a)
            if right:
                rec(level - 1, a, base + half)
        elif left and right:
            bld.x(a)
            t = bld.and_(c, a)
            bld.x(a)
            rec(level - 1, t, base)
            bld.cx(t, c)
            rec(level - 1, c, base + half)
            bld.cx(t, c)
            bld.x(a)
            bld.unand(t, c, a)
            bld.x(a)
        elif left:
            bld.x(a)
            t = bld.and_(c, a)
            bld.x(a)
            rec(level - 1, t, base)
            bld.x(a)
            bld.unand(t, c, a)
            bld.x(a)
        elif right:
            t = bld.and_(c, a)
            rec(level - 1, t, base + half)
            bld.unand(t, c, a)

    rec(k - 1, ctrl, 0)


def lookup(bld: Builder, addr: list[int], table: list[int], out: list[int]) -> None:
    """``out ^= table[addr]``. ``2**len(addr) - 2`` Toffolis."""

    def visit(index, c):
        value = table[index] if index < len(table) else 0
        for j, q in enumerate(out):
            if (value >> j) & 1:
                bld.cx(c, q)

    used = {i for i, v in enumerate(table) if v}
    unary_iteration(bld, addr, visit, used=used)


def one_hot(bld: Builder, addr: list[int]) -> list[int]:
    """``2**len(addr)`` fresh qubits holding the one-hot encoding of ``addr``."""
    out = bld.alloc(1 << len(addr))
    unary_iteration(bld, addr, lambda index, c: bld.cx(c, out[index]))
    return out


def un_one_hot(bld: Builder, addr: list[int], out: list[int]) -> None:
    """Erase a one-hot register by X-basis measurement and a unary iteration of
    classically controlled Z gates."""
    bits = [bld.hmr(q) for q in out]
    unary_iteration(bld, addr, lambda index, c: bld.z(c, cond=bits[index]))
    for b in bits:
        bld.free_bit(b)
    bld.free(out)


def unlookup(bld: Builder, addr: list[int], table: list[int], out: list[int]) -> None:
    """Erase ``out == table[addr]`` without recomputing the lookup.

    Measures ``out`` in the X basis, then removes the phase kickback
    ``(-1)**(c . table[addr])`` with a two-level scheme: the low address bits are
    one-hot encoded (``2**k_lo`` Toffolis), the high bits are iterated over
    (``2**k_hi``), and the one-hot register is erased the same way: about
    ``3 * 2**(k/2)`` Toffolis instead of ``2**k``.
    """
    k = len(addr)
    assert k >= 2
    bits = [bld.hmr(q) for q in out]
    k_lo = max(1, k // 2)
    lo, hi = addr[:k_lo], addr[k_lo:]
    hot = one_hot(bld, lo)

    def visit(high, c):
        for u in range(1 << k_lo):
            index = (high << k_lo) | u
            value = table[index] if index < len(table) else 0
            sources = [bits[j] for j in range(len(out)) if (value >> j) & 1]
            if not sources:
                continue
            parity = bld.alloc_bit()
            bld._emit(14, parity)                         # BIT_STORE0
            for s in sources:
                bld.bit_xor(parity, s)
            bld.cz(c, hot[u], cond=parity)
            bld.free_bit(parity)

    used_high = {i >> k_lo for i, v in enumerate(table) if v}
    unary_iteration(bld, hi, visit, used=used_high)
    un_one_hot(bld, lo, hot)
    for b in bits:
        bld.free_bit(b)


# ----------------------------------------------------------------- shifting

def controlled_rotate(bld: Builder, ctrl: int, reg: list[int], shift: int) -> None:
    """Cyclically rotate ``reg`` towards the most significant end by ``shift`` if
    ``ctrl`` (bit ``j`` moves to ``(j + shift) % len(reg)``). ``len(reg)`` minus
    the number of permutation cycles CSWAPs, one Toffoli each."""
    n = len(reg)
    shift %= n
    if shift == 0:
        return
    seen = [False] * n
    for start in range(n):
        if seen[start]:
            continue
        cycle, j = [], start
        while not seen[j]:
            seen[j] = True
            cycle.append(j)
            j = (j + shift) % n
        # moving each element one step along the cycle takes len(cycle) - 1 swaps
        for u, v in reversed(list(itertools.pairwise(cycle))):
            bld.cswap(ctrl, reg[u], reg[v])


# -------------------------------------------------------- truncated multiplication

def plan_mac(n_ctrl: int, src_len: int, shift: int, src_max: int | None = None):
    """Row windows for ``acc += sum_j ctrl_j * ((src << j) >> shift)``.

    Rows are applied in increasing ``j``; before row ``j`` the accumulator holds at
    most ``acc_max`` and the row adds at most ``row_max``, so the adder only needs
    the bits that sum can reach. Returns ``(rows, acc_len)`` with
    ``rows = [(j, src_lo, win_lo, win_hi)]``: add ``src[src_lo:]`` into
    ``acc[win_lo:win_hi]`` controlled on ``ctrl_j``.
    """
    src_max = (1 << src_len) - 1 if src_max is None else src_max
    rows, acc_max = [], 0
    for j in range(n_ctrl):
        src_lo = max(0, shift - j)
        if src_lo >= src_len:
            continue
        offset = max(0, j - shift)
        row_max = (src_max << j) >> shift
        if row_max == 0:
            continue
        top = (acc_max + row_max).bit_length()
        rows.append((j, src_lo, offset, top))
        acc_max += row_max
    return rows, acc_max.bit_length()


def run_mac(bld: Builder, acc: list[int], ctrls: list[int], src: list[int], rows,
            subtract: bool = False) -> None:
    """Apply a ``plan_mac`` plan: add rows in order, or subtract them in reverse."""
    for j, src_lo, win_lo, win_hi in (reversed(rows) if subtract else rows):
        cadd(bld, ctrls[j], acc[win_lo:win_hi], src[src_lo:], subtract=subtract)
