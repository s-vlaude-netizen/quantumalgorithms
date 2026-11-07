"""Experiment 046 -- Benchmark I of Google's Kaggle competition: phasing by 1/sqrt(x).

Khattar et al. (arXiv:2610.23832, Section III.1; kickmix competition launched
21 Oct 2026) ask for a circuit implementing |x>|0> -> exp(2 pi i / sqrt(x)) |x>|0>
for every 22-bit ``x`` in ``[1, 3_133_452]``, with the phase correct to a relative
error of 2^-24, scored by ``S = Q * C`` (qubits times Toffoli-equivalents). The
reference circuit normalises ``x``, evaluates a cubic polynomial from a 2-entry
table and applies one Newton-Raphson step: Q = 260, C = 7,988, S = 2.08e6.

This module designs a different circuit, verifies it exhaustively on all
3,133,452 inputs with a simulator written from the published instruction set
(``qres.kmlite``; Google's own package was not installable here), and scores it
by the paper's formulas.

**Design.** After normalising ``x`` by pairs of leading zeros to a mantissa
``M in [2^20, 2^22)`` and a shift ``s``, the phase is ``2^(s-10) / sqrt(m)`` with
``m = M / 2^20 in [1, 4)``. The mantissa's top ``K`` bits address a table of
quadratic coefficients ``y0 = c0 - D (c1 - c2 D)`` (``D`` the low bits), two
truncated multiplications evaluate it, the result is shifted back by ``s`` with a
barrel shifter and phased bit by bit with ``Z_POW``, and everything is
uncomputed.
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qres import kmarith as ka
from qres import kmlite as km

X_BITS = 22
X_MAX = 3 * (2 ** 10 - 2) ** 2          # 3_133_452
REFERENCE = {"qubits": 260, "toffoli": 7735, "cost": 7988, "score": 2.08e6}


# ------------------------------------------------------------------ the pipeline

@dataclass(frozen=True)
class Design:
    """Fixed-point parameters of the quadratic-interpolation pipeline."""

    index_bits: int = 8        # K: table index = top K bits of the normalised mantissa
    f0: int = 28               # fractional bits of y0 = m^-1/2 (output phase resolution)
    f1: int = 21               # fractional bits of the linear coefficient / T
    f2: int = 15               # fractional bits of the quadratic coefficient

    @property
    def d_bits(self) -> int:
        return X_BITS - self.index_bits

    @property
    def sh1(self) -> int:      # right shift of the product c2 * D
        return self.f2 + 20 - self.f1

    @property
    def sh2(self) -> int:      # right shift of the product T * D
        return self.f1 + 20 - self.f0


def normalise(x: np.ndarray):
    """(s, M): the number of leading zero pairs of a 22-bit value and ``x << 2s``."""
    bits = np.floor(np.log2(x.astype(np.float64))).astype(np.int64) + 1
    s = (X_BITS - bits) // 2
    return s, x.astype(np.int64) << (2 * s)


def truncated_product(multiplier_bits: np.ndarray, count: int, multiplicand: np.ndarray,
                      shift: int) -> np.ndarray:
    """``sum_j bit_j(multiplier) * (multiplicand << j >> shift)``: the product with
    every partial-product bit below column ``shift`` dropped -- exactly what the
    circuit's row-wise controlled additions compute."""
    acc = np.zeros_like(multiplicand)
    for j in range(count):
        row = multiplicand << (j - shift) if j >= shift else multiplicand >> (shift - j)
        acc = acc + ((multiplier_bits >> j) & 1) * row
    return acc


def evaluate(design: Design, c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, mantissa: np.ndarray):
    """Integer ``y0`` for normalised mantissas, from per-index coefficient arrays."""
    idx = mantissa >> design.d_bits
    d = mantissa & ((1 << design.d_bits) - 1)
    t = c1[idx] - truncated_product(d, design.d_bits, c2[idx], design.sh1)
    return c0[idx] - truncated_product(d, design.d_bits, t, design.sh2), t


def segment_grid(design: Design):
    """Mantissa grid ``(n_segments, 2**d_bits)`` for the used indices."""
    first = 1 << (design.index_bits - 2)
    idx = np.arange(first, 1 << design.index_bits, dtype=np.int64)
    d = np.arange(1 << design.d_bits, dtype=np.int64)
    return idx, (idx[:, None] << design.d_bits) | d[None, :]


def fit_coefficients(design: Design, search: int = 2):
    """Per-index integer coefficients ``(C0, C1, C2)`` minimising the worst relative
    error of the truncated pipeline over every mantissa in the segment.

    Start from the Chebyshev-interpolation quadratic of ``m^-1/2`` on the segment,
    round, and refine ``C1`` and ``C2`` by a local search on the *exact* integer
    pipeline; ``C0`` is then the centre of the residual range."""
    idx, grid = segment_grid(design)
    m = grid.astype(np.float64) * 2.0 ** -20
    y_exact = m ** -0.5
    h = (1 << design.d_bits) * 2.0 ** -20
    # Chebyshev nodes on the segment, quadratic interpolation in D
    nodes = (np.cos((2 * np.arange(3) + 1) * np.pi / 6) + 1) / 2 * h
    c1r = np.empty(len(idx))
    c2r = np.empty(len(idx))
    for k, i in enumerate(idx):
        start = (int(i) << design.d_bits) * 2.0 ** -20
        yy = (start + nodes) ** -0.5
        coeff = np.polyfit(nodes, yy, 2)            # c2', c1', c0'
        c2r[k], c1r[k] = coeff[0], -coeff[1]
    base1 = np.round(c1r * 2 ** design.f1).astype(np.int64)
    base2 = np.round(c2r * 2 ** design.f2).astype(np.int64)
    tol = 2.0 ** -24 * y_exact * 2 ** design.f0           # allowed error in y0 units
    best_err = np.full(len(idx), np.inf)
    best = [base1.copy(), base2.copy(), np.zeros(len(idx), dtype=np.int64)]
    dd = grid & ((1 << design.d_bits) - 1)
    for d2 in range(-search, search + 1):
        for d1 in range(-search * 3, search * 3 + 1):
            c1c, c2c = base1 + d1, base2 + d2
            t = c1c[:, None] - truncated_product(dd, design.d_bits,
                                                 np.broadcast_to(c2c[:, None], dd.shape).copy(),
                                                 design.sh1)
            v = -truncated_product(dd, design.d_bits, t, design.sh2)
            res = v - y_exact * 2 ** design.f0           # y0 = C0 + v  -> C0 = -centre
            lo, hi = res.min(axis=1), res.max(axis=1)
            c0 = np.round(-(lo + hi) / 2).astype(np.int64)
            err = np.abs(res + c0[:, None]) / tol
            worst = err.max(axis=1)
            better = worst < best_err
            best_err = np.where(better, worst, best_err)
            best[0] = np.where(better, c1c, best[0])
            best[1] = np.where(better, c2c, best[1])
            best[2] = np.where(better, c0, best[2])
    size = 1 << design.index_bits
    tables = [np.zeros(size, dtype=np.int64) for _ in range(3)]
    tables[0][idx] = best[2]
    tables[1][idx] = best[0]
    tables[2][idx] = best[1]
    return tables, best_err


def all_inputs() -> np.ndarray:
    return np.arange(1, X_MAX + 1, dtype=np.int64)


def phase_error(design: Design, tables, x: np.ndarray):
    """Exact-phase relative error of the integer pipeline for each ``x``."""
    s, mantissa = normalise(x)
    y0, _ = evaluate(design, tables[0], tables[1], tables[2], mantissa)
    y_exact = (mantissa.astype(np.float64) * 2.0 ** -20) ** -0.5
    return (y0 * 2.0 ** -design.f0 - y_exact) / y_exact, s


# ------------------------------------------------------------------ the circuit

def normalise_circuit(bld, x):
    """Leading-zero-pair normalisation by a cascade of controlled lane rotations
    (Appendix B.2 of the paper): ``f_k`` flags that the top pair of the current
    register is 00 (one Toffoli), and then the live bits move up by two. Returns the
    thermometer flags ``f_0..f_9``; ``2 + 10 * 11`` forward Toffolis, 110 in reverse."""
    flags = []
    for k in range(X_BITS // 2 - 1):
        top1, top0 = x[X_BITS - 1], x[X_BITS - 2]
        bld.x(top1)
        bld.x(top0)
        f = bld.and_(top1, top0)
        bld.x(top1)
        bld.x(top0)
        flags.append(f)
        lanes = (list(range(2 * k, X_BITS - 1, 2)), list(range(2 * k + 1, X_BITS, 2)))
        for lane in lanes:
            ka.controlled_rotate(bld, f, [x[p] for p in lane], 1)
    return flags


def denormalise_circuit(bld, x, flags):
    for k in reversed(range(len(flags))):
        lanes = (list(range(2 * k, X_BITS - 1, 2)), list(range(2 * k + 1, X_BITS, 2)))
        for lane in lanes:
            ka.controlled_rotate(bld, flags[k], [x[p] for p in lane], -1)
        top1, top0 = x[X_BITS - 1], x[X_BITS - 2]
        bld.x(top1)
        bld.x(top0)
        bld.unand(flags[k], top1, top0)
        bld.x(top1)
        bld.x(top0)


def pair_flags(bld, x):
    """``z[k]`` = 1 iff bit pair ``k`` of ``x`` (bits 2k+1, 2k) is 00, k = 1..10, and the
    thermometer code ``t[j] = z[10] & ... & z[11-j]`` (``s >= j`` for ``s`` leading
    zero pairs). 10 + 9 Toffolis; returns ``(z, t)``."""
    z = {}
    for k in range(1, X_BITS // 2):
        hi, lo = x[2 * k + 1], x[2 * k]
        bld.x(hi)
        bld.x(lo)
        z[k] = bld.and_(hi, lo)
        bld.x(hi)
        bld.x(lo)
    t = {1: z[10]}
    for j in range(2, X_BITS // 2):
        t[j] = bld.and_(t[j - 1], z[11 - j])
    return z, t


def unpair_flags(bld, x, z, t):
    for j in range(X_BITS // 2 - 1, 1, -1):
        bld.unand(t[j], t[j - 1], z[11 - j])
    for k in range(X_BITS // 2 - 1, 0, -1):
        hi, lo = x[2 * k + 1], x[2 * k]
        bld.x(hi)
        bld.x(lo)
        bld.unand(z[k], hi, lo)
        bld.x(hi)
        bld.x(lo)


S_GROUPS = ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), (2, 4, 6, 8, 10), (4, 8), (8,))


def binary_bits_from_thermometer(bld, t, sb):
    """XOR the binary digits of ``s = sum_j t[j]`` into the fresh qubits ``sb``
    (digit 0 is the parity of all flags, digit 1 of the even-numbered ones, ...)."""
    for q, group in zip(sb, S_GROUPS):
        for j in group:
            bld.cx(t[j], q)


def normalise_binary(bld, x):
    """Normalise by binary stages: compute the shift ``s`` in four qubits (19
    Toffolis, flags erased again at once), then shift by 16, 8, 4, 2 bits, each
    stage swapping only the bits that can be live. Returns ``(sb, swaps)``."""
    z, t = pair_flags(bld, x)
    sb = bld.alloc(4)
    binary_bits_from_thermometer(bld, t, sb)
    unpair_flags(bld, x, z, t)
    swaps = []
    for ctrl, r in zip(reversed(sb), (16, 8, 4, 2)):         # sb[3] has value 8 pairs
        for i in range(X_BITS - r - 1, -1, -1):
            swaps.append((ctrl, x[i], x[i + r]))
    for ctrl, a, b in swaps:
        bld.cswap(ctrl, a, b)
    return sb, swaps


def denormalise_binary(bld, x, sb, swaps):
    for ctrl, a, b in reversed(swaps):
        bld.cswap(ctrl, a, b)
    z, t = pair_flags(bld, x)
    binary_bits_from_thermometer(bld, t, sb)
    for q in sb:
        bld.assert0(q)
    unpair_flags(bld, x, z, t)
    bld.free(sb)


def binary_shift_bits(bld, flags):
    """The binary digits of ``s = sum(flags)`` as 4 fresh qubits, by CNOTs only:
    ``s mod 2`` is the parity of the flags, ``floor(s/2)`` the parity of the odd-
    numbered ones, and so on."""
    bits = bld.alloc(4)
    groups = (range(10), range(1, 10, 2), (3, 7), (7,))
    for q, group in zip(bits, groups):
        for k in group:
            bld.cx(flags[k], q)
    return bits


def barrel_swaps(sb, reg, n0):
    """CSWAP list shifting ``reg`` (data in ``reg[:n0]``) up by ``s`` positions,
    stages 1, 2, 4, 8 with the windows reachable for ``s <= 10``."""
    swaps = []
    for stage, (r, hi) in enumerate(((1, n0), (2, n0 + 1), (4, n0 + 3), (8, n0 + 2))):
        for i in range(hi - 1, -1, -1):
            swaps.append((sb[stage], reg[i], reg[i + r]))
    return swaps


def build_circuit(design: Design, tables, with_phase: bool = True, normalise: str = "binary"):
    """The full phasing circuit. Returns ``(builder, circuit, info)``."""
    from fractions import Fraction

    bld = km.Builder()
    x = bld.register("x", X_BITS)
    db = design.d_bits
    widths = [max(1, int(t.max()).bit_length()) for t in tables]
    n0, n1, n2 = widths
    with bld.section("normalise"):
        if normalise == "binary":
            sb, norm_swaps = normalise_binary(bld, x)
            flags = None
        else:
            flags = normalise_circuit(bld, x)
            sb = binary_shift_bits(bld, flags)
    addr, dbits = x[db:], x[:db]
    c0, c1, c2 = bld.alloc(n0), bld.alloc(n1), bld.alloc(n2)
    combined = [int(a) | (int(b) << n0) | (int(c) << (n0 + n1))
                for a, b, c in zip(tables[0], tables[1], tables[2])]
    with bld.section("lookup"):
        ka.lookup(bld, addr, combined, c0 + c1 + c2)
    max2, max1 = int(tables[2].max()), int(tables[1].max())
    with bld.section("product 1"):
        rows1, len1 = ka.plan_mac(db, n2, design.sh1, max2)
        acc1 = bld.alloc(len1)
        ka.run_mac(bld, acc1, dbits, c2, rows1)
        ka.sub(bld, c1, acc1)                                  # c1 now holds T
    with bld.section("product 2"):
        rows2, len2 = ka.plan_mac(db, n1, design.sh2, max1)
        acc2 = bld.alloc(len2)
        ka.run_mac(bld, acc2, dbits, c1, rows2)
        ka.sub(bld, c0, acc2)                                  # c0 now holds y0
    pad = bld.alloc(10)
    yreg = c0 + pad
    swaps = barrel_swaps(sb, yreg, n0)
    with bld.section("shift"):
        for ctrl, a, b in swaps:
            bld.cswap(ctrl, a, b)
    if with_phase:
        with bld.section("phase"):
            for p, q in enumerate(yreg):
                bld.zpow(q, Fraction(1, 2 ** (10 + design.f0 - p)) if 10 + design.f0 - p > 0
                         else Fraction(2 ** (p - 10 - design.f0)))
    with bld.section("unshift"):
        for ctrl, a, b in reversed(swaps):
            bld.cswap(ctrl, a, b)
    for q in pad:
        bld.assert0(q)
    bld.free(pad)
    with bld.section("unproduct 2"):
        ka.add(bld, c0, acc2)
        ka.run_mac(bld, acc2, dbits, c1, rows2, subtract=True)
    for q in acc2:
        bld.assert0(q)
    bld.free(acc2)
    with bld.section("unproduct 1"):
        ka.add(bld, c1, acc1)
        ka.run_mac(bld, acc1, dbits, c2, rows1, subtract=True)
    for q in acc1:
        bld.assert0(q)
    bld.free(acc1)
    with bld.section("unlookup"):
        ka.unlookup(bld, addr, combined, c0 + c1 + c2)
    bld.free(c0 + c1 + c2)
    with bld.section("denormalise"):
        if normalise == "binary":
            denormalise_binary(bld, x, sb, norm_swaps)
        else:
            # undo the binary digits (CNOTs, free), then the cascade
            groups = (range(10), range(1, 10, 2), (3, 7), (7,))
            for q, group in zip(sb, groups):
                for k in group:
                    bld.cx(flags[k], q)
                bld.assert0(q)
            bld.free(sb)
            denormalise_circuit(bld, x, flags)
    return bld, bld.finish(), {"widths": widths, "rows1": rows1, "rows2": rows2}


# ------------------------------------------------------------- verification & score

def target_units(x: np.ndarray) -> np.ndarray:
    """``1 / sqrt(x)`` turns in units of 2^-64, modulo one turn."""
    big = np.longdouble(2.0) ** 64
    t = np.floor(big / np.sqrt(x.astype(np.longdouble)))
    return np.fmod(t, big).astype(np.uint64)


def verify(circuit, x: np.ndarray, seed: int = 0):
    """Simulate every ``x`` once. Returns per-shot results and a pass/fail summary."""
    sim = km.Simulator(circuit, len(x), seed=seed)
    xreg = circuit.registers["x"]
    sim.write(xreg, x)
    sim.run()
    restored = bool((sim.read(xreg) == x.astype(np.uint64)).all())
    dirty = int(sim.dirty_qubits(xreg).sum())
    failed = int(sim.failed().sum())
    phase = sim.phase_units()
    diff = (phase - target_units(x)).astype(np.int64).astype(np.float64)      # signed, wraps
    tol = 2.0 ** 40 / np.sqrt(x.astype(np.float64))
    return {"restored": restored, "dirty_shots": dirty, "assert_failures": failed,
            "max_error_over_tolerance": float((np.abs(diff) / tol).max()),
            "margin_turns": float((tol - np.abs(diff)).min() * 2.0 ** -64),
            "error_units": diff}


def score(builder, margin_turns: float):
    """The paper's score: ``Q * (Toffolis + T/4)``, finer rotations priced at
    ``0.56 log2(1/eps) + 5.3`` T gates with ``eps = pi * margin / n_rot``."""
    n_fine, n_t = builder.rotations["fine"], builder.rotations["t"]
    eps = math.pi * margin_turns / max(n_fine, 1)
    t_per_rotation = 0.56 * math.log2(1 / eps) + 5.3 if margin_turns > 0 else math.inf
    t_total = n_t + n_fine * t_per_rotation
    cost = builder.toffoli + t_total / 4
    return {"qubits": builder.n_qubits, "toffoli": builder.toffoli, "t_gates": n_t,
            "fine_rotations": n_fine, "t_per_rotation": t_per_rotation, "cost": cost,
            "score": builder.n_qubits * cost}


def _official_worker(job):
    """One process of the official-simulator sweep: ``x`` in ``[lo, hi)``, every ``x``
    simulated ``shots`` times. Returns counts and the smallest margin."""
    import hashlib

    import kickmix as kx

    text, lo, hi, shots, base_seed, batch_inputs = job
    circuit = kx.Circuit(text)
    scratch = [kx.q(i) for i in range(circuit.num_qubits) if i >= X_BITS]
    mask = (1 << 64) - 1
    out = {"shots": 0, "failures": 0, "restored_failures": 0, "dirty_shots": 0,
           "nondeterministic_inputs": 0, "max_error_over_tolerance": 0.0,
           "min_margin_units": math.inf, "worst_x": 0}
    for start in range(lo, hi, batch_inputs):
        xs = np.arange(start, min(start + batch_inputs, hi), dtype=np.int64)
        shot_x = np.repeat(xs, shots)
        seed = int.from_bytes(hashlib.sha256(f"{base_seed}:{start}".encode()).digest(), "big")
        sim = kx.Simulator(batch_size=len(shot_x), debug_print=False, seed=seed)
        sim.use_same_registers_as(circuit)
        bits = ((shot_x[None, :] >> np.arange(X_BITS)[:, None]) & 1).astype(bool)
        sim.write_across_shots("x", bits)
        sim.do(circuit)
        out["restored_failures"] += int((sim.read_across_shots("x") != bits).any(axis=0).sum())
        out["dirty_shots"] += int(np.asarray(sim.read_across_shots(scratch)).any(axis=0).sum())
        units = np.empty(len(shot_x), dtype=np.uint64)
        for i in range(len(shot_x)):
            p = sim.read_shot_phase(i)                       # half turns
            units[i] = ((p << 63) if isinstance(p, int)
                        else (p.numerator << 63) // p.denominator) & mask
        target = np.array([math.isqrt((1 << 128) // int(x)) & mask for x in xs],
                          dtype=np.uint64)
        diff = (units - np.repeat(target, shots)).astype(np.int64).astype(np.float64)
        tol = 2.0 ** 40 / np.sqrt(shot_x.astype(np.float64))
        ratio = np.abs(diff) / tol
        out["shots"] += len(shot_x)
        out["failures"] += int((ratio > 1.0).sum())
        grid = diff.reshape(len(xs), shots)
        out["nondeterministic_inputs"] += int((grid != grid[:, :1]).any(axis=1).sum())
        out["max_error_over_tolerance"] = max(out["max_error_over_tolerance"], float(ratio.max()))
        margin = tol - np.abs(diff)
        low = float(margin.min())
        if low < out["min_margin_units"]:
            out["min_margin_units"], out["worst_x"] = low, int(shot_x[int(margin.argmin())])
    return out


def verify_official(circuit, shots_per_x: int = 64, workers: int = 4, seed: int = 0,
                    batch_inputs: int = 4096, lo: int = 1, hi: int = X_MAX + 1):
    """Run ``x`` in ``[lo, hi)`` through Google's own ``kickmix`` simulator, every input
    ``shots_per_x`` times, as the competition's verifier does: ``x`` restored, every
    scratch qubit back in |0>, and the tracked phase within ``2^-24 / sqrt(x)`` turns
    of ``floor(2^64 / sqrt(x))`` units. The margin is returned in turns."""
    from concurrent.futures import ProcessPoolExecutor

    from qres import kmofficial

    text = kmofficial.to_text(circuit)
    bounds = np.linspace(lo, hi, workers + 1).astype(np.int64)
    jobs = [(text, int(a), int(b), shots_per_x, seed, batch_inputs)
            for a, b in itertools.pairwise(bounds) if b > a]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_official_worker, jobs))
    total = {key: sum(p[key] for p in parts)
             for key in ("shots", "failures", "restored_failures", "dirty_shots",
                         "nondeterministic_inputs")}
    worst = min(parts, key=lambda p: p["min_margin_units"])
    total["max_error_over_tolerance"] = max(p["max_error_over_tolerance"] for p in parts)
    total["margin_turns"] = worst["min_margin_units"] * 2.0 ** -64
    total["worst_x"] = worst["worst_x"]
    return total


def cached_tables(design: Design, directory: Path):
    """Fitted coefficient tables, cached because the fit takes about 30 s."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"tables_K{design.index_bits}_f{design.f0}_{design.f1}_{design.f2}.npz"
    if path.exists():
        data = np.load(path)
        return [data["c0"], data["c1"], data["c2"]]
    tables, _ = fit_coefficients(design)
    np.savez(path, c0=tables[0], c1=tables[1], c2=tables[2])
    return tables


def official_counts(circuit):
    """Qubit and gate counts as Google's ``kickmix`` reports them, in the shape
    :func:`score` expects."""
    from types import SimpleNamespace

    from qres import kmofficial

    official = kmofficial.to_official(circuit)
    stats = official.operation_statistics()
    n_toffoli, n_t = stats.num_toffoli_like, stats.num_t_like
    return SimpleNamespace(n_qubits=official.num_qubits, toffoli=n_toffoli,
                           rotations={"t": n_t, "fine": stats.num_non_clifford - n_toffoli - n_t})


def main(argv=None) -> int:
    import argparse

    from qres.bench import RESULTS_DIR

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--official", action="store_true",
                        help="also run Google's kickmix simulator (pip install kickmix)")
    parser.add_argument("--shots", type=int, default=64, help="shots per input for --official")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--kmb", type=Path, help="write the circuit to this .kmb file")
    args = parser.parse_args(argv)
    design = Design()
    tables = cached_tables(design, RESULTS_DIR / "raw" / "exp046")
    x = all_inputs()
    rel, _ = phase_error(design, tables, x)
    print(f"model: worst relative error {np.abs(rel).max():.3e} = "
          f"2^{math.log2(np.abs(rel).max()):.2f} (limit 2^-24)")
    bld, circuit, _info = build_circuit(design, tables)
    print(f"circuit: {len(circuit)} instructions, {circuit.n_qubits} qubits, "
          f"{bld.toffoli} Toffolis, rotations {bld.rotations}")
    by_section = {}
    for path, count in bld.section_costs.items():
        by_section[path[0] if path else ""] = by_section.get(path[0] if path else "", 0) + count
    print("Toffolis by section:", by_section)
    import time

    start = time.time()
    result = verify(circuit, x)
    print(f"simulated {len(x)} inputs in {time.time() - start:.1f} s:",
          {k: v for k, v in result.items() if k != "error_units"})
    print(score(bld, result["margin_turns"]))
    if args.kmb:
        from qres import kmofficial

        kmofficial.to_official(circuit).to_file(args.kmb, format="kmb")
        print("wrote", args.kmb)
    if args.official:
        start = time.time()
        official = verify_official(circuit, shots_per_x=args.shots, workers=args.workers)
        print(f"official kickmix simulator, {official['shots']} shots in {time.time() - start:.0f} s:",
              official)
        print("official score:", score(official_counts(circuit), official["margin_turns"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
