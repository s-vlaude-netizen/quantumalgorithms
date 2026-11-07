"""Translate a :mod:`qres.kmlite` circuit into Google's official ``kickmix`` format.

``kickmix`` (the package of arXiv:2610.23832, ``pip install kickmix``) is an optional
dependency: nothing here imports it at module load. The translation is one line of
text per instruction, so a circuit built and verified with the lightweight builder
can be checked, counted and serialised (``.kmx`` text or ``.kmb`` binary) by the
official tools.

``ASSERT0`` is not a kickmix instruction. It becomes ``R``, which discards the qubit:
harmless on a clean qubit and a random phase on a dirty one, so the official fuzz
test fails loudly exactly where my simulator reports an assertion failure.
"""

from __future__ import annotations

from . import kmlite as km

_ONE_PHASE = 1 << 63            # Z_POW exponent 1.0 (a half turn) in units of 2^-64 turns


def zpow_exponent(units: int) -> str:
    """The exact decimal ``Z_POW`` exponent for a phase of ``units`` * 2^-64 turns.

    ``Z_POW t`` phases |1> by ``exp(i pi t)``, so ``t = 2 * turns = units / 2^63``, a
    dyadic rational with a terminating decimal expansion (at most 63 digits)."""
    whole, rest = divmod(units, _ONE_PHASE)
    digits = str(rest * 5 ** 63).zfill(63)
    return f"{whole}.{digits}"


def to_text(circuit: km.Circuit) -> str:
    """The circuit in the official text format, registers first."""
    lines = []
    for rid, (name, qubits) in enumerate(circuit.registers.items()):
        lines.extend(f"APPEND_TO_REGISTER q{q} r{rid}" for q in qubits)
        lines.append(f'REGISTER r{rid} "{name}"')
    op, qa, qb, qc = (circuit.op.tolist(), circuit.qa.tolist(), circuit.qb.tolist(),
                      circuit.qc.tolist())
    cond, ang = circuit.cond.tolist(), circuit.ang.tolist()
    for i, code in enumerate(op):
        a, b, c = qa[i], qb[i], qc[i]
        suffix = f" if b{cond[i]}" if cond[i] >= 0 else ""
        if code == km.X:
            lines.append(f"X q{a}{suffix}")
        elif code == km.CX:
            lines.append(f"CX q{a} q{b}{suffix}")
        elif code == km.CCX:
            lines.append(f"CCX q{a} q{b} q{c}{suffix}")
        elif code == km.SWAP:
            lines.append(f"SWAP q{a} q{b}{suffix}")
        elif code == km.CSWAP:
            lines.append(f"CSWAP q{a} q{b} q{c}{suffix}")
        elif code == km.Z:
            lines.append(f"Z q{a}{suffix}")
        elif code == km.CZ:
            lines.append(f"CZ q{a} q{b}{suffix}")
        elif code == km.CCZ:
            lines.append(f"CCZ q{a} q{b} q{c}{suffix}")
        elif code == km.ZPOW:
            lines.append(f"Z_POW q{a} {zpow_exponent(ang[i])}{suffix}")
        elif code == km.NEG:
            lines.append(f"NEG{suffix}")
        elif code == km.HMR:
            lines.append(f"HMR q{a} b{c}{suffix}")
        elif code == km.R:
            lines.append(f"R q{a}{suffix}")
        elif code == km.ASSERT0:
            lines.append(f"R q{a}")
        elif code == km.BIT_INV:
            lines.append(f"BIT_INVERT b{a}{suffix}")
        elif code == km.BIT_SET0:
            lines.append(f"BIT_STORE0 b{a}{suffix}")
        elif code == km.BIT_SET1:
            lines.append(f"BIT_STORE1 b{a}{suffix}")
        else:                                                   # pragma: no cover
            raise ValueError(f"unknown opcode {code}")
    return "\n".join(lines) + "\n"


def to_official(circuit: km.Circuit):
    """The circuit as a ``kickmix.Circuit`` (requires the ``kickmix`` package)."""
    import kickmix

    return kickmix.Circuit(to_text(circuit))
