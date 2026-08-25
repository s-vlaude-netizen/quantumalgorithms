"""Parameterised trial states.

Two families, with different failure modes:

* **Hardware-efficient** -- shallow and native to the device's coupling map, so
  it survives noise, but it explores a physically meaningless subspace and hits
  barren plateaus as it grows.
* **Chemistry-inspired (UCCSD)** -- respects particle number and spin, converges
  reliably, but is deep enough that device noise eats the answer.

Which one wins is a function of the noise level, and that trade-off is one of
the things this package is built to measure rather than assume.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def hartree_fock_state(bitstring: str) -> QuantumCircuit:
    """Prepare a computational-basis reference state (qubit 0 rightmost)."""
    n = len(bitstring)
    qc = QuantumCircuit(n)
    for q, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            qc.x(q)
    return qc


def hardware_efficient(
    num_qubits: int,
    reps: int = 2,
    *,
    entanglement: str | list[tuple[int, int]] = "linear",
    rotation_gates: tuple[str, ...] = ("ry", "rz"),
    entangler: str = "cx",
    initial_state: QuantumCircuit | None = None,
    coupling_map=None,
    name: str = "hea",
) -> QuantumCircuit:
    """A layered rotation/entangler ansatz.

    ``entanglement`` may be ``"linear"``, ``"circular"``, ``"full"``,
    ``"device"`` (use ``coupling_map`` directly, which avoids every SWAP the
    transpiler would otherwise insert), or an explicit edge list.

    ``entangler`` selects the two-qubit block:

    ``"cx"``
        The textbook choice -- and it silently destroys the reference state.
        A fixed CX layer acts whatever the parameters are, so at ``theta = 0``
        the circuit is *not* the initial state.  Measured on H4: the overlap
        between the zero-parameter ansatz and the Hartree-Fock determinant is
        **0.0000**, and its energy is -1.15 Ha against HF's -5.16.  The standard
        advice to "initialise near zero so you start at Hartree-Fock" is simply
        false here, and an optimiser starting 4 Ha away from the answer does not
        recover within any realistic shot budget.

    ``"cry"`` / ``"crx"``
        Controlled rotations, which are the identity at ``theta = 0``.  The
        zero-parameter circuit reproduces the initial state exactly, so the run
        starts at Hartree-Fock and the first descent direction is meaningful.
        Costs one extra parameter per edge per layer, and each controlled
        rotation compiles to two CX gates.
    """
    pairs = _entangler_pairs(num_qubits, entanglement, coupling_map)
    n_rot = len(rotation_gates) * num_qubits * (reps + 1)
    n_ent = 0 if entangler == "cx" else len(pairs) * reps
    theta = ParameterVector("t", n_rot + n_ent)

    qc = QuantumCircuit(num_qubits, name=name)
    if initial_state is not None:
        qc.compose(initial_state, inplace=True)

    k = 0
    entangler_params: list[int] = []
    for layer in range(reps + 1):
        for q in range(num_qubits):
            for gate in rotation_gates:
                getattr(qc, gate)(theta[k], q)
                k += 1
        if layer < reps:
            for a, b in pairs:
                if entangler == "cx":
                    qc.cx(a, b)
                elif entangler in ("cry", "crx", "crz"):
                    getattr(qc, entangler)(theta[k], a, b)
                    entangler_params.append(k)
                    k += 1
                else:
                    raise ValueError(f"unknown entangler {entangler!r}")
    qc.metadata = {"entangler_params": entangler_params}
    return qc


def _entangler_pairs(num_qubits, entanglement, coupling_map):
    if isinstance(entanglement, list):
        return entanglement
    if entanglement == "linear":
        return [(q, q + 1) for q in range(num_qubits - 1)]
    if entanglement == "circular":
        return [(q, (q + 1) % num_qubits) for q in range(num_qubits)]
    if entanglement == "full":
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    if entanglement == "device":
        if coupling_map is None:
            raise ValueError("entanglement='device' needs a coupling_map")
        edges = [(a, b) for a, b in coupling_map.get_edges() if a < num_qubits and b < num_qubits]
        # keep one direction per pair, and colour into non-overlapping layers
        seen, pairs = set(), []
        for a, b in edges:
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
        return pairs
    raise ValueError(f"unknown entanglement {entanglement!r}")


def _mapper_for(problem):
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )

    name = problem.metadata.get("mapper", "parity")
    if name == "parity":
        return ParityMapper(num_particles=problem.num_particles)
    if name == "bravyi_kitaev":
        return BravyiKitaevMapper()
    return JordanWignerMapper()


def chemistry_ansatz(
    problem,
    family: str = "uccsd",
    *,
    reps: int = 1,
    generalized: bool = False,
) -> QuantumCircuit:
    """A coupled-cluster-family ansatz on the Hartree-Fock reference.

    These matter for one specific reason (see RESEARCH_LOG Result 10): their
    generators include **double** excitations, so they have a non-zero gradient
    at the Hartree-Fock determinant.  A real-amplitude hardware-efficient ansatz
    does not -- its first-order response reaches only single excitations, which
    Brillouin's theorem decouples from HF -- and it therefore sits at
    Hartree-Fock forever, at any depth.

    The price is circuit size, and the families differ enormously in it:

    ``uccsd``   all singles and doubles.  Reaches chemical accuracy on H4
                (9.7e-6) at ~1072 two-qubit gates.
    ``puccd``   *paired* doubles only -- both electrons of a pair move together.
                4 parameters and ~250 two-qubit gates on H4, non-zero gradient
                at HF, but under-expressive (1.6e-2).
    ``succd``   singlet-adapted doubles; between the two.
    ``ucc:d``   doubles only, with ``reps`` repetitions.
    """
    from qiskit_nature.second_q.circuit.library import (
        HartreeFock,
        PUCCD,
        PUCCSD,
        SUCCD,
        UCC,
        UCCSD,
    )

    mapper = _mapper_for(problem)
    initial = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    args = (problem.num_spatial_orbitals, problem.num_particles, mapper)

    if family == "uccsd":
        ansatz = UCCSD(*args, initial_state=initial, reps=reps, generalized=generalized)
    elif family == "puccd":
        ansatz = PUCCD(*args, initial_state=initial, reps=reps)
    elif family == "puccsd":
        ansatz = PUCCSD(*args, initial_state=initial, reps=reps)
    elif family == "succd":
        ansatz = SUCCD(*args, initial_state=initial, reps=reps)
    elif family.startswith("ucc"):
        # UCC takes `excitations` as its THIRD positional argument, ahead of the
        # mapper, so the *args spread used above collides with it.
        excitations = family.split("-", 1)[1] if "-" in family else "sd"
        ansatz = UCC(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            excitations=excitations,
            qubit_mapper=mapper,
            reps=reps,
            generalized=generalized,
            initial_state=initial,
        )
    else:
        raise ValueError(f"unknown chemistry ansatz family {family!r}")

    return QuantumCircuit(ansatz.num_qubits).compose(ansatz.decompose(reps=4))


def uccsd(problem, *, reps: int = 1, generalized: bool = False) -> QuantumCircuit:
    """UCCSD on top of the Hartree-Fock reference for a MolecularProblem."""
    return chemistry_ansatz(problem, "uccsd", reps=reps, generalized=generalized)


def build_ansatz(spec: str, problem, environment=None) -> QuantumCircuit:
    """Build an ansatz from a short spec string.

    ``"hea:2"``            two-rep hardware-efficient, linear CX entanglement
    ``"hea:3:circular"``   circular entanglement
    ``"hea:2:device"``     entangle along the backend's real coupling map
    ``"hea:2:linear:cry"`` controlled-RY entanglers -- theta=0 is the reference
    ``"uccsd"``            all singles and doubles (molecular problems only)
    ``"puccd"``            paired doubles -- far shallower, still has gradient
    ``"succd:2"``          singlet doubles, two repetitions
    ``"ucc-d:3"``          doubles only, three repetitions
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind in ("uccsd", "puccd", "puccsd", "succd") or kind.startswith("ucc-"):
        reps = int(parts[1]) if len(parts) > 1 else 1
        return chemistry_ansatz(problem, kind, reps=reps)
    if kind == "hea":
        reps = int(parts[1]) if len(parts) > 1 else 2
        ent = parts[2] if len(parts) > 2 else "linear"
        entangler = parts[3] if len(parts) > 3 else "cx"
        initial = None
        if hasattr(problem, "hf_bitstring"):
            initial = hartree_fock_state(problem.hf_bitstring)
        return hardware_efficient(
            problem.num_qubits,
            reps=reps,
            entanglement=ent,
            entangler=entangler,
            initial_state=initial,
            coupling_map=environment.coupling_map if environment else None,
        )
    raise ValueError(f"unknown ansatz spec {spec!r}")


def initial_point(
    ansatz: QuantumCircuit,
    seed: int = 0,
    scale: float = 0.05,
    entangler_offset: float = 0.0,
) -> np.ndarray:
    """Small random start, with the entangler parameters deliberately offset.

    The offset is the fix for a real conflict (RESEARCH_LOG Result 19): a
    controlled-rotation entangler preserves the reference state at theta = 0
    precisely *because* it is the identity there -- and that is exactly what
    makes every gradient vanish, since each layer then sees a computational
    basis state and Brillouin's theorem applies.  Reference preservation and
    non-zero gradient cannot both come from the entangler at zero.

    Offsetting only the entangler parameters breaks the stationarity while
    keeping the state close to the reference.  Measured (Result 20):

        offset   H2 overlap  |grad|   H2 error   H4 error
        0.0      1.0000      0.000    2.0e-2     4.2e-2     (stuck at HF)
        0.1      0.9988      0.092    9.1e-11    4.2e-2
        0.5      0.9689      0.435    2.5e-9     2.0e-2
        pi       0.0000      0.000    7.9e-1     3.3e+0     (stationary again)

    Note both 0 *and* pi are stationary points, so a useful offset has to be
    genuinely interior.

    **The default is 0.0 anyway**, because the fix does not survive the regime
    it was meant for.  Those numbers come from noiseless BFGS, which can exploit
    a small gradient.  Under shot noise with COBYLA the offset *hurts*: on H2 at
    a 200k budget it gives 2.4e-2 against 3.3e-3 for a zero offset, because a
    gradient-free optimiser cannot use the gradient it buys and only pays for
    starting further from the reference (overlap 0.97 rather than 1.00).

    So: set it when using a gradient-based optimiser, leave it at zero
    otherwise.  The underlying conflict is not solved, only relocated.

    Has no effect on an ansatz with fixed CX entanglers, which has no entangler
    parameters and gets its gradient from the entanglers acting unconditionally.
    """
    rng = np.random.default_rng(seed)
    point = rng.normal(0.0, scale, size=ansatz.num_parameters)
    for index in (ansatz.metadata or {}).get("entangler_params", []):
        point[index] += entangler_offset
    return point
