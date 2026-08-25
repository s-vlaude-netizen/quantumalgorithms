"""Realistic noise environments built from IBM device calibration data.

Everything in this package is benchmarked against *measured* device parameters
rather than a hand-tuned depolarising channel, so that a reported improvement
has a chance of surviving contact with hardware.  Three fidelity tiers are
provided so an experiment can be run cheaply during development and expensively
for the record:

``ideal``        statevector / sampling only, no noise -- for correctness checks
``device``       Aer noise model derived from a FakeBackendV2's calibration
``device_heavy`` same, plus readout error and larger shot counts by default

The fake backends ship with snapshots of real T1/T2, gate errors and readout
assignment matrices from IBM Quantum systems.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Backends chosen to span the sizes we care about.  27q Eagle-era and 127q
# Eagle devices are the realistic NISQ regime; the small ones keep the
# development loop fast.
BACKEND_TIERS = {
    "small": "fake_manila",      # 5q, high error rates -- stress test
    "medium": "fake_kolkata",    # 27q Falcon r5.11
    "large": "fake_brisbane",    # 127q Eagle r3
    "heron": "fake_torino",      # 133q Heron r1, best available fidelities
}


@functools.lru_cache(maxsize=None)
def get_fake_backend(name: str):
    """Load a FakeBackendV2 by name (accepts tier aliases)."""
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    name = BACKEND_TIERS.get(name, name)
    provider = FakeProviderForBackendV2()
    for backend in provider.backends():
        if backend.name == name:
            return backend
    available = sorted(b.name for b in provider.backends())
    raise ValueError(f"unknown backend {name!r}; available: {available}")


@dataclass
class NoiseEnvironment:
    """A simulator plus the transpilation target it is calibrated to.

    Circuits must be transpiled to ``backend`` before running so that the noise
    model's gate-level errors actually apply; :meth:`prepare` does that and
    caches nothing (callers should cache per-ansatz).
    """

    name: str
    simulator: AerSimulator
    backend: Any | None
    seed: int = 1234

    @property
    def target(self):
        return self.backend.target if self.backend is not None else None

    @property
    def coupling_map(self):
        return self.backend.coupling_map if self.backend is not None else None

    def prepare(self, circuit, optimization_level: int = 3):
        """Transpile ``circuit`` into the ISA of this environment."""
        if self.backend is None:
            return transpile(circuit, self.simulator, optimization_level=optimization_level)
        return transpile(
            circuit,
            self.backend,
            optimization_level=optimization_level,
            seed_transpiler=self.seed,
        )

    def run(self, circuits, shots: int):
        """Execute ISA circuits, returning a list of count dicts."""
        result = self.simulator.run(circuits, shots=shots, seed_simulator=self.seed).result()
        if isinstance(circuits, list):
            return [result.get_counts(i) for i in range(len(circuits))]
        return result.get_counts()


def ideal_environment(seed: int = 1234, method: str = "automatic") -> NoiseEnvironment:
    sim = AerSimulator(method=method, seed_simulator=seed)
    return NoiseEnvironment(name="ideal", simulator=sim, backend=None, seed=seed)


def device_environment(
    tier: str = "medium",
    seed: int = 1234,
    *,
    readout_error: bool = True,
    thermal_relaxation: bool = True,
    scale: float = 1.0,
) -> NoiseEnvironment:
    """Aer simulator carrying a real device's calibrated noise model.

    ``scale`` multiplies all gate error probabilities, letting us sweep "how
    good does hardware have to get before this algorithm wins?" while keeping
    the *structure* of the real error channels.
    """
    backend = get_fake_backend(tier)
    noise_model = NoiseModel.from_backend(
        backend,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )
    if scale != 1.0:
        noise_model = _scale_noise_model(backend, scale, readout_error, thermal_relaxation)
    sim = AerSimulator.from_backend(backend, noise_model=noise_model, seed_simulator=seed)
    label = f"{backend.name}" + (f"@{scale:g}x" if scale != 1.0 else "")
    return NoiseEnvironment(name=label, simulator=sim, backend=backend, seed=seed)


def _scale_noise_model(backend, scale, readout_error, thermal_relaxation):
    """Rebuild a noise model with all error rates multiplied by ``scale``.

    Works on the backend's target properties rather than on the assembled
    channels, which keeps the depolarising/thermal split physically consistent.
    """
    from qiskit.transpiler import Target

    target: Target = backend.target
    scaled = Target(
        num_qubits=target.num_qubits,
        dt=target.dt,
        granularity=target.granularity,
        min_length=target.min_length,
        pulse_alignment=target.pulse_alignment,
        acquire_alignment=target.acquire_alignment,
        # T1/T2 must be carried over or Aer cannot build the thermal
        # relaxation channels and the whole noise model construction fails.
        qubit_properties=target.qubit_properties,
    )
    for name, qargs_map in target.items():
        props = {}
        for qargs, prop in qargs_map.items():
            if prop is None:
                props[qargs] = None
                continue
            new_error = None if prop.error is None else min(1.0, prop.error * scale)
            props[qargs] = type(prop)(duration=prop.duration, error=new_error)
        scaled.add_instruction(target.operation_from_name(name), props, name=name)

    class _Shim:
        """Minimal BackendV2-like view carrying the scaled target."""

        def __init__(self, inner, tgt):
            self._inner = inner
            self.target = tgt
            self.name = inner.name
            self.num_qubits = tgt.num_qubits

        def __getattr__(self, item):
            return getattr(self._inner, item)

    return NoiseModel.from_backend(
        _Shim(backend, scaled),
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )


def make_environment(spec: str, seed: int = 1234) -> NoiseEnvironment:
    """Build an environment from a short spec string.

    ``"ideal"``            -> noiseless
    ``"medium"``           -> fake_kolkata calibration
    ``"large@0.5x"``       -> fake_brisbane with halved gate errors
    """
    if spec == "ideal":
        return ideal_environment(seed=seed)
    scale = 1.0
    if "@" in spec:
        spec, _, tail = spec.partition("@")
        scale = float(tail.rstrip("xX"))
    return device_environment(spec, seed=seed, scale=scale)


def describe_environment(env: NoiseEnvironment) -> dict[str, Any]:
    """Summarise the error rates actually in force -- goes into every result."""
    if env.backend is None:
        return {"name": env.name, "noise": "none"}
    target = env.backend.target
    one_q, two_q, readout = [], [], []
    for name, qargs_map in target.items():
        if name in ("delay", "reset", "barrier"):
            continue
        for qargs, prop in qargs_map.items():
            if prop is None or prop.error is None:
                continue
            if name == "measure":
                readout.append(prop.error)
            elif qargs is not None and len(qargs) == 2:
                two_q.append(prop.error)
            elif qargs is not None and len(qargs) == 1:
                one_q.append(prop.error)

    def _stats(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {"median": xs[len(xs) // 2], "min": xs[0], "max": xs[-1], "n": len(xs)}

    return {
        "name": env.name,
        "backend": env.backend.name,
        "num_qubits": target.num_qubits,
        "one_qubit_error": _stats(one_q),
        "two_qubit_error": _stats(two_q),
        "readout_error": _stats(readout),
    }
