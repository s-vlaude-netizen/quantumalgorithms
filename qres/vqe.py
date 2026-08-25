"""VQE driver: problem + ansatz + noise + optimiser -> a scored result."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ansatz import build_ansatz, initial_point
from .estimator import ShotEstimator
from .noise import NoiseEnvironment, describe_environment, make_environment
from .optimizers import OPTIMIZERS, BudgetExceeded, EnergyOracle, OptimizeResult
from .resources import ResourceLedger


@dataclass
class VQERun:
    """Everything needed to reproduce and score one VQE run."""

    problem_name: str
    config: dict[str, Any]
    energy: float
    #: energy of the optimiser's final parameters with no shot noise and no
    #: device noise -- separates "the ansatz cannot represent it" from
    #: "the noise stopped us finding it"
    noiseless_energy: float
    reference_energy: float
    ledger: dict[str, Any]
    history: list[float] = field(default_factory=list)
    shot_history: list[int] = field(default_factory=list)
    grouping: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    wall_seconds: float = 0.0
    parameters: list[float] = field(default_factory=list)
    message: str = ""

    @property
    def error(self) -> float:
        return abs(self.energy - self.reference_energy)

    @property
    def noiseless_error(self) -> float:
        return abs(self.noiseless_energy - self.reference_energy)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "problem": self.problem_name,
            "config": self.config,
            "energy": self.energy,
            "noiseless_energy": self.noiseless_energy,
            "reference_energy": self.reference_energy,
            "error": self.error,
            "noiseless_error": self.noiseless_error,
            "ledger": self.ledger,
            "grouping": self.grouping,
            "environment": self.environment,
            "wall_seconds": self.wall_seconds,
            "message": self.message,
            "history": self.history,
            "shot_history": self.shot_history,
        }
        return d

    def line(self) -> str:  # pragma: no cover - display only
        hw = self.ledger.get("hardware_seconds", 0.0)
        return (
            f"{self.config.get('label', ''):28s} "
            f"E={self.energy:+.6f} err={self.error:.2e} "
            f"(noiseless {self.noiseless_error:.2e})  "
            f"shots={self.ledger['shots']:>10,} "
            f"circ={self.ledger['circuits']:>7,} "
            f"hw={hw:8.2f}s wall={self.wall_seconds:6.1f}s"
        )


def run_vqe(
    problem,
    *,
    ansatz: str = "hea:2",
    environment: str | NoiseEnvironment = "ideal",
    grouping: str = "commuting",
    reference_params=None,
    allocation: str = "adaptive",
    optimizer: str = "cobyla",
    shots: int = 4096,
    maxiter: int = 150,
    shot_budget: int | None = None,
    seed: int = 0,
    label: str | None = None,
    optimizer_kwargs: dict | None = None,
) -> VQERun:
    """Run one VQE configuration end to end.

    ``shot_budget`` caps the total circuit executions.  When set it is enforced
    inside the oracle, so every optimiser is stopped at the same cost and the
    comparison between them is about how well they spent the budget rather than
    about how much of it they were handed.
    """
    t_start = time.perf_counter()
    env = make_environment(environment, seed=seed) if isinstance(environment, str) else environment

    circuit = build_ansatz(ansatz, problem, env)
    ledger = ResourceLedger()
    estimator = ShotEstimator(
        problem.hamiltonian,
        circuit,
        env,
        grouping=grouping,
        allocation=allocation,
        ledger=ledger,
        reference_params=reference_params,
    )
    oracle = EnergyOracle(estimator, default_shots=shots, budget=shot_budget)
    x0 = initial_point(circuit, seed=seed)

    kwargs = dict(optimizer_kwargs or {})
    if optimizer in ("cobyla", "powell", "nelder-mead", "spsa"):
        kwargs.setdefault("shots", shots)
    kwargs.setdefault("maxiter", maxiter)
    if optimizer in ("spsa", "icans"):
        kwargs.setdefault("seed", seed)

    stopped_early = False
    try:
        result = OPTIMIZERS[optimizer](oracle, x0, **kwargs)
    except BudgetExceeded as exc:
        stopped_early = True
        x_best = oracle.best_params if oracle.best_params is not None else x0
        result = OptimizeResult(
            x=np.asarray(x_best),
            fun=float(oracle.best_value),
            nit=oracle.calls,
            history=[v for _, v in oracle.trace],
            shot_history=[s for s, _ in oracle.trace],
            message=f"budget exhausted: {exc}",
        )

    noiseless = estimator.exact(result.x)
    wall = time.perf_counter() - t_start

    config = {
        "ansatz": ansatz,
        "grouping": grouping,
        "allocation": allocation,
        "optimizer": optimizer,
        "shots": shots,
        "maxiter": maxiter,
        "seed": seed,
        "shot_budget": shot_budget,
        "stopped_early": stopped_early,
        "num_parameters": circuit.num_parameters,
        "label": label or f"{optimizer}/{grouping}/{allocation}",
    }
    return VQERun(
        problem_name=problem.name,
        config=config,
        energy=float(result.fun),
        noiseless_energy=float(noiseless),
        reference_energy=float(problem.fci_energy),
        ledger=ledger.to_dict(),
        history=[float(h) for h in result.history],
        shot_history=[int(s) for s in result.shot_history],
        grouping=estimator.report,
        environment=describe_environment(env),
        wall_seconds=wall,
        parameters=[float(v) for v in result.x],
        message=result.message,
    )
