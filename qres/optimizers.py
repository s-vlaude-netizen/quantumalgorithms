"""Classical optimisers for variational algorithms, charged by shot count.

An optimiser for a variational quantum algorithm is really a *shot budget
allocator*: its job is to spend as few circuit executions as possible reaching
a given energy.  Judged that way, the textbook choices are poor -- COBYLA and
SPSA both request a fixed, hand-tuned number of shots per evaluation, which is
far too many early (when any rough gradient direction will do) and often too
few late (when the optimiser is trying to resolve differences below the shot
noise floor).

The adaptive optimisers here decide the shot count from the measured gradient
signal-to-noise instead.  All of them report through the estimator's ledger, so
"which optimiser is cheapest" is a question with a numerical answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class OptimizeResult:
    x: np.ndarray
    fun: float
    nit: int
    #: energy trace, one entry per iteration
    history: list[float] = field(default_factory=list)
    #: cumulative shots at each iteration -- the x-axis of every plot we care about
    shot_history: list[int] = field(default_factory=list)
    message: str = ""
    extra: dict = field(default_factory=dict)


class BudgetExceeded(Exception):
    """Raised when an optimiser has spent its shot budget.

    Enforcing the budget inside the oracle rather than by tuning ``maxiter``
    per optimiser is what makes a comparison across optimisers fair: every
    configuration is stopped at exactly the same number of circuit executions,
    whatever it chose to spend them on.
    """


class EnergyOracle:
    """Wraps a ShotEstimator into the callable an optimiser wants."""

    def __init__(self, estimator, default_shots: int = 4096, budget: int | None = None):
        self.estimator = estimator
        self.default_shots = default_shots
        self.budget = budget
        self.calls = 0
        self.best_value = np.inf
        self.best_params: np.ndarray | None = None
        #: (cumulative_shots, value) at every call -- the raw record
        self.trace: list[tuple[int, float]] = []

    def _check_budget(self) -> None:
        if self.budget is not None and self.shots_used >= self.budget:
            raise BudgetExceeded(f"spent {self.shots_used} of {self.budget} shots")

    def __call__(self, params, shots: int | None = None) -> float:
        return self.with_error(params, shots)[0]

    def with_error(self, params, shots: int | None = None) -> tuple[float, float]:
        self._check_budget()
        self.calls += 1
        res = self.estimator.estimate(params, shots or self.default_shots)
        if res.value < self.best_value:
            self.best_value = res.value
            self.best_params = np.asarray(params, dtype=float).copy()
        self.trace.append((self.shots_used, res.value))
        return res.value, res.stderr

    @property
    def shots_used(self) -> int:
        return self.estimator.ledger.shots


def estimate_horizon(
    oracle: EnergyOracle, maxiter: int, shots: int, evals_per_iter: int = 2
) -> int:
    """How many iterations this run will actually get.

    Any optimiser with a decaying gain schedule needs this.  Under a shot
    budget the true stopping point is ``budget / (evals_per_iter * shots)``,
    which can be orders of magnitude below the ``maxiter`` sentinel a
    budget-limited driver passes in.
    """
    if oracle.budget is None:
        return max(1, maxiter)
    remaining = max(0, oracle.budget - oracle.shots_used)
    affordable = remaining // max(1, evals_per_iter * max(1, shots))
    return max(1, min(maxiter, int(affordable)))


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def scipy_minimize(
    oracle: EnergyOracle,
    x0: np.ndarray,
    method: str = "COBYLA",
    maxiter: int = 200,
    shots: int | None = None,
    callback=None,
) -> OptimizeResult:
    """Any gradient-free scipy method at a fixed shot count per evaluation."""
    from scipy.optimize import minimize

    history: list[float] = []
    shot_history: list[int] = []

    def f(x):
        val = oracle(x, shots)
        history.append(val)
        shot_history.append(oracle.shots_used)
        if callback:
            callback(x, val)
        return val

    res = minimize(f, np.asarray(x0, dtype=float), method=method, options={"maxiter": maxiter})
    return OptimizeResult(
        x=np.asarray(res.x),
        fun=float(res.fun),
        nit=int(getattr(res, "nit", len(history))),
        history=history,
        shot_history=shot_history,
        message=str(getattr(res, "message", "")),
    )


def spsa(
    oracle: EnergyOracle,
    x0: np.ndarray,
    maxiter: int = 150,
    shots: int | None = None,
    *,
    a: float | None = None,
    c: float = 0.1,
    alpha: float = 0.602,
    gamma: float = 0.101,
    calibrate: bool = True,
    seed: int = 0,
    callback=None,
) -> OptimizeResult:
    """Simultaneous perturbation stochastic approximation.

    Two energy evaluations per iteration regardless of dimension, which is why
    it is the default for noisy VQE.  ``a`` is calibrated from the observed
    gradient magnitude when not given -- an uncalibrated SPSA is the single most
    common reason a VQE benchmark looks worse than it should.

    The gain schedule is anchored to the number of iterations the run will
    *actually* take (:func:`estimate_horizon`), not to ``maxiter``.  Under a
    shot budget ``maxiter`` is only a sentinel, and anchoring ``A`` to it
    freezes the step size at a near-constant tiny value for the whole run --
    a silent corruption that looks exactly like a shot-allocation result.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    horizon = estimate_horizon(oracle, maxiter, shots or oracle.default_shots, evals_per_iter=2)
    A = max(1.0, horizon * 0.1)

    if a is None:
        if calibrate:
            deltas = rng.choice([-1.0, 1.0], size=(min(12, maxiter), n))
            mags = []
            for d in deltas:
                fp = oracle(x + c * d, shots)
                fm = oracle(x - c * d, shots)
                mags.append(abs(fp - fm) / (2 * c))
            typical = float(np.mean(mags)) or 1.0
            # aim for a first step of ~0.1 rad
            a = 0.1 * (A + 1) ** alpha / max(typical, 1e-8)
        else:
            a = 0.2

    history: list[float] = []
    shot_history: list[int] = []
    best_x, best_f = x.copy(), np.inf

    for k in range(maxiter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=n)
        fp = oracle(x + ck * delta, shots)
        fm = oracle(x - ck * delta, shots)
        ghat = (fp - fm) / (2 * ck) * delta
        x = x - ak * ghat

        val = 0.5 * (fp + fm)
        history.append(val)
        shot_history.append(oracle.shots_used)
        if val < best_f:
            best_f, best_x = val, x.copy()
        if callback:
            callback(x, val)

    final = oracle(best_x, shots)
    return OptimizeResult(
        x=best_x,
        fun=float(min(final, best_f)),
        nit=maxiter,
        history=history,
        shot_history=shot_history,
        message=f"spsa a={a:.4g}",
    )


# --------------------------------------------------------------------------
# Adaptive shot allocation across iterations
# --------------------------------------------------------------------------


def icans(
    oracle: EnergyOracle,
    x0: np.ndarray,
    maxiter: int = 200,
    *,
    learning_rate: float = 0.2,
    lipschitz: float | None = None,
    min_shots: int = 8,
    max_shots: int = 8192,
    shot_budget: int | None = None,
    mu: float = 0.99,
    b: float = 1e-6,
    seed: int = 0,
    callback=None,
) -> OptimizeResult:
    """iCANS-style gradient descent with per-parameter adaptive shot counts.

    At each step the shots spent on parameter i are set from the running
    estimates of its gradient ``g_i`` and gradient variance ``sigma_i^2``:

        s_i = ceil( 2 L alpha sigma_i^2 / ((2 - L alpha) (g_i^2 + b)) )

    so a parameter whose gradient is well-resolved gets almost no shots and one
    whose gradient is buried in noise gets many.  Early iterations end up
    costing tens of shots where a fixed-shot optimiser would spend thousands.

    Reference: Kuebler et al., "An Adaptive Optimizer for Measurement-Frugal
    Variational Algorithms" (Quantum 4, 263, 2020).
    """
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    if lipschitz is None:
        # sum |c_i| bounds the second derivative of <H> in any single parameter
        lipschitz = float(np.abs(np.asarray(oracle.estimator.hamiltonian.coeffs).real).sum())
    alpha = learning_rate
    # the update rule is only a descent step while L*alpha < 2
    alpha = min(alpha, 1.9 / max(lipschitz, 1e-12))

    chi = np.zeros(n)  # running gradient estimate
    xi = np.zeros(n)  # running gradient variance estimate
    shots_i = np.full(n, min_shots, dtype=int)
    gamma_acc = 0.0

    history: list[float] = []
    shot_history: list[int] = []
    start_shots = oracle.shots_used

    for k in range(maxiter):
        if shot_budget is not None and oracle.shots_used - start_shots >= shot_budget:
            break

        grad = np.zeros(n)
        grad_var = np.zeros(n)
        for i in range(n):
            s = int(np.clip(shots_i[i], min_shots, max_shots))
            plus = x.copy()
            plus[i] += np.pi / 2
            minus = x.copy()
            minus[i] -= np.pi / 2
            fp, ep = oracle.with_error(plus, s)
            fm, em = oracle.with_error(minus, s)
            grad[i] = (fp - fm) / 2
            # variance of the parameter-shift estimate, scaled back to per-shot
            grad_var[i] = max((ep**2 + em**2) / 4, 1e-12) * s

        gamma_acc = mu * gamma_acc + (1 - mu)
        chi = mu * chi + (1 - mu) * grad
        xi = mu * xi + (1 - mu) * grad_var
        chi_hat = chi / max(gamma_acc, 1e-12)
        xi_hat = xi / max(gamma_acc, 1e-12)

        x = x - alpha * grad

        # s_i = 2 L a sigma_i^2 / ((2 - L a) (g_i^2 + b)).  The denominator must
        # contain only the *gradient* magnitude: adding a sigma^2/s term makes
        # the rule self-limiting -- s appears on both sides and pins itself to
        # the floor, so every gradient stays pure noise and the optimiser random
        # walks.  (Measured: 535 iterations all at the 8-shot floor, energy
        # bouncing over a 0.5 Ha range and never converging.)
        denom = (2 - lipschitz * alpha) * (chi_hat**2 + b)
        shots_i = np.ceil(
            2 * lipschitz * alpha * xi_hat / np.maximum(denom, 1e-30)
        ).astype(int)
        shots_i = np.clip(shots_i, min_shots, max_shots)

        val = oracle(x, min(max_shots, int(np.mean(shots_i)) * 4))
        history.append(val)
        shot_history.append(oracle.shots_used)
        if callback:
            callback(x, val)

    final = oracle(x, max_shots)
    return OptimizeResult(
        x=x,
        fun=float(final),
        nit=len(history),
        history=history,
        shot_history=shot_history,
        message=f"icans alpha={alpha:.4g} L={lipschitz:.4g}",
        extra={"final_shots_per_param": shots_i.tolist()},
    )


def adaptive_spsa(
    oracle: EnergyOracle,
    x0: np.ndarray,
    maxiter: int = 10_000,
    *,
    resolution: float = 1.0,
    min_shots: int = 32,
    max_shots: int = 16_384,
    start_shots: int | None = None,
    max_growth: float = 1.6,
    a: float | None = None,
    c: float = 0.15,
    alpha: float = 0.602,
    gamma: float = 0.101,
    smoothing: float = 0.8,
    seed: int = 0,
    callback=None,
) -> OptimizeResult:
    """SPSA whose shot count tracks the energy change it needs to resolve.

    Two observations motivate this:

    * iCANS-style per-parameter shot adaptation needs ``2n`` circuit
      evaluations per step.  SPSA needs two, whatever the dimension, so the
      same adaptive idea is far cheaper if it can be made to work on a *single*
      scalar.
    * A fixed shot count is wrong at both ends of a run.  Early on any rough
      descent direction suffices and the shots are wasted; late on the
      optimiser is trying to resolve energy differences below its own noise
      floor and stalls.

    **What not to control on.**  The obvious rule -- hold the *gradient*
    signal-to-noise at some target ``tau``, giving
    ``s* = tau^2 sigma^2 / (2 c^2 g^2)`` -- fails, and measurably so.  The
    gradient genuinely goes to zero at the minimum, so no finite shot count
    ever reaches the target; the rule saturates at ``max_shots`` within a few
    iterations and degenerates into fixed-shot SPSA with far too few steps.
    Measured on H2: 1.1e-2 Ha, worse than plain SPSA's 6.5e-3.

    **What to control on instead.**  Match the shot noise on the energy to the
    step-to-step energy change the optimiser is currently making:

        s* = (sigma_E / (resolution * |dE|_recent))^2

    Both sides shrink together as the run converges, so the rule escalates
    smoothly with progress instead of diverging.  Growth is additionally
    rate-limited to ``max_growth`` per iteration, so one anomalously small
    ``|dE|`` cannot spend the remaining budget in a single step.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    shots = int(start_shots or min_shots)
    # Re-estimated every iteration: the shot count moves, so the number of
    # iterations still affordable moves with it, and the gain schedule has to
    # follow or the step size is calibrated to a horizon that never arrives.
    horizon = estimate_horizon(oracle, maxiter, shots, evals_per_iter=2)
    A = max(1.0, horizon * 0.1)

    if a is None:
        # one cheap calibration sweep at the starting shot count
        mags = []
        for _ in range(6):
            d = rng.choice([-1.0, 1.0], size=n)
            fp = oracle(x + c * d, shots)
            fm = oracle(x - c * d, shots)
            mags.append(abs(fp - fm) / (2 * c))
        typical = float(np.mean(mags)) or 1.0
        a = 0.1 * (A + 1) ** alpha / max(typical, 1e-8)

    history: list[float] = []
    shot_history: list[int] = []
    shots_trace: list[int] = []
    best_x, best_f = x.copy(), np.inf
    smooth_dE = None
    prev_val = None

    for k in range(maxiter):
        A = max(1.0, (k + estimate_horizon(oracle, maxiter, shots, 2)) * 0.1)
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=n)

        fp, ep = oracle.with_error(x + ck * delta, shots)
        fm, em = oracle.with_error(x - ck * delta, shots)

        g = (fp - fm) / (2 * ck)
        x = x - ak * g * delta

        val = 0.5 * (fp + fm)
        # per-shot energy variance, recovered from the reported standard errors
        sigma2 = max((ep**2 + em**2) / 2, 1e-18) * shots

        if prev_val is not None:
            dE = abs(val - prev_val)
            smooth_dE = dE if smooth_dE is None else smoothing * smooth_dE + (1 - smoothing) * dE
            if smooth_dE > 1e-12:
                needed = sigma2 / (resolution * smooth_dE) ** 2
                shots = int(
                    np.clip(round(needed), min_shots, min(max_shots, int(shots * max_growth) + 1))
                )
        prev_val = val

        history.append(val)
        shot_history.append(oracle.shots_used)
        shots_trace.append(shots)
        if val < best_f:
            best_f, best_x = val, x.copy()
        if callback:
            callback(x, val)

    final = oracle(best_x, max_shots)
    return OptimizeResult(
        x=best_x,
        fun=float(min(final, best_f)),
        nit=len(history),
        history=history,
        shot_history=shot_history,
        message=f"adaptive_spsa a={a:.4g} res={resolution}",
        extra={"shots_trace": shots_trace},
    )


OPTIMIZERS: dict[str, Callable] = {
    "cobyla": lambda o, x0, **kw: scipy_minimize(o, x0, method="COBYLA", **kw),
    "powell": lambda o, x0, **kw: scipy_minimize(o, x0, method="Powell", **kw),
    "nelder-mead": lambda o, x0, **kw: scipy_minimize(o, x0, method="Nelder-Mead", **kw),
    "spsa": spsa,
    "icans": icans,
    "aspsa": adaptive_spsa,
}
