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


def stochastic_parameter_shift(
    oracle: EnergyOracle,
    x0: np.ndarray,
    maxiter: int = 100_000,
    *,
    coordinates: int = 1,
    shots: int = 256,
    learning_rate: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.99,
    epsilon: float = 1e-8,
    seed: int = 0,
    callback=None,
) -> OptimizeResult:
    """Adam on parameter-shift gradients of a random coordinate subset per step.

    The reason gradient methods lost in session 1 is arithmetic, not principle.
    A full parameter-shift gradient costs ``2n`` circuit evaluations, so H2's
    12-parameter ansatz at 2048 shots and a 200k budget affords **four**
    gradient steps.  iCANS, which needs the same sweep, measured at ~3.5 h for
    8 seeds of H4 and was abandoned.

    Two knobs make the exchange rate tunable instead of fixed:

    * ``coordinates`` -- update ``k`` randomly chosen parameters per step rather
      than all of them, at ``2k`` evaluations. The estimator stays unbiased in
      expectation (it is randomised coordinate descent), and ``k = 1`` buys
      ``n`` times as many steps.
    * ``shots`` -- each parameter-shift evaluation is a *difference*, so it
      tolerates far more shot noise than an absolute energy does. Spending 256
      shots where the fixed-shot optimisers spend 2048 buys another 8×.

    Adam rather than plain descent because the gradient scale varies by orders
    of magnitude across parameters and a single hand-set step size does not fit
    both ends.  Adam's moments are kept per-coordinate and updated only for the
    coordinates actually sampled, so a rarely-visited parameter does not have
    its history decayed away by steps that never looked at it.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    k = max(1, min(coordinates, n))

    m = np.zeros(n)
    v = np.zeros(n)
    visits = np.zeros(n, dtype=int)

    history: list[float] = []
    shot_history: list[int] = []
    best_x, best_f = x.copy(), np.inf

    for _ in range(maxiter):
        picks = rng.choice(n, size=k, replace=False)
        for i in picks:
            plus, minus = x.copy(), x.copy()
            plus[i] += np.pi / 2
            minus[i] -= np.pi / 2
            f_plus = oracle(plus, shots)
            f_minus = oracle(minus, shots)
            gradient = (f_plus - f_minus) / 2

            visits[i] += 1
            m[i] = beta1 * m[i] + (1 - beta1) * gradient
            v[i] = beta2 * v[i] + (1 - beta2) * gradient**2
            # bias correction uses this coordinate's own visit count
            m_hat = m[i] / (1 - beta1 ** visits[i])
            v_hat = v[i] / (1 - beta2 ** visits[i])
            x[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            value = 0.5 * (f_plus + f_minus)
            history.append(value)
            shot_history.append(oracle.shots_used)
            if value < best_f:
                best_f, best_x = value, x.copy()
        if callback:
            callback(x, history[-1] if history else np.inf)

    return OptimizeResult(
        x=x,
        fun=float(best_f),
        nit=len(history),
        history=history,
        shot_history=shot_history,
        message=f"sps k={k} shots={shots} lr={learning_rate}",
        extra={"visits": visits.tolist()},
    )


def shot_ladder(
    oracle: EnergyOracle,
    x0: np.ndarray,
    maxiter: int = 100_000,
    *,
    inner: str = "COBYLA",
    levels: int | None = None,
    growth: float = 6.0,
    evals_per_level: int | None = None,
    min_useful_shots: int = 4096,
    rhobeg: float = 1.0,
    seed: int = 0,
    callback=None,
) -> OptimizeResult:
    """Restart a model-based optimiser repeatedly with escalating shot counts.

    The diagnosis this implements (RESEARCH_LOG Result 26): near the optimum the
    energy *differences* a model-based method must resolve shrink quadratically
    in the distance to the minimum, while shot noise stays flat.  No fixed shot
    count works -- early on any rough model suffices and the shots are wasted;
    late on the model is fitting pure noise.  Measured on H2, COBYLA reaches
    4.96e-5 on exact energies but only 2.68e-3 at 524 288 shots per evaluation,
    a noise level already *below* chemical accuracy.  Precision has to arrive on
    a schedule, not all at once.

    **The shot schedule is derived from the budget, not chosen independently.**
    A geometric split of the budget across levels whose shot counts also grow
    geometrically starves the expensive levels: with 4 levels growing 8x on a
    200k budget, the top level needs 262 144 shots per evaluation and is handed
    175k in total -- not one evaluation.  Instead, fix the evaluations per level
    at ``E`` and solve

        s_0 * E * (g^L - 1) / (g - 1) = budget

    so every level gets ``E`` evaluations and the whole budget is spent.
    ``E`` defaults to ``2(n+1)``, since COBYLA needs ``n+1`` evaluations just to
    build its initial simplex.

    ``rhobeg`` stays *large* on every restart, which is counter-intuitive and
    measured: restarting from a converged point with ``rhobeg = 1.0`` improved
    the error 9x (2.0e-2 to 2.2e-3), while 0.3, 0.1 and 0.03 barely moved it.
    A small trust region re-converges inside the same noise-induced basin; a
    large one escapes it and re-explores with the newly precise evaluations.

    ``levels`` defaults to the most the budget can actually afford, since the
    number of affordable rungs grows with it and over-laddering is *worse* than
    not laddering: on H2 at 800k, four levels measured 2.48x worse than SPSA
    (0/12 wins, p = 0.000), while at 12.8M three levels measured 2.1x better
    (12/0, p = 0.000).  The rule keeps the bottom rung at ``min_useful_shots``
    or above, and reproduces the measured best level count at every budget
    tested (800k -> 2, 3.2M -> 2, 12.8M -> 3).

    **This method needs budget headroom and is the wrong choice without it.**
    Measured against SPSA on H2 (16 seeds, paired):

        budget       ratio   W/L     p
        200 000      1.710   3/13    0.021   <- significantly WORSE
        800 000      0.825   10/6    0.454
        3 200 000    0.594   11/5    0.210
        12 800 000   0.465   12/4    0.077

    The crossover sits near ``budget ~ 8 * evals_per_level * min_useful_shots``.
    Below it there is no room for two well-fed rungs and SPSA is the better
    choice; ``levels`` falls to 1 there, which is plain COBYLA and worse than
    SPSA -- so the caller should be choosing SPSA, not this.
    """
    from scipy.optimize import minimize

    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    evals = evals_per_level if evals_per_level is not None else 2 * (n + 1)

    remaining = (oracle.budget - oracle.shots_used) if oracle.budget else None

    def span(count: int) -> float:
        return (growth**count - 1) / (growth - 1)

    if levels is None:
        if remaining is None:
            levels = 3
        else:
            # the most rungs whose bottom one still gets min_useful_shots
            affordable = remaining / (evals * min_useful_shots)
            levels = 1
            while span(levels + 1) <= affordable and levels < 8:
                levels += 1
    if remaining is None:
        schedule = [int(min_useful_shots * growth**i) for i in range(levels)]
    else:
        base = max(1.0, remaining / (evals * span(levels)))
        schedule = [max(1, int(base * growth**i)) for i in range(levels)]

    history: list[float] = []
    shot_history: list[int] = []
    levels_run = []

    for level, shots in enumerate(schedule):
        spent_at_start = oracle.shots_used
        level_budget = evals * shots

        def f(y):
            if oracle.shots_used - spent_at_start >= level_budget:
                raise _LevelDone()
            value = oracle(y, shots)
            history.append(value)
            shot_history.append(oracle.shots_used)
            if callback:
                callback(y, value)
            return value

        try:
            result = minimize(f, x, method=inner, options={"maxiter": 10**6, "rhobeg": rhobeg})
            x = np.asarray(result.x, dtype=float)
        except _LevelDone:
            if oracle.best_params is not None:
                x = oracle.best_params.copy()
        levels_run.append(
            {"level": level, "shots": shots, "spent": oracle.shots_used - spent_at_start}
        )

    return OptimizeResult(
        x=x,
        fun=float(oracle.best_value),
        nit=len(history),
        history=history,
        shot_history=shot_history,
        message=f"ladder {inner} L={levels} g={growth} E={evals} schedule={schedule}",
        extra={"levels": levels_run, "schedule": schedule},
    )


class _LevelDone(Exception):
    """Internal: one rung of the shot ladder has spent its share."""


#: Measured initial trust radii, by ansatz family.  There is no automatic way
#: to get these -- see :func:`estimate_trust_radius` for three attempts that
#: failed and why -- so they are set explicitly and the measurements are cited.
#:
#: The physical reason they differ: a hardware-efficient ansatz's parameters are
#: rotation angles with period 2*pi and no preferred scale, so a large restart
#: radius explores usefully.  A coupled-cluster ansatz's parameters are cluster
#: amplitudes, which are physically O(0.1); a 1-radian step leaves the region
#: where the ansatz means anything.
TRUST_RADIUS = {
    "hardware_efficient": 1.0,  # H2/hea:2, 9x better than 0.1 on restart (Result 26)
    "coupled_cluster": 0.1,  # H4/UCCSD, 2x better than 1.0 (Result 31)
}


def estimate_trust_radius(*_args, **_kwargs) -> float:
    """Not implemented, deliberately: three principled attempts all failed.

    Kept as a record so the next attempt does not repeat them.  Each probes the
    energy at a few radii along random directions and picks where some model
    assumption breaks; each fails for a different reason:

    1. **Target a step that moves the energy by ten times the estimator noise.**
       Measures the wrong quantity.  Right on H2 (picked 0.8 against a measured
       optimum of 1.0), wrong and unstable on H4 (0.2 or 0.8 across seeds
       against a measured 0.1) -- UCCSD's *noise* is large while its *parameter*
       scale is small, and the heuristic keys on the former.

    2. **Find where the quadratic model breaks**, using
       ``E(2r) - E(0) = 4 (E(r) - E(0))``.  Wrong model: at a general,
       non-stationary point the energy is *linear*-dominated at small radius, so
       the ratio is 2 rather than 4 and the test fails at every radius.  Both
       systems collapsed to the smallest.

    3. **Find where the linear model breaks** -- ratio 2, matched to COBYLA,
       which builds a linear approximation.  The right test, defeated by noise:
       at 8192 shots sigma is 2.7e-3 on H2 and 1.4e-2 on H4, which swamps the
       departure from linearity at small radius.  Both collapsed to 0.05 again,
       and 90k shots per estimate bought nothing.

    The common failure is that the signal (where a model stops holding) is
    smaller than the shot noise at any affordable probe cost.  Use
    :data:`TRUST_RADIUS` and choose per ansatz family.
    """
    raise NotImplementedError(
        "automatic trust-radius estimation does not work at these noise levels; "
        "use TRUST_RADIUS[<ansatz family>] and see this docstring for why"
    )


def multi_start(
    oracle: EnergyOracle,
    x0: np.ndarray,
    maxiter: int = 100_000,
    *,
    inner: Callable | None = None,
    starts: int = 4,
    audit_fraction: float = 0.1,
    spread: float = 0.1,
    seed: int = 0,
    callback=None,
    **inner_kwargs,
) -> OptimizeResult:
    """Split the budget across several starts and keep the best.

    Motivated by a measurement, not by hope: at 256M shots on H4 + UCCSD the
    *best* of eight seeds reached 1.77e-2 while the median stayed at 3.66e-2,
    and four times the budget moved the median by 4% (RESEARCH_LOG Result 37).
    That gap is optimiser variance, and running fewer shots more times is the
    obvious way to buy it -- if the selection can be done honestly.

    **Selection costs budget and is charged for it.**  On hardware there is no
    exact energy to rank candidates by, so ``audit_fraction`` of the budget is
    reserved to re-evaluate every candidate at equal precision and pick the
    lowest.  Ranking by each run's own best observed value instead would be
    free and wrong: that value is the minimum of many noisy draws and is biased
    low by roughly the run's own noise, so it would systematically favour the
    *noisiest* run rather than the best one.
    """
    from .optimizers import shot_ladder  # local import keeps the registry simple

    inner = inner or shot_ladder
    rng = np.random.default_rng(seed)
    total = (oracle.budget - oracle.shots_used) if oracle.budget else None

    audit_budget = int(total * audit_fraction) if total else 0
    per_start = (total - audit_budget) // starts if total else None

    candidates: list[np.ndarray] = []
    history: list[float] = []
    shot_history: list[int] = []

    for index in range(starts):
        start = np.asarray(x0, dtype=float) + (
            rng.normal(0.0, spread, size=len(x0)) if index else 0.0
        )
        limit = None if per_start is None else oracle.shots_used + per_start
        guard = _BudgetWindow(oracle, limit)
        try:
            result = inner(guard, start, maxiter=maxiter, **inner_kwargs)
            candidates.append(np.asarray(result.x, dtype=float))
            history.extend(result.history)
            shot_history.extend(result.shot_history)
        except (_WindowClosed, BudgetExceeded):
            candidates.append(
                oracle.best_params.copy() if oracle.best_params is not None else start
            )
        if callback:
            callback(candidates[-1], history[-1] if history else np.inf)

    # honest selection: equal shots to each candidate, pick the lowest
    audit_shots = max(1, audit_budget // max(1, len(candidates)))
    scores = []
    for candidate in candidates:
        try:
            scores.append(oracle(candidate, audit_shots))
        except BudgetExceeded:
            scores.append(np.inf)
    winner = int(np.argmin(scores))

    return OptimizeResult(
        x=candidates[winner],
        fun=float(scores[winner]),
        nit=len(history),
        history=history,
        shot_history=shot_history,
        message=f"multi_start starts={starts} audit={audit_fraction} winner={winner}",
        extra={"audit_scores": [float(s) for s in scores], "winner": winner},
    )


class _WindowClosed(Exception):
    """Internal: one start has spent its share of the budget."""


class _BudgetWindow:
    """Oracle view that stops after a per-start allowance.

    Wraps rather than mutates the real oracle so the global budget, the best
    point seen and the trace all stay on the original.
    """

    def __init__(self, oracle: EnergyOracle, limit: int | None):
        self._oracle = oracle
        self._limit = limit

    def _check(self):
        if self._limit is not None and self._oracle.shots_used >= self._limit:
            raise _WindowClosed()

    def __call__(self, params, shots: int | None = None) -> float:
        self._check()
        return self._oracle(params, shots)

    def with_error(self, params, shots: int | None = None):
        self._check()
        return self._oracle.with_error(params, shots)

    @property
    def budget(self):
        return self._limit

    @property
    def shots_used(self) -> int:
        return self._oracle.shots_used

    @property
    def best_params(self):
        return self._oracle.best_params

    @property
    def best_value(self):
        return self._oracle.best_value

    @property
    def default_shots(self) -> int:
        return self._oracle.default_shots

    @property
    def estimator(self):
        return self._oracle.estimator


def stochastic_trust_region(
    oracle: EnergyOracle,
    x0: np.ndarray,
    maxiter: int = 100_000,
    *,
    radius: float = 0.5,
    min_radius: float = 1e-3,
    max_radius: float = 2.0,
    kappa: float = 1.0,
    probes: int = 4,
    accept: float = 0.1,
    expand: float = 1.6,
    shrink: float = 0.5,
    min_shots: int = 256,
    max_shots: int = 4_194_304,
    seed: int = 0,
    callback=None,
) -> OptimizeResult:
    """Trust region whose *sample count* is set by its own radius.

    This is the design Result 26 argues for.  Near the optimum the energy
    differences a model interpolates shrink quadratically in the distance to the
    minimum while shot noise stays flat, so any fixed precision eventually fits
    noise -- and precision therefore has to be tied to the radius rather than to
    a schedule.  ``shot_ladder`` does that by hand with a fixed ladder of rungs;
    this does it from the geometry.

    The STORM condition is that the model be *fully linear* on the trust region,
    ``|model - f| <= kappa * radius^2``.  With shot noise ``sigma_1 / sqrt(n)``
    per evaluation that is

        n  >=  (sigma_1 / (kappa * radius^2))^2

    so the shot count grows as ``radius^-4``.  Shrinking the radius by 2 costs
    16x the shots -- which is exactly the escalation the ladder was built to
    approximate, arriving here for a reason rather than by tuning.

    The gradient is estimated by **averaging several** simultaneous-perturbation
    probes at the trust-region scale, and the step goes along that average.  A
    first version used a single probe and stepped along that one random
    direction: in 12 dimensions a random direction has cosine ~1/sqrt(n) with
    the true gradient, so the realised decrease was far below the predicted one,
    every step was rejected, and each rejection shrank the radius -- which under
    the ``radius^-4`` shot rule quadruples the cost of the next iteration.  That
    is a death spiral, and it was: **6 iterations, all rejected, energy never
    moving off Hartree-Fock, and the whole 12.8M budget gone.**

    Averaging costs ``2 * probes`` evaluations per iteration instead of two, and
    is still independent of dimension, unlike the ``n+1`` interpolation set a
    classical trust-region method would build.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)

    # single-shot standard deviation, measured once and rescaled thereafter
    probe_shots = max(min_shots, 1024)
    _, probe_error = oracle.with_error(x, probe_shots)
    sigma_one = float(probe_error * np.sqrt(probe_shots))

    current, _ = oracle.with_error(x, probe_shots)
    history: list[float] = [current]
    shot_history: list[int] = [oracle.shots_used]
    radii: list[float] = []

    for _ in range(maxiter):
        shots = int(np.clip((sigma_one / (kappa * radius**2)) ** 2, min_shots, max_shots))

        gradient = np.zeros(n)
        for _ in range(probes):
            direction = rng.choice([-1.0, 1.0], size=n)
            plus = oracle(x + radius * direction, shots)
            minus = oracle(x - radius * direction, shots)
            gradient += ((plus - minus) / (2.0 * radius)) * direction
        gradient /= probes

        norm = float(np.linalg.norm(gradient))
        if norm < 1e-15:
            radius = max(min_radius, radius * shrink)
            continue

        step = -radius * gradient / norm
        predicted = -radius * norm

        candidate = oracle(x + step, shots)
        actual = candidate - current
        rho = actual / predicted if predicted != 0 else 0.0

        if rho >= accept:
            x = x + step
            current = candidate
            radius = min(max_radius, radius * expand)
        else:
            radius = max(min_radius, radius * shrink)

        history.append(current)
        shot_history.append(oracle.shots_used)
        radii.append(radius)
        if callback:
            callback(x, current)
        if radius <= min_radius and shots >= max_shots:
            break

    return OptimizeResult(
        x=x,
        fun=float(current),
        nit=len(history),
        history=history,
        shot_history=shot_history,
        message=f"storm kappa={kappa} final_radius={radius:.4g}",
        extra={"radii": radii, "sigma_one": sigma_one},
    )


OPTIMIZERS: dict[str, Callable] = {
    "cobyla": lambda o, x0, **kw: scipy_minimize(o, x0, method="COBYLA", **kw),
    "powell": lambda o, x0, **kw: scipy_minimize(o, x0, method="Powell", **kw),
    "nelder-mead": lambda o, x0, **kw: scipy_minimize(o, x0, method="Nelder-Mead", **kw),
    "spsa": spsa,
    "icans": icans,
    "aspsa": adaptive_spsa,
    "sps": stochastic_parameter_shift,
    "ladder": shot_ladder,
    "multistart": multi_start,
    "storm": stochastic_trust_region,
}
