"""Experiment 037 -- does an Ising Born machine model real data better than classical models?

The lead (NEXT_STEPS, added on request): Coyle, Mills, Danos & Kashefi, "The Born
supremacy", npj Quantum Information 6, 60 (2020). The model is

    |psi> = R_y(alpha) exp(i theta(z)) |+>^n ,  theta(z) = sum_{i<j} J_ij z_i z_j + sum_k b_k z_k

measured in the computational basis, ``p(x) = |<x|psi>|^2``. With every final
angle at pi/2 the circuit is **IQP**, whose sampling is classically intractable
under standard assumptions. Without the final layer every bitstring has
probability ``2^-n`` -- the diagonal phases alone represent nothing.

Sampling hardness only matters if the model is *good*. This measures that first,
on data fixed before any circuit was chosen (Result 68's lesson), by the honest
metric: **held-out log-likelihood**. At 16 qubits it is exact for every model
here, quantum included, so there is no estimator to argue about.

**The comparison that isolates the quantum ingredient.** The fully visible
Boltzmann machine uses the *same* energy ``theta(z)`` with the *same* parameters
``J`` and ``b``, and turns it into probabilities by the Gibbs rule
``e^theta / Z`` instead of the Born rule. Same parameter space, two maps. A third
model, the RBM with 8 hidden units, has exactly the Ising Born machine's 152
parameters. The autoregressive FVSBN has 136, and independent bits 16.

| model | parameters (n = 16) | how p(x) is made |
|---|---|---|
| Ising Born machine | 120 + 16 + 16 = 152 | Born rule, final R_y trained |
| Ising Born machine, IQP | 136 | Born rule, final angles fixed at pi/2 |
| Boltzmann machine | 136 | Gibbs rule on the same theta |
| RBM, 8 hidden | 152 | marginal of a bipartite Gibbs model |
| FVSBN | 136 | product of logistic conditionals |
| independent | 16 | product of Bernoullis |

**Data.** The UCI handwritten digits shipped with scikit-learn (1 797 images,
8x8), average-pooled to 4x4 and thresholded at half intensity: 16 bits, 235
distinct patterns. Real data, chosen before the circuit, and small enough that
every likelihood is exact.

**Protocol.** Ten random 80/20 splits. Every model is fitted by exact
L2-penalised maximum likelihood (L-BFGS, full batch). The penalty strength is
chosen per model from the same grid on a validation fifth carved out of
training, and the model is then refitted on all of training and scored on the
held-out 20%. Non-convex models take the best of several random starts by
training likelihood.

Needs torch (CPU): ``pip install --index-url https://download.pytorch.org/whl/cpu torch``.

Run:  python -m experiments.exp037_ising_born_machine [--splits 10 --inits 3]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qres.bench import RESULTS_DIR

BITS = 16
THRESHOLD = 8          # half of the digits' 0..16 intensity scale
TEST_FRACTION = 0.2
RBM_HIDDEN = 8         # 16*8 + 16 + 8 = 152, the Ising Born machine's count
MAX_ITERATIONS = 500
ROUNDS = 10            # L-BFGS calls per fit, stopped early once the loss is flat

#: L2 strengths, chosen per model on a validation split carved from training.
#: Unregularised maximum likelihood is not a fair protocol here: a pair of
#: pixels that never co-occur in training drives the Boltzmann and FVSBN
#: couplings to infinity, so any test image with that pair costs ~10 bits --
#: and the longer the optimiser runs, the worse it gets. The Born machine's
#: periodic phases cannot run away like that, which would hand it an advantage
#: that is an artefact of the protocol. A first run did exactly that.
PENALTIES = (0.0, 1e-4, 1e-3, 1e-2)
VALIDATION_FRACTION = 0.2

#: convex fits need one start; the others get ``--inits``
CONVEX = {"Boltzmann machine", "FVSBN", "independent"}


# ---------------------------------------------------------------- the data

def digits_bits(threshold=THRESHOLD):
    """UCI digits, 2x2 average-pooled to 4x4 and thresholded: (1797, 16) in {0,1}."""
    from sklearn.datasets import load_digits

    images = load_digits().images
    pooled = images.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4))
    return (pooled > threshold).astype(np.int64).reshape(-1, BITS)


def indices(bits):
    """Row index of each bitstring in the enumeration order (bit 0 most significant)."""
    weights = 1 << np.arange(bits.shape[1] - 1, -1, -1)
    return bits @ weights


def split(count, seed, fraction=TEST_FRACTION):
    order = np.random.default_rng(seed).permutation(count)
    cut = int(round(fraction * count))
    return order[cut:], order[:cut]


# ---------------------------------------------------------------- the models

class Enumeration:
    """Every bitstring of ``n`` bits, as 0/1 and as spins, plus pair products."""

    def __init__(self, n):
        import torch

        grid = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
        self.n = n
        self.bits = torch.tensor(grid, dtype=torch.float64)
        self.spins = 1.0 - 2.0 * self.bits
        rows, cols = np.triu_indices(n, k=1)
        self.pairs = self.spins[:, rows] * self.spins[:, cols]


def ising_energy(enum, pair_weights, fields):
    return enum.pairs @ pair_weights + enum.spins @ fields


def born_log_probs(enum, pair_weights, fields, angles):
    """log |<x| R_y(angles) e^{i theta} |+>^n|^2 for every x."""
    import torch

    theta = ising_energy(enum, pair_weights, fields)
    state = torch.exp(1j * theta) / math.sqrt(2**enum.n)
    state = state.reshape([2] * enum.n)
    for qubit in range(enum.n):
        c, s = torch.cos(angles[qubit] / 2), torch.sin(angles[qubit] / 2)
        rotation = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(state.dtype)
        state = torch.movedim(torch.tensordot(rotation, state, dims=([1], [qubit])), 0, qubit)
    probabilities = (state.real**2 + state.imag**2).reshape(-1)
    return torch.log(probabilities.clamp_min(1e-300))


def gibbs_log_probs(enum, pair_weights, fields):
    import torch

    theta = ising_energy(enum, pair_weights, fields)
    return theta - torch.logsumexp(theta, dim=0)


def rbm_log_probs(enum, visible_bias, hidden_bias, weights):
    import torch

    free = enum.bits @ visible_bias + torch.nn.functional.softplus(
        enum.bits @ weights + hidden_bias).sum(dim=1)
    return free - torch.logsumexp(free, dim=0)


def fvsbn_log_probs(enum, lower, bias):
    """Autoregressive: p(x_i | x_<i) = sigmoid(sum_{j<i} W_ij x_j + b_i)."""
    import torch

    weights = torch.zeros(enum.n, enum.n, dtype=torch.float64)
    rows, cols = np.tril_indices(enum.n, k=-1)
    weights = weights.index_put((torch.tensor(rows), torch.tensor(cols)), lower)
    logits = enum.bits @ weights.T + bias
    return (enum.bits * torch.nn.functional.logsigmoid(logits)
            + (1 - enum.bits) * torch.nn.functional.logsigmoid(-logits)).sum(dim=1)


def independent_log_probs(enum, logits):
    import torch

    return (enum.bits * torch.nn.functional.logsigmoid(logits)
            + (1 - enum.bits) * torch.nn.functional.logsigmoid(-logits)).sum(dim=1)


def model_specs(n=BITS, hidden=RBM_HIDDEN):
    """name -> (parameter shapes, log-prob function, frozen settings)."""
    pairs = n * (n - 1) // 2
    return {
        "Ising Born machine": ([(pairs,), (n,), (n,)], born_log_probs),
        "Ising Born machine, IQP": ([(pairs,), (n,)], "iqp"),
        "Boltzmann machine": ([(pairs,), (n,)], gibbs_log_probs),
        "RBM, 8 hidden": ([(n,), (hidden,), (n, hidden)], rbm_log_probs),
        "FVSBN": ([(pairs,), (n,)], fvsbn_log_probs),
        "independent": ([(n,)], independent_log_probs),
    }


def log_prob_function(name, enum):
    shapes, function = model_specs(enum.n)[name]
    if function == "iqp":
        import torch

        fixed = torch.full((enum.n,), math.pi / 2, dtype=torch.float64)
        return shapes, lambda pair_weights, fields: born_log_probs(enum, pair_weights,
                                                                   fields, fixed)
    return shapes, lambda *parameters: function(enum, *parameters)


def parameter_count(name, n=BITS):
    return int(sum(np.prod(shape) for shape in model_specs(n)[name][0]))


def fit(name, enum, train_index, seed, penalty=0.0, scale=0.1, iterations=MAX_ITERATIONS):
    """Exact (L2-penalised) maximum likelihood from one random start.

    Returns (unpenalised train NLL, parameters, log-prob function). The penalty
    applies to every parameter except the Born machine's final angles, which
    set the measurement basis rather than a strength.
    """
    import torch

    shapes, log_probs = log_prob_function(name, enum)
    generator = torch.Generator().manual_seed(seed)
    parameters = [(scale * torch.randn(shape, generator=generator, dtype=torch.float64))
                  .requires_grad_() for shape in shapes]
    if name == "Ising Born machine":
        # start the final layer near IQP's pi/2, not at the uniform-distribution point 0
        with torch.no_grad():
            parameters[2] += math.pi / 2
    target = torch.tensor(train_index)
    optimiser = torch.optim.LBFGS(parameters, max_iter=iterations, tolerance_grad=1e-10,
                                  tolerance_change=1e-12, history_size=50,
                                  line_search_fn="strong_wolfe")

    penalised = parameters[:2] if name == "Ising Born machine" else parameters

    def closure():
        optimiser.zero_grad()
        loss = -log_probs(*parameters)[target].mean()
        if penalty:
            loss = loss + penalty * sum((p**2).sum() for p in penalised)
        loss.backward()
        return loss

    # one L-BFGS call can stop on its iteration budget short of convergence (the
    # RBM did, by 1e-3 bits); repeat until a call no longer moves the loss
    previous = math.inf
    for _ in range(ROUNDS):
        objective = float(optimiser.step(closure))
        if previous - objective < 1e-9:
            break
        previous = objective
    with torch.no_grad():
        final = float(-log_probs(*parameters)[target].mean())
    return final, [p.detach() for p in parameters], log_probs


def best_of_starts(name, enum, train_index, seed, penalty, inits):
    starts = 1 if name in CONVEX else inits
    fits = [fit(name, enum, train_index, seed=seed + start, penalty=penalty)
            for start in range(starts)]
    return min(fits, key=lambda f: f[0])


def select_and_fit(name, enum, train_index, seed, inits, penalties=PENALTIES):
    """Choose the L2 strength on validation, then refit on all of training."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_index))
    cut = int(round(VALIDATION_FRACTION * len(order)))
    inner, validation = train_index[order[cut:]], train_index[order[:cut]]
    scores = {}
    for penalty in penalties:
        _, parameters, log_probs = best_of_starts(name, enum, inner, seed, penalty, inits)
        scores[penalty] = held_out_bits(log_probs, parameters, validation)
    chosen = min(scores, key=scores.get)
    train_nll, parameters, log_probs = best_of_starts(name, enum, train_index, seed,
                                                      chosen, inits)
    return train_nll, parameters, log_probs, chosen, scores


def held_out_bits(log_probs, parameters, test_index):
    """Average held-out negative log-likelihood, in bits per sample."""
    import torch

    with torch.no_grad():
        values = log_probs(*parameters)[torch.tensor(test_index)]
    return float(-values.mean() / math.log(2))


def main() -> int:
    import torch

    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--inits", type=int, default=3)
    args = ap.parse_args()

    print("=== experiment 037 :: Ising Born machine against classical models, held out ===")
    data = digits_bits()
    index = indices(data)
    print(f"{len(data)} images -> {BITS} bits, {len(np.unique(index))} distinct patterns\n")
    enum = Enumeration(BITS)
    names = list(model_specs())

    rows = {name: [] for name in names}
    for split_seed in range(args.splits):
        train, test = split(len(data), split_seed)
        line = []
        for name in names:
            started = time.perf_counter()
            train_nll, parameters, log_probs, chosen, scores = select_and_fit(
                name, enum, index[train], seed=1000 * split_seed, inits=args.inits)
            test_bits = held_out_bits(log_probs, parameters, index[test])
            rows[name].append({"split": split_seed, "train_nll_bits": train_nll / math.log(2),
                               "test_nll_bits": test_bits, "penalty": chosen,
                               "validation_bits": {str(k): v for k, v in scores.items()},
                               "seconds": time.perf_counter() - started})
            line.append(f"{name.split(',')[0][:10]} {test_bits:6.3f}")
        print(f"  split {split_seed}: " + "  ".join(line), flush=True)

    print(f"\n  {'model':<26}{'params':>7}{'held-out bits/sample':>24}{'train':>9}")
    summary = {}
    for name in names:
        test = np.array([r["test_nll_bits"] for r in rows[name]])
        train = np.array([r["train_nll_bits"] for r in rows[name]])
        summary[name] = {"parameters": parameter_count(name),
                         "test_mean": float(test.mean()), "test_std": float(test.std(ddof=1)),
                         "train_mean": float(train.mean())}
        print(f"  {name:<26}{parameter_count(name):>7}{test.mean():>14.3f} +- "
              f"{test.std(ddof=1):5.3f}{train.mean():>9.3f}")

    # paired comparison on identical splits: Born vs Gibbs on the same energy
    born = np.array([r["test_nll_bits"] for r in rows["Ising Born machine"]])
    for other in ("Boltzmann machine", "RBM, 8 hidden", "FVSBN"):
        difference = born - np.array([r["test_nll_bits"] for r in rows[other]])
        summary[f"Born minus {other}"] = {
            "mean": float(difference.mean()),
            "std": float(difference.std(ddof=1)),
            "born_better_on": int((difference < 0).sum()), "splits": len(difference)}
        print(f"  Ising Born machine minus {other:<18}: {difference.mean():+.3f} "
              f"+- {difference.std(ddof=1):.3f} bits  (Born better on "
              f"{(difference < 0).sum()}/{len(difference)} splits)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp037_ising_born_machine.json"
    with open(path, "w") as fh:
        json.dump({"bits": BITS, "threshold": THRESHOLD, "splits": args.splits,
                   "inits": args.inits, "rows": rows, "summary": summary}, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
