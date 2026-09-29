"""Experiment 032 -- double factorisation has a gauge group too, and its cost is NOT invariant.

Result 86 proved that the THC model has an `M`-parameter gauge group under which
both the reconstructed tensor **and** the 1-norm λ are exactly invariant. The
invariance of λ was the whole point there: it meant no penalty could ever steer
the fit, and gauge fixing was the only option.

**Double factorisation has a gauge group too, and it is much larger — and this
time λ is *not* invariant.** That inverts the consequence. Where THC's gauge
offers nothing to optimise, DF's gauge is **free optimisation**: a direction in
which the Hamiltonian is bit-for-bit unchanged and the block-encoding cost moves.

#### The structure

The repository factorises by eigendecomposing the reshaped two-electron matrix,

    W_(pq),(rs) = (pq|rs) = sum_t lambda_t vec(L^t) vec(L^t)^T ,

and the weights are measured to be **strictly positive** on every molecule here
(the ERI matrix is a Gram matrix of densities, hence positive semidefinite). So
absorbing the weight, `A_t = sqrt(lambda_t) L^t`, gives

    W = sum_t vec(A_t) vec(A_t)^T ,

which is invariant under the **full orthogonal group O(T)** acting as
`A'_t = sum_s O_ts A_s`. That is `T(T-1)/2` free parameters — 45 on H4, 210 on
H6, against THC's `M`.

#### Why the cost moves

Rewriting this repository's own DF 1-norm in terms of the `A_t` (the algebra is
in `two_body_one_norm` below, and it is checked against `DoubleFactorization.one_norm`):

    lambda_2body = 2 sum_t ( sum_p |eig_p(A_t)| )^2 = 2 sum_t ||A_t||_*^2

a sum of **squared nuclear norms**. The nuclear norm is not a function of
`vec(A_t) vec(A_t)^T` alone — it depends on how the spectral weight is
distributed inside each factor — so mixing the factors changes it while leaving
their sum of outer products fixed.

So the question this measures is exactly:

    minimise  sum_t ||sum_s O_ts A_s||_*^2   over O in O(T),
    subject to the Hamiltonian being unchanged -- which it is, identically.

Every reduction found is **free**: zero approximation error, not a truncation,
not a compression. That is what separates this from the published work it sits
next to — orbital-basis rotations (arXiv:2103.14753) optimise a *different*
freedom (the p,q indices), and compressed double factorisation
(arXiv:2212.07957) re-fits the factorisation *approximately*. A targeted search
did not find the exact O(T) factor-mixing freedom used this way, but four
searches are not a literature review and the claim is scoped accordingly.

**The verification that matters** is not the λ reduction, it is that the
Hamiltonian did not move. A rotation that quietly changed the tensor would show a
λ improvement that is simply a different molecule. The reconstructed two-electron
tensor is therefore compared to the original to machine precision on every run,
and that check gates the result.

Run:  python -m experiments.exp032_df_gauge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.optimize import minimize

from qres.bench import RESULTS_DIR
from qres.factorization import double_factorize, molecular_integrals
from qres.problems.chemistry import build_molecule

#: spin factor, as in DoubleFactorization.one_norm -- the factorisation is over
#: spatial orbitals while the number operators run over spin orbitals
SPIN_FACTOR = 2


# --------------------------------------------------------------------------
# the representation the gauge acts on
# --------------------------------------------------------------------------

def absorbed_factors(factorisation) -> np.ndarray:
    """``A_t = sqrt(lambda_t) L^t``, the form in which the gauge is orthogonal.

    Returns shape (T, n, n). Each ``A_t`` is symmetric, and
    ``W = sum_t vec(A_t) vec(A_t)^T`` reproduces the reshaped two-electron
    matrix exactly.
    """
    weights = factorisation.factor_weights
    if np.any(weights < 0):
        raise ValueError(
            f"{int((weights < 0).sum())} negative weights: the gauge group is then "
            "the signature-preserving subgroup, not O(T), and this code assumes O(T)"
        )
    rotations = factorisation.factor_rotations
    diagonals = factorisation.factor_diagonals
    factors = np.einsum("tpi,ti,tqi->tpq", rotations, diagonals, rotations)
    return np.sqrt(weights)[:, None, None] * factors


def reconstruct(factors: np.ndarray) -> np.ndarray:
    """``sum_t vec(A_t) vec(A_t)^T``, reshaped back to ``g[p,q,r,s]``."""
    count, orbitals, _ = factors.shape
    flat = factors.reshape(count, orbitals * orbitals)
    return (flat.T @ flat).reshape(orbitals, orbitals, orbitals, orbitals)


def two_body_one_norm(factors: np.ndarray, spin_factor: int = SPIN_FACTOR) -> float:
    """``2 sum_t ||A_t||_*^2`` -- the two-body part of the DF 1-norm.

    Equivalent to ``DoubleFactorization.one_norm``'s two-body term, which is
    checked in the tests rather than asserted here: that one is written as
    ``sum_t |lambda_t| (spin sum_p |d_tp|)^2 / 2`` with unit-norm factors, and
    substituting ``A_t = sqrt(lambda_t) L^t`` turns it into this.
    """
    eigenvalues = np.linalg.eigvalsh(factors)
    return float(np.sum((spin_factor * np.abs(eigenvalues).sum(axis=1)) ** 2) / 2.0)


def mix(factors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """``A'_t = sum_s O_ts A_s`` -- the gauge action."""
    return np.einsum("ts,spq->tpq", rotation, factors)


def orthogonal_from(parameters: np.ndarray, count: int) -> np.ndarray:
    """``expm(K)`` for the antisymmetric ``K`` built from ``T(T-1)/2`` numbers."""
    generator = np.zeros((count, count))
    rows, columns = np.triu_indices(count, k=1)
    generator[rows, columns] = parameters
    generator = generator - generator.T
    return expm(generator)


# --------------------------------------------------------------------------
# 1. the symbolic part
# --------------------------------------------------------------------------

def symbolic_gauge(orbitals=2):
    """Free symbols: the tensor is invariant, and the 1-norm is not.

    Both halves are proved on the same two-factor rotation, so the contrast is
    between two statements about one object rather than two setups.

    **The non-invariance half needs a witness, and the first one chosen was
    degenerate.** It used `Z = diag(1,-1)` and `X = [[0,1],[1,0]]`, whose real
    rotations `cos t Z + sin t X` are unit Bloch vectors and therefore have
    eigenvalues +-1 at *every* angle. The nuclear norm was constant, the script
    reported "1-norm invariant: True", and that flatly contradicted the numerical
    result in section 2 -- which is how it was caught. A witness that happens to
    lie on an invariant orbit proves nothing about the general case.

    `diag(1,0)` and `diag(0,1)` are not degenerate: rotating them gives
    `diag(cos t, sin t)`, whose nuclear norm `|cos t| + |sin t|` runs from 1 to
    sqrt(2). One explicit non-zero difference is all non-invariance needs.
    """
    theta = sp.Symbol("theta", real=True)
    entries = [[sp.Symbol(f"a{p}{q}", real=True) for q in range(orbitals)]
               for p in range(orbitals)]
    other = [[sp.Symbol(f"b{p}{q}", real=True) for q in range(orbitals)]
             for p in range(orbitals)]
    first = sp.Matrix(orbitals, orbitals,
                      lambda p, q: entries[min(p, q)][max(p, q)])
    second = sp.Matrix(orbitals, orbitals,
                       lambda p, q: other[min(p, q)][max(p, q)])

    rotated_first = sp.cos(theta) * first + sp.sin(theta) * second
    rotated_second = -sp.sin(theta) * first + sp.cos(theta) * second

    # (a) the reconstructed tensor -- general, free symbols throughout
    residuals = []
    for p in range(orbitals):
        for q in range(orbitals):
            for r in range(orbitals):
                for s in range(orbitals):
                    before = first[p, q] * first[r, s] + second[p, q] * second[r, s]
                    after = (rotated_first[p, q] * rotated_first[r, s]
                             + rotated_second[p, q] * rotated_second[r, s])
                    residuals.append(sp.simplify(sp.expand(before - after)))
    tensor_invariant = all(residual == 0 for residual in residuals)

    # (b) the 1-norm -- one explicit witness suffices to refute invariance
    witness = {entries[0][0]: 1, entries[0][1]: 0, entries[1][1]: 0,
               other[0][0]: 0, other[0][1]: 0, other[1][1]: 1}

    def squared_nuclear(matrix):
        return sum(sp.Abs(value)
                   for value in matrix.subs(witness).eigenvals(multiple=True)) ** 2

    norm_before = squared_nuclear(first) + squared_nuclear(second)
    norm_after = (squared_nuclear(sp.Matrix(orbitals, orbitals,
                                            lambda p, q: rotated_first[p, q]))
                  + squared_nuclear(sp.Matrix(orbitals, orbitals,
                                              lambda p, q: rotated_second[p, q])))

    difference = sp.simplify(sp.trigsimp(norm_after - norm_before))
    sampled = float(sp.N(difference.subs(theta, sp.pi / 4)))

    return {
        "tensor_invariant": bool(tensor_invariant),
        "tensor_terms_checked": len(residuals),
        "one_norm_difference": str(difference),
        "one_norm_difference_at_pi_over_4": sampled,
        # invariance is refuted by a single non-zero value, not by simplify()
        # failing to reduce the expression -- those are different claims
        "one_norm_invariant": bool(abs(sampled) < 1e-12),
    }


# --------------------------------------------------------------------------
# 2. the numerical part
# --------------------------------------------------------------------------

def minimise_one_norm(factors, restarts=4, iterations=2000, seed=0):
    """Minimise the DF 1-norm over O(T). Every point is an exact Hamiltonian.

    The nuclear norm is non-smooth where an eigenvalue crosses zero, so L-BFGS is
    not guaranteed to converge here and its termination is recorded rather than
    assumed -- the lesson Result 80 paid for.
    """
    count = factors.shape[0]
    dimension = count * (count - 1) // 2
    rng = np.random.default_rng(seed)

    def objective(parameters):
        return two_body_one_norm(mix(factors, orthogonal_from(parameters, count)))

    best, attempts = None, []
    for index in range(restarts):
        start = (np.zeros(dimension) if index == 0
                 else rng.normal(scale=0.3 * index, size=dimension))
        result = minimize(objective, start, method="L-BFGS-B",
                          options={"maxiter": iterations, "maxfun": iterations * 4})
        attempts.append({
            "value": float(result.fun),
            "iterations": int(result.nit),
            "converged": bool(result.success),
            "message": str(result.message),
        })
        if best is None or result.fun < best["value"]:
            best = {"value": float(result.fun), "parameters": result.x,
                    "iterations": int(result.nit), "converged": bool(result.success)}

    best["attempts"] = attempts
    best["spread"] = float(max(a["value"] for a in attempts)
                           / max(min(a["value"] for a in attempts), 1e-300))
    best["gauge_dimension"] = dimension
    return best


def measure(name, restarts=4, iterations=2000):
    started = time.perf_counter()
    problem = build_molecule(name)
    one_body, two_body, _ = molecular_integrals(problem)
    factorisation = double_factorize(one_body, two_body)
    factors = absorbed_factors(factorisation)

    # the representation must reproduce the tensor before anything is claimed
    rebuild_error = float(np.max(np.abs(reconstruct(factors) - two_body)))

    before = two_body_one_norm(factors)
    best = minimise_one_norm(factors, restarts=restarts, iterations=iterations)
    rotated = mix(factors, orthogonal_from(best["parameters"], factors.shape[0]))
    after = two_body_one_norm(rotated)

    # THE gate on the result: did the Hamiltonian move?
    invariance_error = float(np.max(np.abs(reconstruct(rotated) - two_body)))

    return {
        "molecule": name,
        "orbitals": int(one_body.shape[0]),
        "factors": int(factors.shape[0]),
        "gauge_dimension": best["gauge_dimension"],
        "rebuild_error": rebuild_error,
        "one_norm_before": before,
        "one_norm_after": after,
        "reduction": before / after if after else float("nan"),
        "invariance_error": invariance_error,
        "restart_spread": best["spread"],
        "iterations": best["iterations"],
        "converged": best["converged"],
        "attempts": best["attempts"],
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", default="H2,H4,H6")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=2000)
    args = ap.parse_args()

    print("=== experiment 032 :: the DF gauge group, and the cost it does NOT fix ===")
    print("Result 86: THC's gauge leaves lambda invariant, so nothing to optimise.")
    print("Here the tensor is invariant and lambda is not -- so the gauge is free")
    print("optimisation, at exactly zero approximation error.\n")

    print("--- 1. symbolic: invariant tensor, non-invariant 1-norm ---")
    proof = symbolic_gauge()
    print(f"  tensor entries checked with free symbols: {proof['tensor_terms_checked']}")
    print(f"  reconstructed tensor invariant: {proof['tensor_invariant']}")
    print(f"  1-norm invariant:               {proof['one_norm_invariant']}")
    print(f"  1-norm difference vs theta:     {proof['one_norm_difference']}")
    print(f"  at theta = pi/4:                {proof['one_norm_difference_at_pi_over_4']:+.6f}")

    print("\n--- 2. numerical: how much is available for free ---")
    header = (f"{'molecule':>9}{'orb':>5}{'T':>5}{'dim O(T)':>10}"
              f"{'lambda before':>15}{'after':>12}{'gain':>8}{'|dH|':>10}{'conv':>6}")
    print(header)
    print("-" * len(header))

    rows = []
    for name in args.molecules.split(","):
        row = measure(name, restarts=args.restarts, iterations=args.iterations)
        rows.append(row)
        print(f"{row['molecule']:>9}{row['orbitals']:>5}{row['factors']:>5}"
              f"{row['gauge_dimension']:>10}{row['one_norm_before']:>15.4f}"
              f"{row['one_norm_after']:>12.4f}{row['reduction']:>8.3f}"
              f"{row['invariance_error']:>10.1e}"
              f"{'yes' if row['converged'] else 'NO':>6}", flush=True)

    print("\n--- the verdict ---")
    worst_invariance = max(r["invariance_error"] for r in rows)
    if worst_invariance > 1e-10:
        print(f"REJECTED: the Hamiltonian moved by {worst_invariance:.1e}. Any gain")
        print("above is a different molecule, not a cheaper encoding.")
    else:
        gains = [r["reduction"] for r in rows]
        print(f"The Hamiltonian is unchanged to {worst_invariance:.1e} everywhere, so")
        print(f"every gain is free. Reduction: {min(gains):.3f}x to {max(gains):.3f}x.")
        if max(gains) < 1.01:
            print("\nThat is nothing. The eigendecomposition already sits at or near")
            print("the O(T) optimum, which is itself worth knowing: it means the")
            print("obvious free lunch is not there, and CDF's approximate re-fitting")
            print("is doing something the exact gauge cannot.")
        else:
            print(f"\nlambda sets the qubitized walk count (Result 75), so a {max(gains):.3f}x")
            print("reduction in lambda is the same factor off the runtime, for free.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp032_df_gauge.json"
    with open(path, "w") as fh:
        json.dump({"symbolic": proof, "spin_factor": SPIN_FACTOR, "rows": rows},
                  fh, indent=2, default=float)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
