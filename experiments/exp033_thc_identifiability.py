"""Experiment 033 -- where the THC gauge stops being the only flat direction.

Result 86 proved that the THC model has an ``M``-parameter gauge group and read
the Hessian's null space as a **lower bound** of ``M`` flat directions. It left
the count itself open, and explained the extra flatness it did see with a
sentence that is only half right: "where the model can fit the tensor exactly it
is over-parameterised".

This experiment settles the count, **exactly**, and the answer has a second
consequence that Result 86 could not reach.

**The claim.** Write ``P = N(N+1)/2`` for the number of orbital pairs. The
reconstructed tensor is ``V = X Z X^T`` where column ``m`` of ``X`` is the
symmetric rank-one matrix ``chi_m chi_m^T`` read as a ``P``-vector. The generic
rank of the model's Jacobian is::

    P^2                                   if M >= P
    N M + M^2 - M - M * max(0, M + N - 1 - P)   otherwise

so the number of exactly flat directions of *any* objective built from the
reconstructed tensor is ``M`` -- the gauge group and nothing else -- **if and only
if ``M <= N(N-1)/2 + 1``**. Past that boundary the span of the ``M`` rank-one
columns meets the Veronese variety of rank-one matrices in a positive-dimensional
set, the columns can slide along it without changing the span, and every slide
is another exactly flat direction.

**Proved, not measured, and both halves are needed.**

* *Lower bound on the rank, computed.* The Jacobian (closed form, checked entry
  by entry against SymPy's own differentiation of the model) is evaluated at an
  integer point and its rank taken modulo the prime ``2^31 - 1``. A minor that is
  non-zero mod p is non-zero over the integers, so this is a certified lower
  bound on the rank at that point, and the rank at any point is a lower bound on
  the generic rank.
* *Upper bound on the rank, argued.* The gauge gives ``M`` independent kernel
  vectors everywhere (Result 86). If ``M >= P`` the image has at most ``P^2``
  dimensions. Otherwise let ``U`` be the span of the ``M`` columns: a
  ``(M-1)``-plane in ``P^(P-1)`` meeting the ``(N-1)``-dimensional Veronese
  variety of rank-one matrices, so every component of the intersection through a
  column has dimension at least ``d = M + N - 1 - P``. Moving any column inside it
  keeps ``U``, hence (re-solving ``Z``, which ``X`` of full column rank allows)
  keeps ``V`` -- ``M d`` further kernel directions, independent of the gauge and
  of each other because they move different columns.

Where the computed lower bound meets the argued upper bound the count is exact,
and it meets it in every case checked: all ``(N, M)`` with ``N <= 6`` and
``M <= P + 1``.

**The consequence Result 86 could not reach.** On the gauge directions the 1-norm
is constant (Result 86). On the *sliding* directions it is **not** -- its exact
directional derivative is non-zero. So past the boundary the unpenalised fit has
a manifold of exact minimisers along which lambda, the quantity the algorithm's
cost is proportional to, varies freely. There, and only there, a penalty is not
a preference but the thing that decides lambda at all.

**Why it matters for this repository.** The chemical-accuracy ranks Result 78
found are ``M = 4, 8, 18`` for ``N = 2, 4, 6``. The boundary is ``2, 7, 16``. All
three lie past it, and H2's lies past ``P = 3``, where the model represents
*every* pair-symmetric tensor exactly -- so that point is fixed by algebra, not
by chemistry.

Run:  python -m experiments.exp033_thc_identifiability
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from qres.bench import RESULTS_DIR

#: every rank up to one past saturation, for every orbital count up to this
MAX_ORBITALS = 6

#: ranks are taken modulo this prime: a certified lower bound on the rational
#: rank, and about 7x faster than exact rationals, whose entries grow during
#: elimination (N=6, M=12: 3.4 s against 22 s, same answer)
PRIME = 2**31 - 1

#: the chemical-accuracy ranks Result 78 measured, per orbital count
RESULT_78_RANKS = {2: 4, 4: 8, 6: 18}

#: the three cases in Result 86's Hessian table: (molecule, orbitals, rank,
#: flat directions it reported)
RESULT_86_CASES = (("H2", 2, 4, 15), ("H4", 4, 4, 4), ("H4", 4, 6, 8))


def pairs(orbitals):
    return orbitals * (orbitals + 1) // 2


def identifiable_up_to(orbitals):
    """Largest rank at which the gauge is the only continuous redundancy."""
    return orbitals * (orbitals - 1) // 2 + 1


def predicted_rank(orbitals, rank):
    """Generic Jacobian rank of the THC model, from the dimension count."""
    size = pairs(orbitals)
    if rank >= size:
        return size * size
    slide = max(0, rank + orbitals - 1 - size)
    return orbitals * rank + rank * rank - rank - rank * slide


def symbolic_jacobian(orbitals, rank):
    """The model's Jacobian, differentiated by SymPy rather than by hand.

    Rows are the ``P x P`` entries of ``V = X Z X^T`` (pairs ``p <= q``, so the
    duplicated ``(q, p)`` rows of the full tensor are dropped -- they cannot
    change a rank); columns are ``chi`` row-major, then ``Z`` row-major, the
    packing ``_objective`` uses.
    """
    chi = sp.Matrix(orbitals, rank, lambda p, m: sp.Symbol(f"c{p}_{m}"))
    coupling = sp.Matrix(rank, rank, lambda m, n: sp.Symbol(f"z{m}_{n}"))
    index = [(p, q) for p in range(orbitals) for q in range(p, orbitals)]
    columns = sp.Matrix(len(index), rank,
                        lambda i, m: chi[index[i][0], m] * chi[index[i][1], m])
    tensor = sp.Matrix(list(columns * coupling * columns.T))
    parameters = list(chi) + list(coupling)
    return tensor.jacobian(parameters), parameters


def exact_jacobian(chi, coupling):
    """The same Jacobian in closed form, for integer (exact) or float inputs.

    Used for the sweep because differentiating symbolically at ``N = 5`` is slow;
    agreement with :func:`symbolic_jacobian` is checked in the tests entry by
    entry, so the proof does not rest on a hand derivation.
    """
    orbitals, rank = len(chi), len(chi[0])
    index = [(p, q) for p in range(orbitals) for q in range(p, orbitals)]
    size = len(index)
    X = [[chi[p][m] * chi[q][m] for m in range(rank)] for p, q in index]
    XZ = [[sum(X[i][n] * coupling[n][m] for n in range(rank)) for m in range(rank)]
          for i in range(size)]
    ZXt = [[sum(coupling[m][n] * X[j][n] for n in range(rank)) for j in range(size)]
           for m in range(rank)]
    # d X[(p,q), m] / d chi[a, m] = delta_pa chi_qm + delta_qa chi_pm
    D = [[[(chi[q][m] if p == a else 0) + (chi[p][m] if q == a else 0)
           for m in range(rank)] for a in range(orbitals)] for p, q in index]

    rows = []
    for i in range(size):
        for j in range(size):
            row = [D[i][a][m] * ZXt[m][j] + XZ[i][m] * D[j][a][m]
                   for a in range(orbitals) for m in range(rank)]
            row += [X[i][m] * X[j][n] for m in range(rank) for n in range(rank)]
            rows.append(row)
    return rows


def integer_point(orbitals, rank, seed):
    """A reproducible point with small non-zero integer entries."""
    rng = random.Random(seed)
    draw = lambda: rng.choice([v for v in range(-9, 10) if v])  # noqa: E731
    chi = [[draw() for _ in range(rank)] for _ in range(orbitals)]
    coupling = [[draw() for _ in range(rank)] for _ in range(rank)]
    return chi, coupling


def certified_rank(rows, domain=None):
    """Rank of an integer matrix: over ``GF(PRIME)`` by default, a lower bound.

    Pass ``domain=sp.QQ`` for the exact rational rank; the tests check the two
    agree where the rational one is cheap.
    """
    domain = sp.GF(PRIME) if domain is None else domain
    matrix = DomainMatrix([[sp.ZZ(v) for v in row] for row in rows],
                          (len(rows), len(rows[0])), sp.ZZ)
    return matrix.convert_to(domain).rank()


def generic_count(orbitals, rank, seed=1):
    """Certified rank at one integer point, against the prediction."""
    chi, coupling = integer_point(orbitals, rank, seed)
    parameters = orbitals * rank + rank * rank
    found = certified_rank(exact_jacobian(chi, coupling))
    return {
        "orbitals": orbitals,
        "rank": rank,
        "pairs": pairs(orbitals),
        "parameters": parameters,
        "jacobian_rank": found,
        "flat_directions": parameters - found,
        "predicted_rank": predicted_rank(orbitals, rank),
        "gauge_only": parameters - found == rank,
        "matches_prediction": found == predicted_rank(orbitals, rank),
    }


def gauge_generators(chi, coupling):
    """Exact tangent vectors of the gauge orbit (Result 86's, in integers)."""
    orbitals, rank = len(chi), len(chi[0])
    generators = []
    for k in range(rank):
        vector = [chi[p][m] if m == k else 0
                  for p in range(orbitals) for m in range(rank)]
        vector += [-2 * ((m == k) + (n == k)) * coupling[m][n]
                   for m in range(rank) for n in range(rank)]
        generators.append(vector)
    return generators


def one_norm_gradient(chi, coupling):
    """Exact gradient of ``lambda = 1/2 sum_mn |Z_mn| w_m w_n``, ``w_m = |chi_m|^2``.

    Differentiable wherever no ``Z_mn`` is zero, which :func:`integer_point`
    guarantees.
    """
    orbitals, rank = len(chi), len(chi[0])
    weight = [sum(chi[p][m] ** 2 for p in range(orbitals)) for m in range(rank)]
    grad_chi = [chi[p][m] * sum((abs(coupling[m][n]) + abs(coupling[n][m])) * weight[n]
                                for n in range(rank))
                for p in range(orbitals) for m in range(rank)]
    grad_coupling = [sp.Rational(1, 2) * sp.sign(coupling[m][n]) * weight[m] * weight[n]
                     for m in range(rank) for n in range(rank)]
    return [sp.Integer(v) for v in grad_chi] + grad_coupling


def one_norm_along_flat_directions(orbitals, rank, seed=1):
    """Does lambda move along the exactly flat directions?

    Kernel of the exact Jacobian, over the rationals. The gauge generators must
    lie in it and lambda must be constant along them (Result 86). Whether lambda
    is constant along the *whole* kernel is the question.
    """
    chi, coupling = integer_point(orbitals, rank, seed)
    rows = exact_jacobian(chi, coupling)
    jacobian = DomainMatrix([[sp.QQ(v) for v in row] for row in rows],
                            (len(rows), len(rows[0])), sp.QQ)
    kernel = jacobian.nullspace().to_Matrix()
    gradient = sp.Matrix(one_norm_gradient(chi, coupling))
    generators = sp.Matrix(gauge_generators(chi, coupling))

    gauge_in_kernel = (jacobian.to_Matrix() * generators.T).is_zero_matrix
    along_gauge = [sp.simplify((generators[k, :] * gradient)[0]) for k in range(rank)]
    along_kernel = [(kernel[i, :] * gradient)[0] for i in range(kernel.rows)]
    return {
        "orbitals": orbitals,
        "rank": rank,
        "flat_directions": kernel.rows,
        "gauge_in_kernel": bool(gauge_in_kernel),
        "lambda_constant_along_gauge": all(v == 0 for v in along_gauge),
        "lambda_constant_along_all_flat_directions": all(v == 0 for v in along_kernel),
        "largest_directional_derivative": str(max((abs(v) for v in along_kernel),
                                                  default=0)),
    }


def jacobian_spectrum_at_fit(molecule, rank):
    """Singular values of the Jacobian at the point exp031 actually fits.

    Result 86 counted flat directions from Hessian eigenvalues below 1e-6 of the
    largest. The Jacobian's exact zeros are the model's; anything else is
    conditioning, and the ratio between the last zero and the first non-zero
    value says which a cutoff is looking at.
    """
    from qres.factorization import molecular_integrals
    from qres.problems.chemistry import build_molecule
    from experiments.exp029_rank_exponent import gauge_thc_fit
    from experiments.exp031_gauge_structure import numerical_hessian

    problem = build_molecule(molecule)
    one_body, two_body, _ = molecular_integrals(problem)
    orbitals = one_body.shape[0]
    fit = gauge_thc_fit(one_body, two_body, rank, alpha=0.0, restarts=1,
                        iterations=20000)
    chi, coupling = fit["chi"], fit["coupling"]
    singular = np.linalg.svd(np.array(exact_jacobian(chi.tolist(), coupling.tolist()),
                                      dtype=float), compute_uv=False)
    singular = np.sort(singular) / singular.max()
    packed = np.concatenate([chi.ravel(), coupling.ravel()])
    hessian = np.sort(np.abs(np.linalg.eigvalsh(
        numerical_hessian(packed, orbitals, rank, two_body, 0.0))))
    hessian = hessian / hessian.max()
    zeros = int(np.sum(singular < 1e-12))
    return {
        "molecule": molecule,
        "rank": rank,
        "residual": fit["residual"],
        "jacobian_zero_singular_values": zeros,
        "first_nonzero_singular_value": float(singular[zeros]),
        "hessian_smallest": [float(v) for v in hessian[:rank + 4]],
        "hessian_below_1e-6": int(np.sum(hessian < 1e-6)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-orbitals", type=int, default=MAX_ORBITALS)
    ap.add_argument("--skip-refit", action="store_true",
                    help="skip the H4 fits behind section 4")
    args = ap.parse_args()

    print("=== experiment 033 :: where the THC gauge stops being the only flat direction ===")
    print(f"Jacobian rank mod {PRIME} at one integer point per (N, M): a certified")
    print("lower bound on the generic rank. The gauge and the dimension count give")
    print("the upper bound; where they meet, the count is exact.\n")

    # ----------------------------------------------------- 1. the exact count
    print("--- 1. flat directions of the THC model, exactly ---")
    print(f"{'N':>3}{'P':>4}{'M':>4}{'params':>8}{'rank':>7}{'flat':>6}"
          f"{'gauge':>7}{'predicted':>11}   regime")
    sweep = []
    for orbitals in range(2, args.max_orbitals + 1):
        for rank in range(1, pairs(orbitals) + 2):
            started = time.perf_counter()
            row = generic_count(orbitals, rank)
            row["seconds"] = time.perf_counter() - started
            sweep.append(row)
            regime = ("gauge only" if row["gauge_only"] else
                      "saturated (M >= P)" if rank >= pairs(orbitals) else
                      "columns slide")
            print(f"{orbitals:>3}{row['pairs']:>4}{rank:>4}{row['parameters']:>8}"
                  f"{row['jacobian_rank']:>7}{row['flat_directions']:>6}{rank:>7}"
                  f"{row['predicted_rank']:>11}   {regime}"
                  f"{'' if row['matches_prediction'] else '   <-- MISMATCH'}", flush=True)

    matched = sum(row["matches_prediction"] for row in sweep)
    boundary_holds = all(row["gauge_only"] == (row["rank"] <= identifiable_up_to(row["orbitals"]))
                         for row in sweep)
    print(f"\n  prediction exact in {matched}/{len(sweep)} cases")
    print(f"  'flat = M exactly  <=>  M <= N(N-1)/2 + 1' holds in every case: {boundary_holds}")

    # ------------------------------------- 2. lambda along the flat directions
    print("\n--- 2. does lambda move along the flat directions? ---")
    lam_cases = []
    for orbitals, rank in ((3, 4), (3, 5), (4, 7), (4, 8), (2, 3)):
        row = one_norm_along_flat_directions(orbitals, rank)
        lam_cases.append(row)
        print(f"  N={orbitals} M={rank}: {row['flat_directions']:>2} flat, gauge in kernel "
              f"{row['gauge_in_kernel']}, lambda constant along gauge "
              f"{row['lambda_constant_along_gauge']}, along ALL flat directions "
              f"{row['lambda_constant_along_all_flat_directions']}")

    # ----------------------- 3. where this repository's measured ranks sit
    print("\n--- 3. where Result 78's chemical-accuracy ranks sit ---")
    placements = []
    for orbitals, rank in RESULT_78_RANKS.items():
        placement = {
            "orbitals": orbitals, "rank": rank, "pairs": pairs(orbitals),
            "identifiable_up_to": identifiable_up_to(orbitals),
            "extra_flat_directions": (orbitals * rank + rank * rank - rank
                                      - predicted_rank(orbitals, rank)),
            "exactly_representable": rank >= pairs(orbitals),
        }
        placements.append(placement)
        print(f"  N={orbitals}: measured M={rank}, gauge-only up to "
              f"M={placement['identifiable_up_to']}, saturates at M={placement['pairs']}"
              f"  -> {placement['extra_flat_directions']} flat directions beyond the gauge"
              f"{', and EVERY tensor is exact' if placement['exactly_representable'] else ''}")

    # ------------------------------------------ 4. Result 86's table, revisited
    refits = []
    if not args.skip_refit:
        print("\n--- 4. Result 86's Hessian table against the exact count ---")
        for molecule, orbitals, rank, reported in RESULT_86_CASES:
            exact = generic_count(orbitals, rank)["flat_directions"]
            line = f"  {molecule} M={rank}: reported {reported}, exact generic {exact}"
            if molecule == "H4":
                spectrum = jacobian_spectrum_at_fit(molecule, rank)
                spectrum["reported"], spectrum["exact_generic"] = reported, exact
                refits.append(spectrum)
                line += (f"; at the fitted point J has {spectrum['jacobian_zero_singular_values']}"
                         f" zero singular values, next {spectrum['first_nonzero_singular_value']:.1e};"
                         f" Hessian below 1e-6 here: {spectrum['hessian_below_1e-6']}")
            print(line, flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp033_thc_identifiability.json"
    with open(path, "w") as fh:
        json.dump({
            "sweep": sweep,
            "prediction_exact": f"{matched}/{len(sweep)}",
            "boundary_holds": boundary_holds,
            "one_norm_along_flat_directions": lam_cases,
            "result_78_ranks": placements,
            "result_86_refits": refits,
        }, fh, indent=2, default=float)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
