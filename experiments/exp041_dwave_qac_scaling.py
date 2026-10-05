"""Experiment 041 -- does a classical annealer match D-Wave's "scaling advantage"?

**The claim.** Munoz-Bauza & Lidar, PRL 134, 160601 (2025), arXiv:2401.07184:
"the first demonstration of an algorithmic quantum speedup in approximate
optimization". On random spin glasses on the degree-5 logical graph that
quantum annealing correction (QAC) induces on D-Wave's Pegasus chip, the median
time-to-epsilon at a 1% optimality gap scales as ``N^1.69 +- 0.12`` for QAC,
against ``N^1.93 +- 0.03`` for parallel tempering with isoenergetic cluster moves
(PT-ICM), "the top classical heuristic". Simulated annealing was tried and "not
competitive at large problem sizes"; nothing else classical was.

**Why the claim is testable classically.** For a fixed *relative* gap the
target is an energy *density*. The final energy density of an annealer
self-averages, so once a fixed number of sweeps reaches the target density the
success probability tends to one as N grows, and the time-to-epsilon of a
serial annealer tends to ``N^1`` -- the cost of one sweep. The paper notes such
linear behaviour is "unsurprising" at large gaps; the question is whether a
plainly tuned classical annealer already gets there at 1%.

**What is reproduced.**

* *The graph.* QAC's ``[[3,1,3]]`` code cell by cell on Pegasus P16 (nice
  coordinates, three interleaved Chimera layers): in every K4,4 cell, logical A
  = three vertical data qubits plus a horizontal penalty qubit, logical B the
  mirror image. Two logical qubits are coupled when at least two physical
  couplers join their data qubits. That gives exactly what the paper describes:
  6L^2 logical qubits (1350 at L = 15; the paper's 1322 lacks its chip's dead
  qubits), bulk degree 5, native 5-loops, a honeycomb with non-planar bonds.
* *The instances.* Sidon-28 couplings, ``J_ij`` uniform on
  ``+-{8/28, 13/28, 19/28, 1}``, no fields, sizes L = 5 ... 15.
* *The metric.* ``TTe = t * log(1 - 0.99) / log(1 - p_e)`` (R clipped at 1), with
  ``p_e`` the probability of ending within ``e |E0|`` of the ground state; the
  per-run effort ``t`` optimised per size to minimise the median over
  instances, as the paper does for QA; a power law fitted to the medians.
  Classical time is counted as serial spin updates -- the same convention the
  paper uses (it multiplies QA's time by N/N_max for the same reason).
* *The baseline.* PT-ICM with the paper's temperature set 1 (32 temperatures,
  beta 0.1 ... 5, ICM on the 8 coldest), to check that its exponent is
  reproduced before anything is compared with it.

Run:  python -m experiments.exp041_dwave_qac_scaling [--instances 30] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qres.bench import RESULTS_DIR

GRAPH_FILE = RESULTS_DIR / "exp041_qac_logical_graph.json"
RAW = RESULTS_DIR / "raw" / "exp041"
SIDON28 = np.array([8 / 28, 13 / 28, 19 / 28, 1.0])
SIZES = tuple(range(5, 16))
EPSILON = 0.01
#: the paper's PT-ICM temperature set 1
PT_TEMPERATURES, PT_BETA_MIN, PT_BETA_MAX, PT_ICM = 32, 0.1, 5.0, 8
#: SA's geometric schedule, tuned once on held-out instances (L = 10, seeds
#: 1000-1003) among six ranges and frozen for every size; only the number of
#: sweeps is optimised per size, as the paper optimises QA's annealing time
SA_BETA_MIN, SA_BETA_MAX = 1.0, 8.0
#: path-integral SQA: Trotter slices, inverse temperature, initial transverse field;
#: beta and Gamma chosen among six pairs on the same held-out instances as SA, by
#: single-slice success
SQA_SLICES, SQA_BETA, SQA_GAMMA = 16, 16.0, 1.5


# ------------------------------------------------------------------ the graph

def build_qac_logical_graph(m=16, penalty=3):
    """QAC logical graph on Pegasus P_m, from dwave_networkx (needed only here).

    Returns ``(nodes, edges)`` with nodes as ``(t, y, x, u)`` and edges as index
    pairs with the number of physical couplers realising each logical bond.
    """
    import itertools
    import warnings

    import networkx as nx

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import dwave_networkx as dnx

    physical = dnx.pegasus_graph(m, nice_coordinates=True)
    side = m - 1
    nodes, data = [], {}
    for t, y, x, u in itertools.product(range(3), range(side), range(side), range(2)):
        name = (t, y, x, u)
        nodes.append(name)
        data[name] = [(t, y, x, u, k) for k in range(4) if k != penalty]
        assert all(physical.has_edge(d, (t, y, x, 1 - u, penalty)) for d in data[name])
    owner = {q: name for name, qs in data.items() for q in qs}
    links = {}
    for a, b in physical.edges():
        if a in owner and b in owner and owner[a] != owner[b]:
            key = tuple(sorted((owner[a], owner[b])))
            links.setdefault(key, []).append((a, b))
    index = {name: i for i, name in enumerate(nodes)}
    edges = []
    for (i, j), couplers in sorted(links.items()):
        matched = len(nx.max_weight_matching(nx.Graph(couplers), maxcardinality=True))
        if matched >= 2:
            edges.append((index[i], index[j], matched))
    return nodes, edges


@lru_cache(maxsize=1)
def full_graph():
    """The L = 15 graph, from the cached file (rebuilt if dwave_networkx is present)."""
    if not GRAPH_FILE.exists():
        nodes, edges = build_qac_logical_graph()
        GRAPH_FILE.write_text(json.dumps({"nodes": nodes, "edges": edges}))
    payload = json.loads(GRAPH_FILE.read_text())
    return [tuple(n) for n in payload["nodes"]], [tuple(e) for e in payload["edges"]]


def logical_graph(side):
    """The L x L corner of the logical graph: nodes with y, x < L, relabelled."""
    nodes, edges = full_graph()
    keep = [i for i, (t, y, x, u) in enumerate(nodes) if y < side and x < side]
    relabel = {old: new for new, old in enumerate(keep)}
    pairs = [(relabel[i], relabel[j]) for i, j, _ in edges if i in relabel and j in relabel]
    return len(keep), np.array(pairs, dtype=np.int64)


def instance(side, seed):
    """Sidon-28 spin glass on the L x L logical graph: ``(n, i, j, J)``."""
    n, pairs = logical_graph(side)
    rng = np.random.default_rng([side, seed])
    couplings = rng.choice(SIDON28, size=len(pairs)) * rng.choice([-1.0, 1.0], size=len(pairs))
    return n, pairs[:, 0], pairs[:, 1], couplings


def neighbour_arrays(n, i, j, couplings):
    """CSR neighbour lists for the numba kernels."""
    degree = np.bincount(np.concatenate([i, j]), minlength=n)
    pointer = np.zeros(n + 1, dtype=np.int64)
    pointer[1:] = np.cumsum(degree)
    index = np.empty(pointer[-1], dtype=np.int64)
    weight = np.empty(pointer[-1])
    fill = pointer[:-1].copy()
    for a, b, w in zip(i, j, couplings):
        index[fill[a]], weight[fill[a]] = b, w
        fill[a] += 1
        index[fill[b]], weight[fill[b]] = a, w
        fill[b] += 1
    return pointer, index, weight


def energy(spins, i, j, couplings):
    """``H = sum_(ij) J_ij s_i s_j``."""
    return float((couplings * spins[i] * spins[j]).sum())


# ------------------------------------------------------------- the kernels
#
# Every Sidon-28 coupling is a multiple of 1/28, so the kernels work in integer
# units of 1/28: exact energies, and Metropolis acceptance from a table per
# temperature instead of an exponential per attempt. Random numbers come from an
# inline xorshift64*.

UNIT = 28


def integer_weights(weight):
    q = np.rint(weight * UNIT).astype(np.int64)
    assert np.allclose(q / UNIT, weight)
    return q


def _kernels():
    from numba import njit

    @njit(cache=True, inline="always")
    def uniform(state):
        x = state[0]
        x ^= x >> np.uint64(12)
        x ^= x << np.uint64(25)
        x ^= x >> np.uint64(27)
        state[0] = x
        return float((x * np.uint64(2685821657736338717)) >> np.uint64(11)) * (1.0 / 9007199254740992.0)

    @njit(cache=True)
    def seeded(seed):
        state = np.empty(1, dtype=np.uint64)
        state[0] = np.uint64(seed * 2654435761 + 88172645463325252) | np.uint64(1)
        for _ in range(8):
            uniform(state)
        return state

    @njit(cache=True)
    def integer_energy(spins, pointer, index, weight):
        total = 0
        for a in range(spins.shape[0]):
            for t in range(pointer[a], pointer[a + 1]):
                total += weight[t] * spins[a] * spins[index[t]]
        return total // 2

    @njit(cache=True)
    def table(beta, size):
        """Metropolis acceptance for every energy change in ``[-size, size]``."""
        out = np.empty(2 * size + 1)
        for k in range(-size, size + 1):
            out[k + size] = 1.0 if k <= 0 else math.exp(-beta * k / 28.0)
        return out

    @njit(cache=True)
    def sweep(spins, accept, pointer, index, weight, state):
        """One Metropolis sweep in fixed order, branch-free; returns the energy
        change (1/28 units)."""
        size = (accept.shape[0] - 1) // 2
        change = 0
        for a in range(spins.shape[0]):
            field = 0
            for t in range(pointer[a], pointer[a + 1]):
                field += weight[t] * spins[index[t]]
            delta = -2 * spins[a] * field
            flip = uniform(state) < accept[delta + size]
            spins[a] = spins[a] * (1 - 2 * flip)
            change += delta * flip
        return change

    @njit(cache=True)
    def largest_delta(pointer, weight):
        top = 0
        for a in range(pointer.shape[0] - 1):
            row = 0
            for t in range(pointer[a], pointer[a + 1]):
                row += abs(weight[t])
            top = max(top, 2 * row)
        return top

    @njit(cache=True)
    def anneal(pointer, index, weight, betas, runs, seed):
        """Simulated annealing; final energies (1/28 units) of ``runs`` runs."""
        state = seeded(seed)
        n = pointer.shape[0] - 1
        size = largest_delta(pointer, weight)
        tables = np.empty((betas.shape[0], 2 * size + 1))
        for b in range(betas.shape[0]):
            tables[b] = table(betas[b], size)
        out = np.empty(runs, dtype=np.int64)
        spins = np.empty(n, dtype=np.int64)
        for r in range(runs):
            for a in range(n):
                spins[a] = 1 if uniform(state) < 0.5 else -1
            for b in range(betas.shape[0]):
                sweep(spins, tables[b], pointer, index, weight, state)
            out[r] = integer_energy(spins, pointer, index, weight)
        return out

    @njit(cache=True)
    def quantum_anneal(pointer, index, weight, slices, beta, gammas, runs, seed):
        """Path-integral simulated quantum annealing (discrete imaginary time).

        ``H = sum J s s - Gamma sum sigma^x`` at inverse temperature ``beta`` with
        ``slices`` Trotter slices, Gamma lowered along ``gammas``; one sweep is one
        Metropolis attempt per spin per slice. Returns, per run, the lowest
        classical energy (1/28 units) among the slices at the end, and the
        energy of slice 0 alone. Heim, Ronnow, Isakov & Troyer (Science 2015)
        showed that discrete time plus best-of-slices readout -- which no
        physical annealer can do -- manufactures SQA's apparent scaling
        advantage over SA; the single-slice readout is the honest one.
        """
        state = seeded(seed)
        n = pointer.shape[0] - 1
        size = largest_delta(pointer, weight)
        out = np.empty(runs, dtype=np.int64)
        single = np.empty(runs, dtype=np.int64)
        spins = np.empty((slices, n), dtype=np.int64)
        accept = np.empty((2 * size + 1, 3))
        for r in range(runs):
            for s in range(slices):
                for a in range(n):
                    spins[s, a] = 1 if uniform(state) < 0.5 else -1
            for gamma in gammas:
                coupling = -0.5 * math.log(math.tanh(beta * gamma / slices))
                for d in range(-size, size + 1):
                    for m in range(3):
                        accept[d + size, m] = math.exp(-(beta / slices * d / 28.0
                                                         + 2.0 * coupling * (2 * m - 2)))
                for s in range(slices):
                    up, down = (s + 1) % slices, (s - 1) % slices
                    for a in range(n):
                        field = 0
                        for t in range(pointer[a], pointer[a + 1]):
                            field += weight[t] * spins[s, index[t]]
                        delta = -2 * spins[s, a] * field
                        m = (spins[s, a] * (spins[up, a] + spins[down, a]) + 2) // 2
                        if uniform(state) < accept[delta + size, m]:
                            spins[s, a] = -spins[s, a]
            best = integer_energy(spins[0], pointer, index, weight)
            single[r] = best
            for s in range(1, slices):
                best = min(best, integer_energy(spins[s], pointer, index, weight))
            out[r] = best
        return out, single

    @njit(cache=True)
    def tempering(pointer, index, weight, betas, n_icm, checkpoints, runs, seed):
        """PT-ICM (Zhu, Ochoa & Katzgraber 2015): two replicas per temperature,
        Houdayer cluster moves between the pair at the ``n_icm`` coldest
        temperatures, then replica exchange. Runs ``checkpoints[-1]`` sweeps and
        returns, per run, the lowest energy seen by any replica up to each
        checkpoint (1/28 units), so any target can be evaluated afterwards."""
        state = seeded(seed)
        n = pointer.shape[0] - 1
        m = betas.shape[0]
        size = largest_delta(pointer, weight)
        tables = np.empty((m, 2 * size + 1))
        for k in range(m):
            tables[k] = table(betas[k], size)
        sweeps = checkpoints[-1]
        trace = np.empty((runs, checkpoints.shape[0]), dtype=np.int64)
        spins = np.empty((2, m, n), dtype=np.int64)
        energies = np.empty((2, m), dtype=np.int64)
        stack = np.empty(n, dtype=np.int64)
        seen = np.zeros(n, dtype=np.bool_)
        for r in range(runs):
            for c in range(2):
                for k in range(m):
                    for a in range(n):
                        spins[c, k, a] = 1 if uniform(state) < 0.5 else -1
                    energies[c, k] = integer_energy(spins[c, k], pointer, index, weight)
            low = energies[0, 0]
            mark = 0
            for step in range(sweeps):
                for c in range(2):
                    for k in range(m):
                        energies[c, k] += sweep(spins[c, k], tables[k], pointer, index, weight,
                                                state)
                for k in range(m - n_icm, m):
                    count = 0
                    for a in range(n):
                        if spins[0, k, a] != spins[1, k, a]:
                            count += 1
                    if count == 0 or count == n:
                        continue
                    pick = int(uniform(state) * count)
                    start = 0
                    for a in range(n):
                        if spins[0, k, a] != spins[1, k, a]:
                            if pick == 0:
                                start = a
                                break
                            pick -= 1
                    stack[0] = start
                    seen[start] = True
                    top, end = 0, 1
                    while top < end:
                        a = stack[top]
                        top += 1
                        for t in range(pointer[a], pointer[a + 1]):
                            b = index[t]
                            if not seen[b] and spins[0, k, b] != spins[1, k, b]:
                                seen[b] = True
                                stack[end] = b
                                end += 1
                    for q in range(end):
                        a = stack[q]
                        seen[a] = False
                        spins[0, k, a] = -spins[0, k, a]
                        spins[1, k, a] = -spins[1, k, a]
                    energies[0, k] = integer_energy(spins[0, k], pointer, index, weight)
                    energies[1, k] = integer_energy(spins[1, k], pointer, index, weight)
                for c in range(2):
                    for k in range(m - 1):
                        d = (betas[k + 1] - betas[k]) * (energies[c, k + 1] - energies[c, k]) / 28.0
                        if d >= 0.0 or uniform(state) < math.exp(d):
                            for a in range(n):
                                tmp = spins[c, k, a]
                                spins[c, k, a] = spins[c, k + 1, a]
                                spins[c, k + 1, a] = tmp
                            tmp_e = energies[c, k]
                            energies[c, k] = energies[c, k + 1]
                            energies[c, k + 1] = tmp_e
                for c in range(2):
                    for k in range(m):
                        low = min(low, energies[c, k])
                while mark < checkpoints.shape[0] and checkpoints[mark] == step + 1:
                    trace[r, mark] = low
                    mark += 1
        return trace

    return integer_energy, sweep, anneal, quantum_anneal, tempering


@lru_cache(maxsize=1)
def kernels():
    return _kernels()


# ------------------------------------------------------------ the metric

def repetitions(p):
    """Runs needed for 99% success; clipped at one run."""
    if p >= 0.99:
        return 1.0
    if p <= 0.0:
        return math.inf
    return max(1.0, math.log(1 - 0.99) / math.log(1 - p))


def tte(cost, p):
    return cost * repetitions(p)


def fit_exponent(sizes, values):
    """Power law ``c N^alpha`` by least squares in log-log."""
    x, y = np.log(np.asarray(sizes, float)), np.log(np.asarray(values, float))
    return float(np.polyfit(x, y, 1)[0])


# ------------------------------------------------------------ the runs

def betas_geometric(steps, low=PT_BETA_MIN, high=PT_BETA_MAX):
    return np.geomspace(low, high, steps)


def ground_energy(side, seed, effort=4):
    """Best energy from PT-ICM and SA runs; cached on disk.

    Two stages. The first (2 PT-ICM runs of ``2000 effort max(1, L//5)``
    sweeps, 32 SA runs) left PT-ICM and SA disagreeing on every instance from
    L = 8 up, by up to 0.7% -- enough to loosen a 1% target. The second adds
    two PT-ICM runs of 50 000 sweeps for L >= 8 and records whether they found
    anything lower, which is the evidence the reference is a ground state.
    """
    cache = RAW / f"ground_L{side:02d}_s{seed:03d}.json"
    record = json.loads(cache.read_text()) if cache.exists() else None
    _, _, anneal, _, tempering = kernels()
    n, i, j, couplings = instance(side, seed)
    pointer, index, weight = neighbour_arrays(n, i, j, couplings)
    weight = integer_weights(weight)
    betas = betas_geometric(PT_TEMPERATURES)
    if record is None:
        sweeps = 2000 * effort * max(1, side // 5)
        best_pt = tempering(pointer, index, weight, betas, PT_ICM, np.array([sweeps]), 2,
                            seed)[:, -1]
        sa = anneal(pointer, index, weight, betas_geometric(4000 * effort), 32, seed + 1)
        record = {"energy": float(min(best_pt.min(), sa.min())) / UNIT,
                  "pt": (best_pt / UNIT).tolist(), "sa_best": float(sa.min()) / UNIT,
                  "pt_sweeps": sweeps}
    if side >= 8 and "refined" not in record:
        long_runs = tempering(pointer, index, weight, betas, PT_ICM, np.array([50_000]), 2,
                              seed + 7919)[:, -1] / UNIT
        record["refined"] = long_runs.tolist()
        record["refinement_improved"] = bool(long_runs.min() < record["energy"] - 1e-9)
        record["first_stage_energy"] = record["energy"]
        record["energy"] = float(min(record["energy"], long_runs.min()))
    RAW.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(record))
    return record["energy"]


def solve_one(task):
    """One (method, side, seed): raw final energies across the effort grid.

    Energies, not success rates, are stored, so targets can be re-evaluated
    against any reference energy and any gap.
    """
    method, side, seed, grid, runs = task
    _, _, anneal, quantum_anneal, tempering = kernels()
    n, i, j, couplings = instance(side, seed)
    pointer, index, weight = neighbour_arrays(n, i, j, couplings)
    weight = integer_weights(weight)
    rows = []
    started = time.perf_counter()
    if method == "sa":
        for sweeps in grid:
            final = anneal(pointer, index, weight,
                           betas_geometric(sweeps, SA_BETA_MIN, SA_BETA_MAX), runs,
                           1000 * seed + sweeps)
            rows.append({"sweeps": sweeps, "updates": sweeps * n, "energies": final.tolist()})
    elif method == "sqa":
        for sweeps in grid:
            gammas = np.linspace(SQA_GAMMA, 1e-3, sweeps)
            best, single = quantum_anneal(pointer, index, weight, SQA_SLICES, SQA_BETA, gammas,
                                          runs, 1000 * seed + sweeps)
            rows.append({"sweeps": sweeps, "updates": sweeps * n * SQA_SLICES,
                         "energies": best.tolist(), "single": single.tolist()})
    elif method == "pt_icm":
        betas = betas_geometric(PT_TEMPERATURES)
        trace = tempering(pointer, index, weight, betas, PT_ICM, np.array(grid), runs, seed)
        per_sweep = 2 * PT_TEMPERATURES * n
        for g, sweeps in enumerate(grid):
            rows.append({"sweeps": sweeps, "updates": sweeps * per_sweep,
                         "energies": trace[:, g].tolist()})
    return {"method": method, "side": side, "seed": seed, "n": n,
            "seconds": time.perf_counter() - started, "rows": rows}


def with_success(results, epsilon, readout="energies"):
    """Attach ``p`` to every row against each instance's reference energy."""
    out = []
    for r in results:
        e0 = ground_energy(r["side"], r["seed"])
        target = math.floor(round((e0 + epsilon * abs(e0)) * UNIT, 6))
        rows = [{"sweeps": row["sweeps"], "updates": row["updates"],
                 "p": float(np.mean(np.array(row[readout]) <= target))} for row in r["rows"]]
        out.append(dict(r, rows=rows, method=r["method"] + ("" if readout == "energies"
                                                             else "_" + readout)))
    return out


def median_tte(results, side):
    """Optimise the per-run effort for the median TTe over instances at one size."""
    mine = [r for r in results if r["side"] == side]
    grid = [row["sweeps"] for row in mine[0]["rows"]]
    best = None
    for g, sweeps in enumerate(grid):
        values = [tte(r["rows"][g]["updates"], r["rows"][g]["p"]) for r in mine]
        median = float(np.median(values))
        if best is None or median < best[0]:
            best = (median, sweeps, values)
    return best


#: the upper half of the size range: L >= 10, N >= 600. The paper fits QAC only
#: on sizes whose optimal annealing time is resolved, which are the larger ones
LARGE = 600


def summarise(results, method, boot=300):
    """Median TTe per size with the effort optimised per size, a power-law fit
    over all sizes and over N >= LARGE, and bootstrap intervals for both
    (instances resampled within each size, effort re-optimised per resample)."""
    mine = [r for r in results if r["method"] == method]
    sides = sorted({r["side"] for r in mine})
    sizes, medians, chosen, at_edge = [], [], [], []
    for side in sides:
        median, sweeps, _ = median_tte(mine, side)
        grid = [row["sweeps"] for row in next(r for r in mine if r["side"] == side)["rows"]]
        sizes.append(next(r["n"] for r in mine if r["side"] == side))
        medians.append(median)
        chosen.append(sweeps)
        at_edge.append(sweeps in (grid[0], grid[-1]))

    def fits(points):
        full = [(n, m) for n, m in points if math.isfinite(m)]
        large = [(n, m) for n, m in full if n >= LARGE]
        return (fit_exponent(*zip(*full)),
                fit_exponent(*zip(*large)) if len(large) >= 3 else math.nan)

    alpha, alpha_large = fits(list(zip(sizes, medians)))
    rng = np.random.default_rng(0)
    draws = []
    by_side = {side: [r for r in mine if r["side"] == side] for side in sides}
    for _ in range(boot):
        points = []
        for side in sides:
            subset = by_side[side]
            sample = [subset[k] for k in rng.integers(0, len(subset), len(subset))]
            points.append((subset[0]["n"], median_tte(sample, side)[0]))
        draws.append(fits(points))
    draws = np.array(draws)

    def interval(column):
        values = draws[:, column][np.isfinite(draws[:, column])]
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {"method": method, "sizes": sizes, "median_tte_updates": medians,
            "optimal_sweeps": chosen, "optimum_at_grid_edge": at_edge,
            "alpha": alpha, "alpha_2se": 2 * float(np.nanstd(draws[:, 0])),
            "alpha_bootstrap_95": interval(0),
            "alpha_large": alpha_large, "alpha_large_2se": 2 * float(np.nanstd(draws[:, 1])),
            "alpha_large_bootstrap_95": interval(1)}


GRIDS = {
    "sa": tuple(2 ** k for k in range(6, 15)),
    "sqa": tuple(2 ** k for k in range(4, 12)),
    "pt_icm": tuple(int(v) for v in np.unique(np.geomspace(1, 2048, 23).astype(int))),
}
RUNS = {"sa": 64, "sqa": 32, "pt_icm": 20}
#: the paper's gaps: 1% is the headline, 1.25% and 1.5% where QA's exponent falls further
EPSILONS = (0.01, 0.0125, 0.015)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--methods", default="sa,pt_icm,sqa")
    ap.add_argument("--grounds-only", action="store_true")
    ap.add_argument("--skip-grounds", action="store_true",
                    help="solvers only; summarise later with --summarise")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    sides = [int(s) for s in args.sizes.split(",")]
    methods = args.methods.split(",")

    if not args.summarise:
        import multiprocessing as mp

        tasks = [(m, side, seed, GRIDS[m], RUNS[m]) for m in methods for side in sides
                 for seed in range(args.instances)
                 if not (RAW / f"{m}_L{side:02d}_s{seed:03d}.json").exists()]
        # reference energies, largest first; solvers store raw energies and never
        # need them, so the two can run as separate invocations
        grounds = [(side, seed) for side in sorted(sides, reverse=True)
                   for seed in range(args.instances)]
        if args.skip_grounds:
            grounds = []
        if args.grounds_only:
            tasks = []
        with mp.Pool(args.workers) as pool:
            for _ in pool.imap_unordered(_ground_task, grounds):
                pass
            for result in pool.imap_unordered(solve_one, tasks):
                path = RAW / f"{result['method']}_L{result['side']:02d}_s{result['seed']:03d}.json"
                path.write_text(json.dumps(result))
                print(f"  {result['method']:>6} L={result['side']:>2} s{result['seed']:<3} "
                      f"({result['seconds']:.1f}s)", flush=True)

    if args.skip_grounds or args.grounds_only:
        return 0
    raw = [json.loads(p.read_text()) for p in sorted(RAW.glob("*_L*_s*.json"))
           if not p.name.startswith("ground")]
    summary = {}
    for epsilon in EPSILONS:
        evaluated = []
        for m in methods:
            mine = [r for r in raw if r["method"] == m]
            if not mine:
                continue
            evaluated += with_success(mine, epsilon)
            if m == "sqa":
                evaluated += with_success(mine, epsilon, readout="single")
        for m in sorted({r["method"] for r in evaluated}):
            s = summarise(evaluated, m)
            summary[f"{m}@{epsilon:g}"] = s
            print(f"\n  {m} @ e = {epsilon:g}: alpha = {s['alpha']:.2f} +- {s['alpha_2se']:.2f} "
                  f"(95% {s['alpha_bootstrap_95'][0]:.2f}..{s['alpha_bootstrap_95'][1]:.2f});"
                  f"  N >= {LARGE}: {s['alpha_large']:.2f} +- {s['alpha_large_2se']:.2f}")
            for n, med, sw, edge in zip(s["sizes"], s["median_tte_updates"], s["optimal_sweeps"],
                                        s["optimum_at_grid_edge"]):
                print(f"    N={n:>5}  median TTe {med:12.4g} spin updates  optimal sweeps {sw:>5}"
                      f"{'  (grid edge)' if edge else ''}")
    grounds = [json.loads(p.read_text()) for p in sorted(RAW.glob("ground_L*_s*.json"))]
    refined = [g for g in grounds if "refined" in g]
    print(f"\n  ground states: {len(refined)} refined with 2 x 50 000 PT-ICM sweeps; "
          f"{sum(g['refinement_improved'] for g in refined)} improved on the first stage")
    path = RESULTS_DIR / "exp041_dwave_qac_scaling.json"
    path.write_text(json.dumps({"epsilons": EPSILONS, "instances": args.instances,
                                "summary": summary,
                                "ground_refined": len(refined),
                                "ground_improved_by_refinement":
                                    sum(g["refinement_improved"] for g in refined)},
                               indent=1, default=float))
    print(f"saved -> {path}")
    return 0


def _ground_task(key):
    return ground_energy(*key)


if __name__ == "__main__":
    raise SystemExit(main())
