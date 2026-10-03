"""Experiment 040 -- does the IonQ + Ansys pipeline beat METIS where it is measured?

Result 93 showed that every partition VarQITE handed to LS-DYNA is computed
exactly by a classical solver, so the paper's 12% is a pipeline against a
pipeline. That leaves the pipeline itself: coarsen to <= 32 vertices, bisect
exactly, refine on the way back up. The paper compares it with LS-GPart. This
compares it with METIS, the standard open partitioner, by the paper's own merit
metrics: **the non-zeros of the Cholesky factor and the flops to compute it**.

Orderings compared, on FEA-type meshes of 24 000 - 31 000 vertices:

* ``metis_nd`` -- METIS's own nested dissection, end to end (the reference);
* ``metis_top`` -- top-level bisection by METIS;
* ``exact32`` -- the quantum step's best case: the exact optimum of the paper's
  QUBO on the 32-vertex coarse graph, refined multilevel by FM;
* ``exact32_best_of_10`` -- the paper's own selection: of the 10 lowest-energy
  coarse partitions, the one whose ordering is cheapest (they chose by wall
  clock, the closest thing measurable here);
* ``exact256`` -- an exact balanced bisection of a 256-vertex coarse graph (MILP),
  a resolution the quantum step cannot reach.

Every non-reference ordering uses the same construction below the top level --
minimum vertex separator of the cut edges (Koenig), each half ordered by METIS
nested dissection, separator last -- so the orderings differ only in the top
bisection, which is exactly the part the quantum computer supplied.

Run:  python -m experiments.exp040_nested_dissection_fill [--seeds 3]
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp039_ionq_ansys_partitioning import (
    NU,
    coarsen,
    exhaustive,
    housing_mesh,
    milp_bisection,
    penalty_weight,
    shell_mesh,
    unstructured_mesh,
)
from qres.bench import RESULTS_DIR


def meshes():
    """The three mesh types of Result 93, about four times larger."""
    return {"shell": shell_mesh(nx=192, ny=128),
            "housing": housing_mesh(nr=4, nt=96, nz=64),
            "unstructured": unstructured_mesh(points=24000)}


# ------------------------------------------------------- symbolic Cholesky

def symbolic_cholesky(graph, order):
    """Non-zeros of ``L`` and flops of the factorisation for ``A[order][:, order]``.

    Elimination tree by Liu's algorithm with path compression, then row
    subtrees: row ``i`` of ``L`` is the union of the tree paths from each
    ``k < i`` with ``A_ik != 0`` up to ``i``. ``O(nnz(L))``. Flops are
    ``sum_j c_j^2`` over column counts ``c_j`` (diagonal included).
    """
    n = graph.shape[0]
    permuted = graph[order][:, order]
    lower = sp.tril(permuted, k=-1).tocsr()
    indptr, indices = lower.indptr, lower.indices
    parent = np.full(n, -1)
    ancestor = np.full(n, -1)
    for i in range(n):
        for k in indices[indptr[i]:indptr[i + 1]]:
            r = k
            while ancestor[r] != -1 and ancestor[r] != i:
                following = ancestor[r]
                ancestor[r] = i
                r = following
            if ancestor[r] == -1:
                ancestor[r] = i
                parent[r] = i
    parent_list = parent.tolist()
    mark = [-1] * n
    counts = [1] * n
    for i in range(n):
        mark[i] = i
        for k in indices[indptr[i]:indptr[i + 1]].tolist():
            j = k
            while mark[j] != i:
                mark[j] = i
                counts[j] += 1
                j = parent_list[j]
    counts = np.array(counts, dtype=np.int64)
    return int(counts.sum()), int((counts.astype(float) ** 2).sum())


# ------------------------------------------------------------- refinement

def fm_refine(graph, weights, x, nu=NU, max_passes=8, patience=200):
    """Fiduccia-Mattheyses on a sparse graph, moving from the heavier side.

    The paper's rule (their Algorithm 1). A pass moves vertices best-gain first,
    locks each, and keeps the best prefix that respects the balance -- cut first,
    imbalance as the tie-break; a pass gives up after ``patience`` moves without
    improving. Passes repeat until one does not improve.
    """
    x = np.asarray(x, dtype=np.int64).copy()
    weights = np.asarray(weights, dtype=np.int64)
    total = int(weights.sum())
    limit = (0.5 + nu) * total
    indptr, indices, data = graph.indptr, graph.indices, graph.data

    def score(side_one, cut):
        excess = max(side_one, total - side_one) - limit
        return (max(0.0, excess), cut)

    for _ in range(max_passes):
        sign = 1 - 2 * x
        gain = (-(graph @ sign) * sign).astype(np.int64)
        side_one = int(weights @ x)
        cut = _cut(graph, x)
        heaps = ([], [])
        for v in range(len(x)):
            heaps[x[v]].append((-gain[v], v))
        for h in heaps:
            heapq.heapify(h)
        locked = np.zeros(len(x), bool)
        start = score(side_one, cut)
        best, best_moves, moves, since = start, 0, [], 0
        while True:
            heavier = 1 if side_one > total - side_one else 0
            heap = heaps[heavier]
            v = -1
            while heap:
                g, u = heapq.heappop(heap)
                if not locked[u] and x[u] == heavier and -g == gain[u]:
                    v = u
                    break
            if v < 0:
                break
            cut -= int(gain[v])
            side_one += int(weights[v]) * (1 - 2 * int(x[v]))
            old = x[v]
            for t in range(indptr[v], indptr[v + 1]):
                u, w = indices[t], data[t]
                gain[u] += 2 * w if x[u] == old else -2 * w
                if not locked[u]:
                    heapq.heappush(heaps[x[u]], (-gain[u], u))
            gain[v] = -gain[v]
            x[v] = 1 - old
            locked[v] = True
            moves.append(v)
            current = score(side_one, cut)
            if current < best:
                best, best_moves, since = current, len(moves), 0
            else:
                since += 1
                if since >= patience:
                    break
        for v in moves[best_moves:]:
            x[v] = 1 - x[v]
        if best >= start:
            break
    return x


def _cut(graph, x):
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    return int((graph.data * (x[rows] != x[graph.indices])).sum() // 2)


def multilevel(graph, coarse_x, levels, nu=NU):
    """Project a coarse bisection back through the hierarchy, refining each level."""
    x = np.asarray(coarse_x, dtype=np.int64)
    for level_graph, level_weights, group in reversed(levels):
        x = fm_refine(level_graph, level_weights, x[group], nu)
    return x


# ------------------------------------------------------- ordering pieces

def vertex_separator(graph, x):
    """Minimum vertex cover of the cut edges (Koenig), so no edge joins the halves."""
    from scipy.sparse.csgraph import maximum_bipartite_matching

    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    cols = graph.indices
    crossing = (x[rows] == 0) & (x[cols] == 1)
    left = np.unique(rows[crossing])
    right = np.unique(cols[crossing])
    if len(left) == 0:
        return np.array([], dtype=np.int64)
    li = {v: i for i, v in enumerate(left)}
    ri = {v: i for i, v in enumerate(right)}
    bip = sp.csr_matrix((np.ones(crossing.sum()),
                         ([li[v] for v in rows[crossing]], [ri[v] for v in cols[crossing]])),
                        shape=(len(left), len(right)))
    match_right = maximum_bipartite_matching(bip, perm_type="column")  # right matched to each left
    match_left = np.full(len(right), -1)
    for i, j in enumerate(match_right):
        if j >= 0:
            match_left[j] = i
    # alternating BFS from unmatched left vertices
    seen_left = np.zeros(len(left), bool)
    seen_right = np.zeros(len(right), bool)
    queue = [i for i in range(len(left)) if match_right[i] < 0]
    for i in queue:
        seen_left[i] = True
    while queue:
        i = queue.pop()
        for j in bip.indices[bip.indptr[i]:bip.indptr[i + 1]]:
            if not seen_right[j]:
                seen_right[j] = True
                k = match_left[j]
                if k >= 0 and not seen_left[k]:
                    seen_left[k] = True
                    queue.append(k)
    return np.concatenate([left[~seen_left], right[seen_right]]).astype(np.int64)


def metis_nd(graph):
    """METIS nested dissection; returns the elimination order (new -> old)."""
    import pymetis

    if graph.shape[0] <= 2:
        return np.arange(graph.shape[0])
    perm, _ = pymetis.nested_dissection(adjacency=pymetis.CSRAdjacency(graph.indptr,
                                                                       graph.indices))
    return np.asarray(perm, dtype=np.int64)


def ordering_from_bisection(graph, x):
    """Separator last, each half ordered by METIS nested dissection."""
    separator = vertex_separator(graph, x)
    keep = np.ones(graph.shape[0], bool)
    keep[separator] = False
    parts = []
    for side in (0, 1):
        members = np.where(keep & (x == side))[0]
        sub = graph[members][:, members].tocsr()
        parts.append(members[metis_nd(sub)])
    return np.concatenate(parts + [separator]), len(separator)


def metis_bisection_fine(graph, nu=NU, seed=0):
    import pymetis

    options = pymetis.Options(ufactor=round(2000 * nu), seed=seed)
    result = pymetis.part_graph(2, adjacency=pymetis.CSRAdjacency(graph.indptr, graph.indices),
                                options=options)
    return np.array(result.vertex_part, dtype=np.int64)


# ---------------------------------------------------------------- the runs

def measure(graph, order, separator=None, x=None):
    started = time.perf_counter()
    nnz, flops = symbolic_cholesky(graph, order)
    return {"nnz_L": nnz, "flops": flops, "separator": separator,
            "top_cut": None if x is None else _cut(graph, x),
            "imbalance": None if x is None else float(abs(x.mean() - 0.5) * 2),
            "symbolic_seconds": time.perf_counter() - started}


def run_mesh(kind, graph, seeds):
    graph = graph.tocsr().astype(np.int64)
    rows = []
    reference = measure(graph, metis_nd(graph))
    print(f"\n  {kind}: {graph.shape[0]} vertices, {graph.nnz // 2} edges; "
          f"METIS ND nnz(L) {reference['nnz_L']:,}  flops {reference['flops']:.3e}", flush=True)
    for seed in range(seeds):
        row = {"mesh": kind, "seed": seed, "metis_nd": reference}

        x = metis_bisection_fine(graph, seed=seed)
        order, size = ordering_from_bisection(graph, x)
        row["metis_top"] = measure(graph, order, size, x)

        coarse, weights, levels = coarsen(graph, 32, seed, return_levels=True)
        started = time.perf_counter()
        exact = exhaustive(coarse, weights, penalty_weight(coarse, weights))
        enumerate_seconds = time.perf_counter() - started
        candidates = []
        for rank, partition in enumerate(exact["partitions"]):
            fine = multilevel(graph, partition, levels)
            order, size = ordering_from_bisection(graph, fine)
            result = measure(graph, order, size, fine)
            result["energy_rank"] = rank
            candidates.append(result)
        row["exact32"] = dict(candidates[0], enumerate_seconds=enumerate_seconds)
        row["exact32_best_of_10"] = min(candidates, key=lambda r: r["flops"])
        row["exact32_all"] = [{k: c[k] for k in ("nnz_L", "flops", "separator", "energy_rank")}
                              for c in candidates]

        coarse, weights, levels = coarsen(graph, 256, seed, return_levels=True)
        started = time.perf_counter()
        partition = milp_bisection(coarse, weights)
        milp_seconds = time.perf_counter() - started
        fine = multilevel(graph, partition, levels)
        order, size = ordering_from_bisection(graph, fine)
        row["exact256"] = dict(measure(graph, order, size, fine), milp_seconds=milp_seconds)

        rows.append(row)
        print(f"    seed {seed}: " + "  ".join(
            f"{name} {row[name]['flops'] / reference['flops']:.3f}"
            for name in ("metis_top", "exact32", "exact32_best_of_10", "exact256")),
            "(flops relative to METIS ND)", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--meshes", default="shell,housing,unstructured")
    args = ap.parse_args()
    print("=== experiment 040 :: nested-dissection fill, exact-coarse pipeline against METIS ===")
    all_meshes = meshes()
    rows = []
    for kind in args.meshes.split(","):
        rows += run_mesh(kind, all_meshes[kind], args.seeds)
    summary = {}
    for name in ("metis_top", "exact32", "exact32_best_of_10", "exact256"):
        ratios = [r[name]["flops"] / r["metis_nd"]["flops"] for r in rows]
        nnz = [r[name]["nnz_L"] / r["metis_nd"]["nnz_L"] for r in rows]
        summary[name] = {"flops_vs_metis_nd": {"min": min(ratios), "median": float(np.median(ratios)),
                                               "max": max(ratios)},
                         "nnz_vs_metis_nd": {"min": min(nnz), "median": float(np.median(nnz)),
                                             "max": max(nnz)}}
    print(json.dumps(summary, indent=2))
    path = RESULTS_DIR / "exp040_nested_dissection_fill.json"
    with open(path, "w") as fh:
        json.dump({"nu": NU, "summary": summary, "rows": rows}, fh, indent=1, default=float)
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
