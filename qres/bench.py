"""Benchmark harness: run configurations over seeds, aggregate honestly.

Single-seed VQE numbers are close to meaningless -- the spread across random
initialisations and shot-noise realisations is routinely larger than the effect
being measured.  Everything here runs a configuration over several seeds and
reports the median with an interquartile range, plus a paired comparison
against a named baseline (paired because the same seed means the same
initial point and the same simulator stream, which removes most of the
variance from the difference).
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class Aggregate:
    """Summary of one configuration across seeds."""

    label: str
    n: int
    values: dict[str, list[float]] = field(default_factory=dict)

    def stat(self, key: str) -> dict[str, float]:
        v = np.asarray(self.values.get(key, []), dtype=float)
        if v.size == 0:
            return {}
        return {
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)),
            "min": float(v.min()),
            "max": float(v.max()),
            "n": int(v.size),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "stats": {k: self.stat(k) for k in self.values},
            "raw": self.values,
        }


def aggregate_runs(label: str, runs: Sequence[Any], keys: Sequence[str]) -> Aggregate:
    agg = Aggregate(label=label, n=len(runs))
    for key in keys:
        agg.values[key] = [float(_dig(r, key)) for r in runs]
    return agg


def _dig(obj, dotted: str):
    """Fetch ``a.b.c`` from nested dataclasses/dicts."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def paired_comparison(
    baseline: Sequence[float], candidate: Sequence[float], lower_is_better: bool = True
) -> dict[str, float]:
    """Seed-paired comparison of two configurations.

    Returns the median ratio and a sign-test p-value.  The sign test is used
    rather than a t-test because VQE error distributions are heavy-tailed and
    nowhere near normal, and because with 5-20 seeds a t-test's assumptions do
    more harm than its extra power does good.
    """
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(candidate, dtype=float)
    n = min(len(b), len(c))
    b, c = b[:n], c[:n]
    if n == 0:
        return {}

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b != 0, c / b, np.nan)

    wins = int(np.sum(c < b)) if lower_is_better else int(np.sum(c > b))
    ties = int(np.sum(c == b))
    trials = n - ties
    p = _sign_test_p(wins, trials)

    return {
        "n_pairs": n,
        "median_ratio": float(np.nanmedian(ratio)),
        "median_baseline": float(np.median(b)),
        "median_candidate": float(np.median(c)),
        "wins": wins,
        "losses": trials - wins,
        "ties": ties,
        "sign_test_p": p,
    }


def _sign_test_p(wins: int, trials: int) -> float:
    """Two-sided exact binomial p-value under H0: p(win) = 1/2."""
    if trials == 0:
        return 1.0
    from math import comb

    k = min(wins, trials - wins)
    tail = sum(comb(trials, i) for i in range(0, k + 1)) / 2**trials
    return float(min(1.0, 2 * tail))


@dataclass
class Study:
    """A named set of configurations run over a shared list of seeds."""

    name: str
    seeds: Sequence[int]
    metrics: Sequence[str] = (
        "noiseless_error",
        "error",
        "ledger.shots",
        "ledger.circuits",
        "ledger.hardware_seconds",
        "wall_seconds",
    )
    results: dict[str, list] = field(default_factory=dict)
    raw: list[dict] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def add(self, label: str, runs: Sequence[Any]) -> None:
        self.results[label] = list(runs)
        for r in runs:
            self.raw.append(r.to_dict() if hasattr(r, "to_dict") else dict(r))

    def aggregates(self) -> dict[str, Aggregate]:
        return {
            label: aggregate_runs(label, runs, self.metrics)
            for label, runs in self.results.items()
        }

    def table(self, sort_by: str = "noiseless_error") -> str:
        aggs = self.aggregates()
        rows = sorted(aggs.values(), key=lambda a: a.stat(sort_by).get("median", np.inf))
        head = (
            f"{'configuration':<34}{'err(med)':>11}{'err(IQR)':>21}"
            f"{'shots':>12}{'hw_s':>10}{'wall_s':>9}"
        )
        lines = [head, "-" * len(head)]
        for a in rows:
            e = a.stat("noiseless_error")
            s = a.stat("ledger.shots")
            h = a.stat("ledger.hardware_seconds")
            w = a.stat("wall_seconds")
            iqr = "[{:.2e}, {:.2e}]".format(e["q1"], e["q3"])
            lines.append(
                f"{a.label:<34}{e['median']:>11.3e}{iqr:>21}"
                f"{s['median']:>12,.0f}{h['median']:>10.1f}{w['median']:>9.1f}"
            )
        return "\n".join(lines)

    def compare(self, baseline: str, metric: str = "noiseless_error") -> str:
        aggs = self.aggregates()
        if baseline not in aggs:
            return f"(no baseline {baseline!r})"
        base = aggs[baseline].values[metric]
        lines = [f"paired vs {baseline!r} on {metric} (ratio < 1 is better):"]
        for label, agg in aggs.items():
            if label == baseline:
                continue
            cmp = paired_comparison(base, agg.values[metric])
            if not cmp:
                continue
            lines.append(
                f"  {label:<32} ratio={cmp['median_ratio']:.3f}  "
                f"W/L/T={cmp['wins']}/{cmp['losses']}/{cmp['ties']}  "
                f"p={cmp['sign_test_p']:.3f}"
            )
        return "\n".join(lines)

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or RESULTS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        payload = {
            "study": self.name,
            "seeds": list(self.seeds),
            "elapsed_seconds": time.time() - self.started,
            "aggregates": {k: v.to_dict() for k, v in self.aggregates().items()},
            "runs": self.raw,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path


def run_over_seeds(
    fn: Callable,
    seeds: Iterable[int],
    workers: int | None = None,
    **kwargs,
) -> list:
    """Run ``fn(seed=s, **kwargs)`` for each seed, in parallel where useful.

    Serial by default, and that is a considered choice.  Aer holds an OpenMP
    thread pool, so a forked worker deadlocks -- which presents as a study that
    never produces its first row.  ``spawn`` avoids the deadlock but costs a
    fresh interpreter (~4 s of qiskit imports) per worker per call, which
    exceeds the runtime of a small run.  Aer also already threads internally,
    so process-level parallelism mostly oversubscribes the machine and
    distorts the wall-clock numbers we record.

    Set ``QRES_WORKERS`` or pass ``workers`` to opt in for studies where a
    single run costs minutes rather than seconds; parallelising *across
    experiment processes* is usually the better lever.
    """
    seeds = list(seeds)
    if workers is None:
        workers = int(os.environ.get("QRES_WORKERS", "1"))
    workers = min(workers, len(seeds), os.cpu_count() or 1)
    if workers <= 1 or len(seeds) == 1:
        return [fn(seed=s, **kwargs) for s in seeds]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(fn, seed=s, **kwargs) for s in seeds]
        return [f.result() for f in futures]
