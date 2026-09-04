"""Experiment 017 -- the time dimension, and a retraction of this project's own claim.

Result 70 and the README both ended on this sentence: a drug-sized molecule needs
1.5e8 T gates, and "one factory feeding them sequentially is months of
wall-clock".  **That number was never computed.**  It was written as a plausible
closing caveat, and it is the only quantity in this repository that was asserted
rather than measured.  This experiment does the arithmetic.

The answer is that it was wrong by roughly two orders of magnitude, in the
direction that makes the conclusion *less* pessimistic: one factory is ~10 hours,
and about ten factories bring it to a floor near one hour.

**This is an estimate, not a measurement**, in exactly the sense exp016 is: it
combines measured inputs (T counts from transpiled ansaetze, code distance from
the measured physical error rate) with a standard model:

    surface-code cycle time            ~1 us  (superconducting, measurement-limited)
    15-to-1 factory output period      ~10 d cycles
    T-gate teleportation into the data block  ~d cycles

Two distinct limits fall out, and which one binds is the whole point:

* **factory-limited** -- how fast magic states are produced.  Parallel factories
  divide this, at a linear cost in qubits.
* **data-block-limited** -- how fast the algorithm can *consume* them.  UCCSD is
  an essentially sequential chain of rotations, so its T-depth is close to its
  T-count, and no number of factories goes below this floor.

The floor is the honest headline: it is what parallelism cannot buy.

**Which way the floor assumption errs.**  Taking T-depth = T-count assumes *no*
two rotations run concurrently, so the floor computed here is an **upper bound on
the floor**.  Excitations on disjoint orbitals could in principle be parallelised,
which would push the floor down -- i.e. the honest error bar on "59 minutes" runs
towards *faster*, not slower, and the conclusion is safe in the direction it is
being used.

**What this does NOT settle.**  Everything here is the cost of *one circuit
execution*.  A variational loop needs one per energy evaluation and a phase
estimation needs repetitions for its own precision; that multiplier is a
different question and this model deliberately does not guess at it.  It is
reported as an explicit knob so the reader supplies their own.

Run:  python -m experiments.exp017_magic_state_time
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.exp016_error_correction_overhead import (
    CHEMICAL_ACCURACY,
    FACTORY_LOGICAL_QUBITS,
    MOLECULES,
    distillation_rounds,
    required_distance,
    t_count,
)
from qres.bench import RESULTS_DIR

#: surface-code cycle time.  Superconducting hardware is measurement-limited at
#: roughly a microsecond per round; trapped ions are ~1000x slower, which is why
#: the sensitivity sweep below spans three orders of magnitude.
CYCLE_TIME_SECONDS = 1e-6

#: A 15-to-1 factory emits one output state every ~10 code cycles per unit
#: distance.  Litinski's constructions land between 6d and 20d depending on
#: layout, so that whole range is swept rather than a single value defended.
FACTORY_PERIOD_IN_DISTANCES = 10.0

#: Teleporting one T gate into the data block costs ~d cycles.  With a sequential
#: algorithm this sets a floor no amount of distillation parallelism goes below.
CONSUMPTION_IN_DISTANCES = 1.0

#: the drug-sized extrapolation carried through exp016 and RESEARCH_LOG
DRUG_ORBITALS = 50
DRUG_ROTATIONS = 1_665_043
DRUG_LOGICAL_QUBITS = 2 * DRUG_ORBITALS - 2
DRUG_GATES = 1_471 * (DRUG_ORBITALS / 4) ** 5.12

BEST_PHYSICAL_ERROR = 0.00127


def factory_limited_seconds(
    t_gates: int,
    distance: int,
    factories: int = 1,
    period: float = FACTORY_PERIOD_IN_DISTANCES,
    cycle: float = CYCLE_TIME_SECONDS,
) -> float:
    """Wall-clock if magic-state *production* is the bottleneck."""
    if factories < 1:
        raise ValueError("factories must be >= 1")
    return t_gates * period * distance * cycle / factories


def consumption_floor_seconds(
    t_gates: int,
    distance: int,
    consumption: float = CONSUMPTION_IN_DISTANCES,
    cycle: float = CYCLE_TIME_SECONDS,
) -> float:
    """Wall-clock floor set by the data block, for a sequential algorithm.

    No number of factories goes below this: the T gates lie on the algorithm's
    critical path, so they are consumed one after another regardless of how many
    are waiting.
    """
    return t_gates * consumption * distance * cycle


#: The two limits meet at an exact equality (`period / factories == consumption`),
#: and both sides are the same product computed in a different order -- so on the
#: boundary the comparison is decided by the last bit of the mantissa rather than
#: by physics.  Left strict, `saturating_factories` returned 11 instead of 10 for
#: some (T count, distance) pairs and the `binding` label could flip arbitrarily.
#: A relative tolerance well above float noise and far below any real difference
#: makes the boundary deterministic.
_BOUNDARY_TOLERANCE = 1e-9


def runtime(
    t_gates: int, distance: int, factories: int = 1, **kwargs
) -> tuple[float, str]:
    """Wall-clock and which of the two limits is binding."""
    produced = factory_limited_seconds(t_gates, distance, factories, **{
        k: v for k, v in kwargs.items() if k in {"period", "cycle"}
    })
    floor = consumption_floor_seconds(t_gates, distance, **{
        k: v for k, v in kwargs.items() if k in {"consumption", "cycle"}
    })
    if produced > floor * (1.0 + _BOUNDARY_TOLERANCE):
        return produced, "factory"
    return floor, "data block"


def saturating_factories(
    t_gates: int,
    distance: int,
    period: float = FACTORY_PERIOD_IN_DISTANCES,
    consumption: float = CONSUMPTION_IN_DISTANCES,
    limit: int = 4096,
) -> int:
    """Fewest factories that reach the data-block floor.

    Closed form rather than a search: the condition is
    ``t * period * d / k <= t * consumption * d``, and every factor except
    ``period``, ``consumption`` and ``k`` cancels.  So this is
    ``ceil(period / consumption)`` and depends on **neither the molecule nor the
    code distance** -- see `tests/test_magic_state_time.py`, which exists to stop
    that being read as a finding.  The arguments are kept for call-site clarity.
    """
    if consumption <= 0:
        raise ValueError("consumption must be positive")
    needed = int(np.ceil(period / consumption - _BOUNDARY_TOLERANCE))
    return max(1, min(needed, limit))


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400 * 2:
        return f"{seconds / 3600:.1f} h"
    if seconds < 86400 * 400:
        return f"{seconds / 86400:.1f} days"
    return f"{seconds / (86400 * 365.25):.1f} years"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle-time", type=float, default=CYCLE_TIME_SECONDS,
                    help="surface-code cycle time in seconds")
    ap.add_argument("--repetitions", type=int, default=1,
                    help="circuit executions the algorithm needs (see docstring)")
    args = ap.parse_args()

    cycle = args.cycle_time
    physical = BEST_PHYSICAL_ERROR

    print("=== experiment 017 :: the time dimension ===")
    print("ESTIMATE, not a measurement: measured T counts and code distances")
    print("combined with a standard surface-code timing model")
    print(f"cycle time {cycle:.0e} s, factory period {FACTORY_PERIOD_IN_DISTANCES:g}d, "
          f"consumption {CONSUMPTION_IN_DISTANCES:g}d\n")

    cases = [
        (name, qubits, gates, rotations)
        for name, _orbitals, qubits, gates, rotations in MOLECULES
        if rotations > 0
    ]
    cases.append(("drug/50", DRUG_LOGICAL_QUBITS, DRUG_GATES, DRUG_ROTATIONS))

    header = (
        f"{'case':<10}{'T gates':>14}{'dist':>6}{'1 factory':>14}"
        f"{'floor':>13}{'factories':>11}{'binding':>12}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name, logical_qubits, gates, rotations in cases:
        target = CHEMICAL_ACCURACY / gates
        distance = required_distance(target, physical)
        if distance is None:
            print(f"{name:<10}{'beyond the model':>60}")
            continue

        t_gates, _ = t_count(rotations, CHEMICAL_ACCURACY)
        single = factory_limited_seconds(t_gates, distance, 1, cycle=cycle)
        floor = consumption_floor_seconds(t_gates, distance, cycle=cycle)
        needed = saturating_factories(t_gates, distance)
        _, binding = runtime(t_gates, distance, needed, cycle=cycle)

        print(
            f"{name:<10}{t_gates:>14,d}{distance:>6}{human(single):>14}"
            f"{human(floor):>13}{needed:>11}{binding:>12}",
            flush=True,
        )

        # what the parallelism costs in qubits, against exp016's data register
        data_qubits = logical_qubits * 2 * distance**2
        rounds = distillation_rounds(physical, CHEMICAL_ACCURACY / max(t_gates, 1))
        one_factory = FACTORY_LOGICAL_QUBITS * 2 * distance**2 * (rounds or 0)

        rows.append({
            "case": name, "logical_qubits": logical_qubits,
            "t_gates": t_gates, "code_distance": distance,
            "one_factory_seconds": single, "floor_seconds": floor,
            "saturating_factories": needed, "binding_limit": binding,
            "data_qubits": data_qubits,
            "qubits_one_factory": data_qubits + one_factory,
            "qubits_saturated": data_qubits + one_factory * needed,
        })

    # ---------------------------------------------------------------- the trade
    print("\n--- the parallel trade on the drug-sized case ---")
    t_gates, _ = t_count(DRUG_ROTATIONS, CHEMICAL_ACCURACY)
    distance = required_distance(CHEMICAL_ACCURACY / DRUG_GATES, physical)
    data_qubits = DRUG_LOGICAL_QUBITS * 2 * distance**2
    rounds = distillation_rounds(physical, CHEMICAL_ACCURACY / max(t_gates, 1))
    one_factory = FACTORY_LOGICAL_QUBITS * 2 * distance**2 * (rounds or 0)
    floor = consumption_floor_seconds(t_gates, distance, cycle=cycle)

    print(f"{'factories':>10}{'wall-clock':>14}{'physical qubits':>18}{'binding':>13}")
    trade = []
    for count in (1, 2, 5, 10, 20, 50):
        seconds, binding = runtime(t_gates, distance, count, cycle=cycle)
        qubits = data_qubits + one_factory * count
        print(f"{count:>10}{human(seconds):>14}{qubits:>18,d}{binding:>13}")
        trade.append({"factories": count, "seconds": seconds,
                      "physical_qubits": qubits, "binding": binding})

    print(f"\nThe floor is {human(floor)} and {saturating_factories(t_gates, distance)} "
          f"factories reach it. Past that, factories cost qubits and buy nothing:")
    print("the T gates sit on the algorithm's critical path, so they are consumed")
    print("one at a time however many are waiting.")

    # ------------------------------------------------------- sensitivity sweeps
    print("\n--- sensitivity: the two model constants, drug-sized, one factory ---")
    print(f"{'factory period':>16}{'wall-clock':>14}   {'cycle time':>12}{'wall-clock':>14}")
    sensitivity = []
    periods = (6.0, 10.0, 20.0)
    cycles = (1e-7, 1e-6, 1e-5)
    for period, candidate_cycle in zip(periods, cycles):
        by_period = factory_limited_seconds(t_gates, distance, 1, period, cycle)
        by_cycle = factory_limited_seconds(
            t_gates, distance, 1, FACTORY_PERIOD_IN_DISTANCES, candidate_cycle
        )
        print(f"{period:>15g}d{human(by_period):>14}   "
              f"{candidate_cycle:>12.0e}{human(by_cycle):>14}")
        sensitivity.append({"factory_period": period, "seconds_by_period": by_period,
                            "cycle_time": candidate_cycle, "seconds_by_cycle": by_cycle})

    print("\nThe conclusion is not sensitive to either within its plausible range:")
    print("across 6d-20d and 1e-7 to 1e-6 s it stays inside hours. A trapped-ion")
    print("cycle time of 1e-5 s is the one assumption that would change the answer.")

    # ------------------------------------------------------------- the repetition
    print("\n--- what actually multiplies this ---")
    print("Everything above is ONE circuit execution. What an algorithm needs:")
    for label, count in (("one execution", 1), ("1e3 evaluations", 1_000),
                         ("1e6 evaluations", 1_000_000)):
        print(f"  {label:<18} {human(floor * count)}")
    print("\nThat multiplier, not the distillation rate, is the open question.")
    print("This model does not guess at it -- pass --repetitions to supply one.")

    if args.repetitions != 1:
        print(f"\nAt {args.repetitions:,} repetitions the floor is "
              f"{human(floor * args.repetitions)}.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "exp017_magic_state_time.json"
    with open(path, "w") as fh:
        json.dump({
            "cycle_time": cycle,
            "factory_period_in_distances": FACTORY_PERIOD_IN_DISTANCES,
            "consumption_in_distances": CONSUMPTION_IN_DISTANCES,
            "rows": rows, "drug_trade": trade, "sensitivity": sensitivity,
            "drug_floor_seconds": floor,
        }, fh, indent=2)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
