"""The time model, and a guard against reading a tautology as a result.

Result 71 retracts this project's own "months of wall-clock" claim.  Two things
need pinning so the retraction cannot silently un-happen, and one thing needs
pinning so a *model artefact* is never mistaken for a measurement:

* the corrected magnitude (hours, not months) for the drug-sized case
* the data-block floor as a genuine floor -- parallelism cannot go below it
* **the saturating factory count is `period / consumption`, a ratio of two model
  constants, and is therefore the same for every molecule.**  It appears in the
  experiment's output as a column that varies with nothing, which is exactly how
  a tautology disguises itself as a finding.
"""

from __future__ import annotations

import pytest

from experiments.exp016_error_correction_overhead import CHEMICAL_ACCURACY, t_count
from experiments.exp017_magic_state_time import (
    CONSUMPTION_IN_DISTANCES,
    DRUG_ROTATIONS,
    FACTORY_PERIOD_IN_DISTANCES,
    consumption_floor_seconds,
    factory_limited_seconds,
    runtime,
    saturating_factories,
)

DRUG_DISTANCE = 23


def drug_t_gates() -> int:
    gates, _ = t_count(DRUG_ROTATIONS, CHEMICAL_ACCURACY)
    return gates


def test_the_months_claim_was_wrong_by_two_orders_of_magnitude():
    """Result 71's retraction, pinned.

    The README and Result 70 both said "months" for a single factory on a
    drug-sized molecule.  It is hours.  If this ever genuinely becomes months,
    the retraction has to be re-derived rather than quietly inherited.
    """
    seconds = factory_limited_seconds(drug_t_gates(), DRUG_DISTANCE, factories=1)
    hours = seconds / 3600

    assert 1 < hours < 48, f"expected hours, got {hours:.1f} h"
    # and specifically: nowhere near the retracted claim
    assert seconds < 30 * 86400 / 10, "this is supposed to be the retraction"


def test_the_floor_is_a_floor_no_matter_how_many_factories():
    """The load-bearing structural claim: parallelism has a hard limit.

    T gates sit on a sequential algorithm's critical path, so they are consumed
    one at a time however many factories are waiting.  If adding factories ever
    went below this, the qubit/time trade would be unbounded and the conclusion
    would change.
    """
    t_gates = drug_t_gates()
    floor = consumption_floor_seconds(t_gates, DRUG_DISTANCE)

    previous = float("inf")
    for factories in (1, 2, 10, 100, 10_000):
        seconds, _ = runtime(t_gates, DRUG_DISTANCE, factories)
        assert seconds >= floor, f"{factories} factories went below the floor"
        assert seconds <= previous, "more factories must never be slower"
        previous = seconds

    # and the floor is genuinely reached, not merely approached
    saturated, binding = runtime(t_gates, DRUG_DISTANCE, 10_000)
    assert saturated == pytest.approx(floor)
    assert binding == "data block"


def test_the_saturating_count_is_a_model_ratio_not_a_measurement():
    """The guard against reading exp017's most eye-catching column as a result.

    `saturating_factories` is `ceil(period / consumption)` and nothing else.  It
    does not depend on the molecule, the T count, or the code distance -- which
    is precisely why it prints as an identical 10 on every row.  Anyone tempted
    to quote "ten factories" as a finding about chemistry should read this test.
    """
    expected = int(-(-FACTORY_PERIOD_IN_DISTANCES // CONSUMPTION_IN_DISTANCES))

    # same answer across four orders of magnitude in T count and 5x in distance
    for t_gates in (148, 7_997, 150_790, drug_t_gates()):
        for distance in (5, 11, 23):
            assert saturating_factories(t_gates, distance) == expected

    # and it moves only when the model constants move
    assert saturating_factories(drug_t_gates(), DRUG_DISTANCE, period=4.0) == 4
    assert saturating_factories(drug_t_gates(), DRUG_DISTANCE, period=1.0) == 1


def test_binding_limit_is_reported_correctly():
    """Which limit binds is the whole point; mislabelling it inverts the advice."""
    t_gates = drug_t_gates()

    _, few = runtime(t_gates, DRUG_DISTANCE, factories=1)
    _, many = runtime(t_gates, DRUG_DISTANCE, factories=1_000)
    assert few == "factory", "one factory must be production-limited"
    assert many == "data block", "many factories must be consumption-limited"


def test_runtime_scales_linearly_in_t_count():
    """Doubling the work doubles the time -- no hidden nonlinearity in the model."""
    base = factory_limited_seconds(1_000_000, DRUG_DISTANCE)
    assert factory_limited_seconds(2_000_000, DRUG_DISTANCE) == pytest.approx(2 * base)
    assert consumption_floor_seconds(2_000_000, DRUG_DISTANCE) == pytest.approx(
        2 * consumption_floor_seconds(1_000_000, DRUG_DISTANCE)
    )


def test_zero_factories_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        factory_limited_seconds(1_000, DRUG_DISTANCE, factories=0)
