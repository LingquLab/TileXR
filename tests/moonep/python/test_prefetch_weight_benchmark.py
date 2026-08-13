from __future__ import annotations

import pytest

from tools.moonep.prefetch_weight_benchmark import (
    MIN_VALID_EVENT_US,
    percentile,
    ring_plan,
    statistics,
)


def test_ring_plan_pulls_remote_slots() -> None:
    assert ring_plan(4, 2, 2) == [
        [6, 7],
        [0, 1],
        [2, 3],
        [4, 5],
    ]


def test_statistics_use_cross_rank_max_and_retain_invalid_events() -> None:
    result = statistics(
        (
            (0.044, 8.0, 6.0),
            (0.044, 5.0, 7.0),
        )
    )

    assert result["cross_rank_max_us"] == [0.044, 8.0, 7.0]
    assert result["valid_cross_rank_max_us"] == [8.0, 7.0]
    assert result["invalid_event_sample_count"] == 1
    assert result["minimum_valid_event_us_exclusive"] == MIN_VALID_EVENT_US
    assert result["p50_us"] == 7.5
    assert result["p99_us"] == pytest.approx(7.99)


def test_statistics_reject_mismatched_or_entirely_invalid_samples() -> None:
    with pytest.raises(ValueError, match="same number"):
        statistics(((2.0,), (2.0, 3.0)))
    with pytest.raises(ValueError, match="all cross-rank NPU event samples"):
        statistics(((0.044,), (0.044,)))


def test_percentile_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)
