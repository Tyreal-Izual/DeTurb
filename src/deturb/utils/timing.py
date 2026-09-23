"""Timing aggregation helpers for production-like DeTurb training."""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence


TIMING_KEYS = ("end_to_end_seconds", "model_step_seconds", "data_wait_seconds")


def timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("timing values cannot be empty")
    finite_values = [float(value) for value in values]
    if not all(math.isfinite(value) and value >= 0 for value in finite_values):
        raise ValueError("timing values must be finite and non-negative")
    ordered = sorted(finite_values)
    p90_index = max(math.ceil(0.9 * len(ordered)) - 1, 0)
    return {
        "count": len(ordered),
        "mean_seconds": statistics.fmean(ordered),
        "median_seconds": statistics.median(ordered),
        "p90_seconds": ordered[p90_index],
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
    }


def aggregate_rank_step_timings(
    rank_records: Sequence[dict[str, Any]],
    *,
    warmup_iterations: int,
    measured_iterations: int,
    global_batch_size: int,
    force_edge_loss: bool,
) -> dict[str, Any]:
    if len(rank_records) == 0:
        raise ValueError("rank timing records cannot be empty")
    if measured_iterations <= 0 or global_batch_size <= 0:
        raise ValueError("measured iterations and global batch size must be positive")
    per_rank_steps = [record["steps"] for record in rank_records]
    if any(len(steps) != measured_iterations for steps in per_rank_steps):
        raise ValueError("each rank must provide every measured iteration")

    aggregate: dict[str, Any] = {}
    for key in TIMING_KEYS:
        slowest_rank_values = [
            max(float(steps[index][key]) for steps in per_rank_steps)
            for index in range(measured_iterations)
        ]
        aggregate[key.removesuffix("_seconds")] = {
            **timing_summary(slowest_rank_values),
            "raw_seconds": slowest_rank_values,
        }

    end_to_end = aggregate["end_to_end"]
    aggregate["throughput"] = {
        "global_samples_per_mean_second": (
            global_batch_size / end_to_end["mean_seconds"]
        ),
        "global_samples_per_median_second": (
            global_batch_size / end_to_end["median_seconds"]
        ),
    }
    aggregate["per_rank"] = [
        {
            "rank": int(record["rank"]),
            **{
                key.removesuffix("_seconds"): timing_summary(
                    [float(step[key]) for step in record["steps"]]
                )
                for key in TIMING_KEYS
            },
        }
        for record in rank_records
    ]
    aggregate.update(
        {
            "world_size": len(rank_records),
            "global_batch_size": global_batch_size,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "rank_aggregation": "per-iteration maximum across ranks",
            "force_edge_loss": force_edge_loss,
        }
    )
    return aggregate


def aggregate_rank_events(
    rank_records: Sequence[dict[str, Any]],
    event_key: str,
) -> list[dict[str, Any]]:
    if not rank_records:
        raise ValueError("rank event records cannot be empty")
    per_rank_events = [record[event_key] for record in rank_records]
    event_count = len(per_rank_events[0])
    if any(len(events) != event_count for events in per_rank_events):
        raise ValueError(f"rank {event_key} counts differ")

    aggregated = []
    for index in range(event_count):
        identity = {
            key: value
            for key, value in per_rank_events[0][index].items()
            if key != "seconds"
        }
        rank_seconds = []
        for events in per_rank_events:
            current_identity = {
                key: value for key, value in events[index].items() if key != "seconds"
            }
            if current_identity != identity:
                raise ValueError(f"rank {event_key} identities differ")
            rank_seconds.append(float(events[index]["seconds"]))
        aggregated.append(
            {
                **identity,
                "seconds": max(rank_seconds),
                "rank_seconds": rank_seconds,
                "rank_aggregation": "maximum",
            }
        )
    return aggregated
