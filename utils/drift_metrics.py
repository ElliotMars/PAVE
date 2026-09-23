"""Metrics and lightweight memory timelines for known synthetic drift events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_TRANSITION_KEYS = (
    "stable_to_recovery",
    "recovery_to_stable",
    "recovery_dropped_after_success",
    "recovery_attempt_exhausted",
)

_DIAGNOSTIC_KEYS = (
    "retained",
    "low_alignment",
    "harmful_drift",
    "beneficial_evolution",
    "capability_rebase",
    "stable_to_recovery",
    "recovery_admission",
    "recovery_attempt",
    "performance_recovery",
    "recovery_to_stable",
    "recovery_failed",
    "recovery_attempt_exhausted",
    "recovery_dropped_after_success",
    "recovery_skipped_non_degraded",
    "matured_records",
)


def _event_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    converter = getattr(event, "to_dict", None)
    if converter is None:
        raise TypeError("drift events must be mappings or expose to_dict()")
    return dict(converter())


def _mean_or_none(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def recovery_time(
    mse_timeline: Sequence[float],
    *,
    drift_origin: int,
    pre_drift_error: float,
    recovery_window: int,
    recovery_tolerance: float,
    search_end: int | None = None,
) -> int | None:
    """Return the first offset whose W-origin mean reaches baseline tolerance."""

    values = np.asarray(mse_timeline, dtype=np.float64).reshape(-1)
    if recovery_window <= 0:
        raise ValueError("recovery_window must be positive")
    if recovery_tolerance < 0.0 or not np.isfinite(recovery_tolerance):
        raise ValueError("recovery_tolerance must be finite and non-negative")
    if not np.isfinite(pre_drift_error) or pre_drift_error < 0.0:
        raise ValueError("pre_drift_error must be finite and non-negative")
    start = max(0, int(drift_origin))
    end = len(values) if search_end is None else min(len(values), int(search_end))
    threshold = float(pre_drift_error) * (1.0 + float(recovery_tolerance))
    last_start = end - int(recovery_window)
    for candidate in range(start, last_start + 1):
        window = values[candidate : candidate + recovery_window]
        if np.isfinite(window).all() and float(window.mean()) <= threshold:
            return candidate - start
    return None


def rolling_mean(
    values: Sequence[float], window: int
) -> np.ndarray:
    # Trailing rolling mean aligned to the input timeline.
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 0:
        raise ValueError("rolling window must be positive")
    result = np.full(array.shape, np.nan, dtype=np.float64)
    if array.size < window:
        return result
    cumulative = np.concatenate(([0.0], np.cumsum(array)))
    result[window - 1 :] = (
        cumulative[window:] - cumulative[:-window]
    ) / float(window)
    return result


def sustained_recovery_time(
    mse_timeline: Sequence[float],
    *,
    drift_origin: int,
    reference_error: float,
    rolling_window: int,
    recovery_tolerance: float,
    hold_steps: int,
    search_end: int | None = None,
) -> int | None:
    # First A2 offset whose rolling MSE stays recovered for hold_steps.
    values = np.asarray(mse_timeline, dtype=np.float64).reshape(-1)
    if hold_steps <= 0:
        raise ValueError("hold_steps must be positive")
    if recovery_tolerance < 0.0 or not np.isfinite(recovery_tolerance):
        raise ValueError("recovery_tolerance must be finite and non-negative")
    if not np.isfinite(reference_error) or reference_error < 0.0:
        raise ValueError("reference_error must be finite and non-negative")
    start = max(0, int(drift_origin))
    end = len(values) if search_end is None else min(len(values), int(search_end))
    local = values[start:end]
    rolled = rolling_mean(local, rolling_window)
    threshold = float(reference_error) * (1.0 + float(recovery_tolerance))
    recovered = np.isfinite(rolled) & (rolled <= threshold)
    last_start = recovered.size - int(hold_steps)
    for candidate in range(0, last_start + 1):
        if bool(recovered[candidate : candidate + hold_steps].all()):
            return candidate
    return None


def compute_drift_metrics(
    mse_timeline: Sequence[float],
    events: Sequence[Any],
    *,
    pre_window: int = 16,
    early_window: int = 8,
    recovery_window: int = 4,
    recovery_tolerance: float = 0.2,
    recovery_hold_steps: int = 1,
    recurring_intervals: Mapping[str, Sequence[int]] | None = None,
    seq_len: int = 0,
) -> dict[str, Any]:
    """Compute interpretable event and recurring-mode recovery statistics."""

    values = np.asarray(mse_timeline, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("mse_timeline must contain finite non-negative values")
    if pre_window <= 0 or early_window <= 0:
        raise ValueError("pre_window and early_window must be positive")
    if recovery_hold_steps <= 0:
        raise ValueError("recovery_hold_steps must be positive")
    event_dicts = [_event_dict(event) for event in events]
    event_results: list[dict[str, Any]] = []
    for index, event in enumerate(event_dicts):
        start = min(len(values), max(0, int(event["origin"])))
        next_start = (
            min(len(values), max(start, int(event_dicts[index + 1]["origin"])))
            if index + 1 < len(event_dicts)
            else len(values)
        )
        baseline = _mean_or_none(values[max(0, start - pre_window) : start])
        early = _mean_or_none(values[start : min(next_start, start + early_window)])
        segment = values[start:next_start]
        peak = float(segment.max()) if segment.size else None
        if baseline is None:
            excess = None
            recovered = None
        else:
            excess = float(np.maximum(segment - baseline, 0.0).sum())
            recovered = recovery_time(
                values,
                drift_origin=start,
                pre_drift_error=baseline,
                recovery_window=recovery_window,
                recovery_tolerance=recovery_tolerance,
                search_end=next_start,
            )
        event_results.append(
            {
                "name": event.get("name", f"event_{index}"),
                "kind": event.get("kind"),
                "origin": start,
                "pre_drift_error": baseline,
                "early_post_drift_error": early,
                "peak_error": peak,
                "recovery_time": recovered,
                "cumulative_excess_error": excess,
            }
        )

    result: dict[str, Any] = {
        "definition": {
            "pre_window": int(pre_window),
            "early_window": int(early_window),
            "recovery_window": int(recovery_window),
            "recovery_tolerance": float(recovery_tolerance),
            "recovery_hold_steps": int(recovery_hold_steps),
            "no_recovery_sentinel": None,
        },
        "events": event_results,
    }
    required = {"first_A", "B", "recurring_A"}
    if recurring_intervals is not None and required.issubset(recurring_intervals):
        origin_intervals = {
            name: (
                max(0, int(bounds[0]) - int(seq_len)),
                max(0, int(bounds[1]) - int(seq_len)),
            )
            for name, bounds in recurring_intervals.items()
        }
        first_start, first_end = origin_intervals["first_A"]
        b_start, b_end = origin_intervals["B"]
        recurring_start, recurring_end = origin_intervals["recurring_A"]
        first_start = min(first_start, len(values))
        first_end = min(first_end, len(values))
        b_start = min(b_start, len(values))
        b_end = min(b_end, len(values))
        recurring_start = min(recurring_start, len(values))
        recurring_end = min(recurring_end, len(values))

        first_values = values[first_start:first_end]
        b_values = values[b_start:b_end]
        recurring_values = values[recurring_start:recurring_end]
        first_error = _mean_or_none(first_values)
        reference_error = _mean_or_none(first_values[-pre_window:])
        b_early = _mean_or_none(b_values[:early_window])
        b_late = _mean_or_none(b_values[-early_window:])
        recurring_early = _mean_or_none(recurring_values[:early_window])
        recurring_late = _mean_or_none(recurring_values[-early_window:])
        reacquisition = (
            sustained_recovery_time(
                values,
                drift_origin=recurring_start,
                reference_error=reference_error,
                rolling_window=recovery_window,
                recovery_tolerance=recovery_tolerance,
                hold_steps=recovery_hold_steps,
                search_end=recurring_end,
            )
            if reference_error is not None
            else None
        )
        recovered_error = None
        if reacquisition is not None:
            recovered_start = recurring_start + reacquisition
            recovered_error = _mean_or_none(
                values[
                    recovered_start
                    : min(recurring_end, recovered_start + recovery_window)
                ]
            )
        retention_ratio = (
            recurring_early / first_error
            if first_error is not None
            and first_error > 0.0
            and recurring_early is not None
            else None
        )
        normalized_degradation = (
            recurring_early / (reference_error + 1e-12)
            if reference_error is not None and recurring_early is not None
            else None
        )
        cumulative_excess = (
            float(
                np.maximum(recurring_values - reference_error, 0.0).sum()
            )
            if reference_error is not None
            else None
        )
        result["recurring_mode"] = {
            "origin_intervals": {
                name: [int(bounds[0]), int(bounds[1])]
                for name, bounds in origin_intervals.items()
            },
            "first_A_error": first_error,
            "recurring_A_early_error": recurring_early,
            "recurring_A_recovered_error": recovered_error,
            "A1_reference_mse": reference_error,
            "B_early_mse": b_early,
            "B_late_mse": b_late,
            "A2_early_mse": recurring_early,
            "A2_late_mse": recurring_late,
            "normalized_recurring_degradation": normalized_degradation,
            "reacquisition_time": reacquisition,
            "cumulative_excess_error": cumulative_excess,
            "old_mode_retention_error_ratio": retention_ratio,
            "old_mode_retention_definition": (
                "recurring_A_early_error / first_A_error; 1 is perfect retention"
            ),
        }
    return result


class DriftMetricTracker:
    """Resettable holder for an online MSE timeline."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.mse: list[float] = []

    def update(self, mse: float) -> None:
        value = float(mse)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("MSE must be finite and non-negative")
        self.mse.append(value)


class MemoryTimelineRecorder:
    """Record small occupancy arrays and aggregate lifecycle transitions."""

    def __init__(self, num_experts: int) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.num_experts = int(num_experts)
        self.reset()

    def reset(self) -> None:
        self.steps: list[int] = []
        self.stable_occupancy: list[list[int]] = []
        self.recovery_occupancy: list[list[int]] = []
        self.mean_recovery_attempts: list[float] = []
        self.stable_regime_occupancy: list[dict[int, int]] = []
        self.recovery_regime_occupancy: list[dict[int, int]] = []
        self.transitions = {key: 0 for key in _TRANSITION_KEYS}
        self.diagnostic_counts = {key: 0 for key in _DIAGNOSTIC_KEYS}
        self.event_counts = {key: [] for key in _DIAGNOSTIC_KEYS}

    def record(
        self,
        step: int,
        memory_manager: Any,
        *,
        transition_stats: Mapping[str, int] | None = None,
        sample_regimes: Mapping[int, int] | None = None,
    ) -> None:
        stable_sizes, recovery_sizes = memory_manager.buffer_sizes()
        if len(stable_sizes) != self.num_experts:
            raise ValueError("memory manager Expert count mismatch")
        self.steps.append(int(step))
        self.stable_occupancy.append([int(value) for value in stable_sizes])
        self.recovery_occupancy.append([int(value) for value in recovery_sizes])
        recovery_items = [
            item
            for buffer in memory_manager.recovery_buffers
            for item in buffer.items
        ]
        self.mean_recovery_attempts.append(
            float(np.mean([item.recovery_attempts for item in recovery_items]))
            if recovery_items
            else 0.0
        )
        sample_regimes = sample_regimes or {}
        stable_counts: dict[int, int] = {}
        recovery_counts: dict[int, int] = {}
        for buffers, destination in (
            (memory_manager.stable_buffers, stable_counts),
            (memory_manager.recovery_buffers, recovery_counts),
        ):
            for buffer in buffers:
                for item in buffer.items:
                    regime = int(sample_regimes.get(item.sample_id, -1))
                    destination[regime] = destination.get(regime, 0) + 1
        self.stable_regime_occupancy.append(stable_counts)
        self.recovery_regime_occupancy.append(recovery_counts)
        stats = transition_stats or {}
        for key in _TRANSITION_KEYS:
            self.transitions[key] += int(stats.get(key, 0))
        for key in _DIAGNOSTIC_KEYS:
            value = int(stats.get(key, 0))
            self.diagnostic_counts[key] += value
            self.event_counts[key].append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        regime_labels = sorted(
            {
                regime
                for timeline in (
                    self.stable_regime_occupancy,
                    self.recovery_regime_occupancy,
                )
                for counts in timeline
                for regime in counts
            }
        )

        def regime_array(records: list[dict[int, int]]) -> np.ndarray:
            return np.asarray(
                [
                    [counts.get(regime, 0) for regime in regime_labels]
                    for counts in records
                ],
                dtype=np.int64,
            ).reshape(len(records), len(regime_labels))

        return {
            "step": np.asarray(self.steps, dtype=np.int64),
            "stable_occupancy": np.asarray(
                self.stable_occupancy, dtype=np.int64
            ).reshape(-1, self.num_experts),
            "recovery_occupancy": np.asarray(
                self.recovery_occupancy, dtype=np.int64
            ).reshape(-1, self.num_experts),
            "mean_recovery_attempts": np.asarray(
                self.mean_recovery_attempts, dtype=np.float64
            ),
            "regime_labels": np.asarray(regime_labels, dtype=np.int64),
            "stable_occupancy_by_regime": regime_array(
                self.stable_regime_occupancy
            ),
            "recovery_occupancy_by_regime": regime_array(
                self.recovery_regime_occupancy
            ),
            **{
                f"{key}_events": np.asarray(values, dtype=np.int64)
                for key, values in self.event_counts.items()
            },
        }

    def summary(self) -> dict[str, Any]:
        arrays = self.arrays()
        return {
            "num_steps": len(self.steps),
            "transitions": dict(self.transitions),
            "diagnostic_counts": {
                f"{key}_count": int(value)
                for key, value in self.diagnostic_counts.items()
            },
            "mean_recovery_attempts_definition": (
                "mean attempts among current Recovery items, averaged over origins"
            ),
            "mean_recovery_attempts": (
                float(arrays["mean_recovery_attempts"].mean())
                if self.steps
                else 0.0
            ),
            "final_stable_occupancy": (
                arrays["stable_occupancy"][-1].tolist() if self.steps else []
            ),
            "final_recovery_occupancy": (
                arrays["recovery_occupancy"][-1].tolist() if self.steps else []
            ),
            "stable_regime_occupancy_over_time": self.stable_regime_occupancy,
            "recovery_regime_occupancy_over_time": self.recovery_regime_occupancy,
        }
