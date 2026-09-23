"""Compact interval diagnostics for progressive online experiments."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


class StreamingTSBGradientDiagnostics:
    """O(1) running aggregates for TSB gradient mechanism diagnostics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.reference_count = 0
        self.modification_count = 0
        self.projection_count = 0
        self.grad_cosine_sum = 0.0
        self.conflict_count = 0
        self.modification_ratio_sum = 0.0
        self.projection_removal_ratio_sum = 0.0

    def update(
        self,
        *,
        grad_cosine: float | None,
        conflict: bool | None,
        modification_ratio: float,
        projection_removal_ratio: float | None,
    ) -> None:
        modification_ratio = float(modification_ratio)
        if not np.isfinite(modification_ratio) or modification_ratio < 0.0:
            raise ValueError("TSB modification ratio must be finite and non-negative")
        self.modification_ratio_sum += modification_ratio
        self.modification_count += 1

        if grad_cosine is not None:
            grad_cosine = float(grad_cosine)
            if not np.isfinite(grad_cosine):
                raise ValueError("TSB gradient cosine must be finite")
            self.grad_cosine_sum += max(-1.0, min(1.0, grad_cosine))
            self.conflict_count += int(bool(conflict))
            self.reference_count += 1

        if projection_removal_ratio is not None:
            projection_removal_ratio = float(projection_removal_ratio)
            if (
                not np.isfinite(projection_removal_ratio)
                or projection_removal_ratio < 0.0
            ):
                raise ValueError(
                    "TSB projection removal ratio must be finite and non-negative"
                )
            self.projection_removal_ratio_sum += projection_removal_ratio
            self.projection_count += 1

    def metrics(self) -> dict[str, float | int | None]:
        return {
            "mean_grad_cosine": (
                self.grad_cosine_sum / self.reference_count
                if self.reference_count
                else None
            ),
            "conflict_rate": (
                self.conflict_count / self.reference_count
                if self.reference_count
                else None
            ),
            "mean_tsb_modification_ratio": (
                self.modification_ratio_sum / self.modification_count
                if self.modification_count
                else None
            ),
            "mean_tsb_projection_removal_ratio": (
                self.projection_removal_ratio_sum / self.projection_count
                if self.projection_count
                else None
            ),
            "reference_count": self.reference_count,
            "modification_count": self.modification_count,
            "projection_count": self.projection_count,
        }


class OnlineDiagnosticsRecorder:
    """Accumulate latest metrics and periodically materialize snapshots."""

    def __init__(self, num_experts: int, interval: int) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.num_experts = int(num_experts)
        self.interval = int(interval)
        self.reset()

    def reset(self) -> None:
        self.latest: dict[str, Any] = {}
        self.counters: dict[str, float] = {
            "promotion_count": 0.0,
            "demotion_count": 0.0,
            "drop_count": 0.0,
            "recovery_attempts": 0.0,
            "recovery_successes": 0.0,
            "recovery_to_stable": 0.0,
            "recovery_dropped_after_success": 0.0,
            "recovery_failed": 0.0,
            "recovery_evicted": 0.0,
            "recovery_attempt_exhausted": 0.0,
            "harmful_drift_count": 0.0,
            "beneficial_evolution_count": 0.0,
            "directional_recovery_admission_count": 0.0,
            "recovery_skipped_non_degraded_count": 0.0,
            "performance_recovery_count": 0.0,
            "capability_rebase_count": 0.0,
        }
        self.records: list[dict[str, Any]] = []
        self._last_recorded_step: int | None = None

    def update(self, **metrics: Any) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype.kind not in "biuf":
                raise TypeError(f"diagnostic {key} must be numeric")
            self.latest[key] = array.astype(np.float64, copy=True)

    def increment(self, **counts: float) -> None:
        for key, value in counts.items():
            self.counters[key] = self.counters.get(key, 0.0) + float(value)

    def maybe_record(self, step: int, force: bool = False) -> bool:
        step = int(step)
        if not force:
            if self.interval <= 0 or step % self.interval != 0:
                return False
        if self._last_recorded_step == step:
            return False
        snapshot: dict[str, Any] = {"step": np.asarray(step, dtype=np.int64)}
        snapshot.update(
            {key: np.asarray(value).copy() for key, value in self.latest.items()}
        )
        snapshot.update(
            {
                key: np.asarray(value, dtype=np.float64)
                for key, value in self.counters.items()
            }
        )
        attempts = self.counters.get("recovery_attempts", 0.0)
        successes = self.counters.get("recovery_successes", 0.0)
        snapshot["recovery_success_rate"] = np.asarray(
            successes / attempts if attempts > 0 else 0.0
        )
        self.records.append(snapshot)
        self._last_recorded_step = step
        return True

    @staticmethod
    def _stack(values: list[np.ndarray]) -> np.ndarray:
        shapes = {value.shape for value in values}
        if len(shapes) == 1:
            return np.stack(values, axis=0)
        return np.asarray(values, dtype=object)

    def _arrays(self) -> dict[str, np.ndarray]:
        keys = sorted({key for record in self.records for key in record})
        arrays: dict[str, np.ndarray] = {}
        for key in keys:
            values = [
                np.asarray(record[key])
                for record in self.records
                if key in record
            ]
            arrays[key] = self._stack(values)
        return arrays

    @staticmethod
    def _json_value(value: np.ndarray) -> Any:
        if value.ndim == 0:
            return float(value.item())
        return value.tolist()

    def save(self, directory: str) -> tuple[str, str]:
        """Save interval arrays and a finite-value summary."""

        os.makedirs(directory, exist_ok=True)
        arrays = self._arrays()
        npz_path = os.path.join(directory, "online_diagnostics.npz")
        np.savez_compressed(npz_path, **arrays)

        summary: dict[str, Any] = {
            "num_records": len(self.records),
            "counters": dict(self.counters),
            "metrics": {},
        }
        for key, values in arrays.items():
            if values.dtype == object or key == "step":
                continue
            finite = np.isfinite(values)
            if not finite.any():
                continue
            safe = np.where(finite, values, np.nan)
            summary["metrics"][key] = {
                "mean": self._json_value(np.nanmean(safe, axis=0)),
                "last": self._json_value(np.asarray(values[-1])),
            }
        json_path = os.path.join(
            directory, "online_diagnostics_summary.json"
        )
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return npz_path, json_path
