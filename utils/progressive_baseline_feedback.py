"""Causal progressive-feedback protocol for baseline fairness controls.

This module deliberately shares only PACE's target-maturity schedule. It does
not reuse any PACE routing, credit, correction, or memory mechanism.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from utils.progressive_feedback import matured_horizon_index


@dataclass
class ProgressiveBaselineRecord:
    """Prediction-time state containing no target from the forecast window."""

    origin: int
    x: torch.Tensor
    method: str
    x_mark: Optional[torch.Tensor] = None
    representation: Optional[torch.Tensor] = None
    expert_predictions: Optional[torch.Tensor] = None
    blend: Optional[float] = None
    released_horizons: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.origin = int(self.origin)
        self.method = str(self.method).lower()
        self.x = self._snapshot(self.x)
        self.x_mark = self._optional_snapshot(self.x_mark)
        self.representation = self._optional_snapshot(self.representation)
        self.expert_predictions = self._optional_snapshot(
            self.expert_predictions
        )
        if self.blend is not None:
            self.blend = float(self.blend)

        if self.method == "dyname":
            if self.representation is None or self.expert_predictions is None:
                raise ValueError(
                    "DynaME progressive records require prediction-time caches"
                )
            if self.expert_predictions.ndim != 4:
                raise ValueError(
                    "cached DynaME experts must have shape [B,E,H,C]"
                )
            horizon = int(self.expert_predictions.shape[2])
        else:
            horizon = 0
        self.released_horizons = torch.zeros(horizon, dtype=torch.bool)

    @staticmethod
    def _snapshot(value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("progressive baseline tensor fields must be tensors")
        return value.detach().to(device="cpu").clone()

    @classmethod
    def _optional_snapshot(
        cls, value: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        return None if value is None else cls._snapshot(value)

    def initialize_horizon(self, pred_len: int) -> None:
        pred_len = int(pred_len)
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if self.released_horizons.numel() not in (0, pred_len):
            raise ValueError("record prediction cache has the wrong horizon")
        if self.released_horizons.numel() == 0:
            self.released_horizons = torch.zeros(pred_len, dtype=torch.bool)

    @property
    def is_complete(self) -> bool:
        return bool(self.released_horizons.all().item())


@dataclass(frozen=True)
class ProgressiveBaselineEvent:
    record: ProgressiveBaselineRecord
    horizon_index: int
    target: torch.Tensor


class ProgressiveBaselineFeedbackManager:
    """Release observable partial labels using PACE's exact maturity rule."""

    def __init__(self, pred_len: int, c_out: int) -> None:
        if pred_len <= 0 or c_out <= 0:
            raise ValueError("pred_len and c_out must be positive")
        self.pred_len = int(pred_len)
        self.c_out = int(c_out)
        self._records: dict[int, ProgressiveBaselineRecord] = {}
        self._last_release_origin: Optional[int] = None

    def add_record(self, record: ProgressiveBaselineRecord) -> None:
        record.initialize_horizon(self.pred_len)
        if record.origin in self._records:
            raise ValueError(f"duplicate forecast origin {record.origin}")
        if (
            self._last_release_origin is not None
            and record.origin < self._last_release_origin
        ):
            raise ValueError("cannot add a record from an already released origin")
        self._records[record.origin] = record

    def release(
        self, current_origin: int, observation: torch.Tensor
    ) -> list[ProgressiveBaselineEvent]:
        current_origin = int(current_origin)
        if (
            self._last_release_origin is not None
            and current_origin <= self._last_release_origin
        ):
            raise ValueError("release origins must be strictly increasing")
        self._last_release_origin = current_origin

        target = observation.detach().to(device="cpu").reshape(-1).clone()
        if tuple(target.shape) != (self.c_out,):
            raise ValueError(
                f"observation must contain c_out={self.c_out} values, got "
                f"{tuple(target.shape)}"
            )
        if not torch.isfinite(target).all():
            raise ValueError("newly observed target contains NaN or Inf")

        events: list[ProgressiveBaselineEvent] = []
        completed: list[int] = []
        for record_origin, record in list(self._records.items()):
            horizon_index = matured_horizon_index(
                record_origin, current_origin, self.pred_len
            )
            if horizon_index is None:
                continue
            if bool(record.released_horizons[horizon_index].item()):
                raise RuntimeError(
                    f"origin {record_origin} horizon {horizon_index + 1} "
                    "was released twice"
                )
            record.released_horizons[horizon_index] = True
            events.append(
                ProgressiveBaselineEvent(
                    record=record,
                    horizon_index=horizon_index,
                    target=target.clone(),
                )
            )
            if record.is_complete:
                completed.append(record_origin)

        for record_origin in completed:
            del self._records[record_origin]
        return events

    @property
    def pending_records(self) -> tuple[ProgressiveBaselineRecord, ...]:
        return tuple(self._records.values())

    def __len__(self) -> int:
        return len(self._records)


class ProgressiveBaselineDiagnostics:
    """Small audit trail for one progressive baseline stream."""

    protocol = "progressive_baseline_control"

    def __init__(self, pred_len: int) -> None:
        self.pred_len = int(pred_len)
        self.total_origins = 0
        self.prediction_count = 0
        self.released_event_count = 0
        self.released_event_count_by_horizon = [0] * self.pred_len
        self.origins_with_feedback = 0
        self.optimizer_step_count = 0
        self.max_events_per_origin = 0
        self.max_optimizer_steps_per_origin = 0
        self.pending_records_at_end = 0
        self.first_feedback_origin: Optional[int] = None
        self.last_feedback_origin: Optional[int] = None
        self.future_target_leakage_detected = False
        self.event_release_timeline: list[dict[str, object]] = []
        self._steps_this_origin = 0

    def begin_origin(
        self, origin: int, events: list[ProgressiveBaselineEvent]
    ) -> None:
        self._steps_this_origin = 0
        count = len(events)
        self.total_origins += 1
        self.released_event_count += count
        self.max_events_per_origin = max(self.max_events_per_origin, count)
        if count:
            self.origins_with_feedback += 1
            if self.first_feedback_origin is None:
                self.first_feedback_origin = int(origin)
            self.last_feedback_origin = int(origin)
        released = []
        for event in events:
            self.released_event_count_by_horizon[event.horizon_index] += 1
            released.append(
                {
                    "forecast_origin": int(event.record.origin),
                    "horizon": int(event.horizon_index + 1),
                }
            )
        self.event_release_timeline.append(
            {
                "origin": int(origin),
                "released_event_count": count,
                "events": released,
            }
        )

    def optimizer_step(self) -> None:
        self._steps_this_origin += 1
        if self._steps_this_origin > 1:
            raise RuntimeError("more than one optimizer step at one origin")
        self.optimizer_step_count += 1
        self.max_optimizer_steps_per_origin = max(
            self.max_optimizer_steps_per_origin, self._steps_this_origin
        )

    def prediction(self) -> None:
        self.prediction_count += 1

    def finish(self, pending_records: int) -> None:
        self.pending_records_at_end = int(pending_records)

    def as_dict(self) -> dict[str, object]:
        if self.max_optimizer_steps_per_origin > 1:
            raise RuntimeError("progressive baseline optimizer-step invariant failed")
        return {
            "protocol": self.protocol,
            "total_origins": self.total_origins,
            "prediction_count": self.prediction_count,
            "released_event_count": self.released_event_count,
            "released_event_count_by_horizon": (
                self.released_event_count_by_horizon
            ),
            "origins_with_feedback": self.origins_with_feedback,
            "optimizer_step_count": self.optimizer_step_count,
            "max_events_per_origin": self.max_events_per_origin,
            "max_optimizer_steps_per_origin": (
                self.max_optimizer_steps_per_origin
            ),
            "pending_records_at_end": self.pending_records_at_end,
            "first_feedback_origin": self.first_feedback_origin,
            "last_feedback_origin": self.last_feedback_origin,
            "future_target_leakage_detected": (
                self.future_target_leakage_detected
            ),
            "event_release_timeline": self.event_release_timeline,
        }

    def save(self, result_directory: str) -> str:
        path = Path(result_directory) / "protocol_diagnostics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.as_dict(),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        return str(path)


def progressive_partial_mse(
    predictions: torch.Tensor,
    horizon_indices: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean event loss with only genuinely matured horizon/channel values."""

    if predictions.ndim != 3:
        raise ValueError("predictions must have shape [events,H,C]")
    if horizon_indices.ndim != 1 or targets.ndim != 2:
        raise ValueError("indices/targets must have shapes [events] and [events,C]")
    if predictions.shape[0] != horizon_indices.shape[0]:
        raise ValueError("event batch size mismatch")
    batch_indices = torch.arange(predictions.shape[0], device=predictions.device)
    selected = predictions[batch_indices, horizon_indices]
    if tuple(selected.shape) != tuple(targets.shape):
        raise ValueError("selected prediction and target shapes differ")
    return (selected - targets).pow(2).mean(dim=1).mean()
