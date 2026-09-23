from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from utils.credit_assignment import compute_local_credit, compute_sample_credit


def matured_horizon_index(
    record_origin: int, current_origin: int, pred_len: int
) -> Optional[int]:
    """Return the zero-based horizon newly observable at ``current_origin``.

    Rolling origin ``o`` predicts timestamps ``o + 1`` through ``o + H``.
    Consequently, its one-based horizon ``h`` matures exactly at ``o + h``.
    ``None`` means that this record has no target maturing at the current origin.
    """

    horizon = int(current_origin) - int(record_origin)
    if horizon < 1 or horizon > int(pred_len):
        return None
    return horizon - 1


@dataclass
class ProgressiveForecastRecord:
    """Prediction-time snapshot whose targets are revealed one step at a time."""

    origin: int
    x: torch.Tensor
    x_mark: torch.Tensor
    expert_predictions: torch.Tensor
    router_prior: torch.Tensor
    router_weights: torch.Tensor
    mixture_prediction: torch.Tensor
    capability_sketch: Optional[torch.Tensor] = None
    matured_targets: torch.Tensor = field(init=False)
    matured_mask: torch.Tensor = field(init=False)
    expert_squared_error: torch.Tensor = field(init=False)
    local_responsibility: torch.Tensor = field(init=False)
    local_confidence: torch.Tensor = field(init=False)
    sample_responsibility: torch.Tensor = field(init=False)
    sample_confidence: float = field(default=0.0, init=False)
    num_matured: int = field(default=0, init=False)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.origin = int(self.origin)
        self.x = self._cpu_snapshot(self.x)
        self.x_mark = self._cpu_snapshot(self.x_mark)
        self.expert_predictions = self._cpu_snapshot(self.expert_predictions)
        self.router_prior = self._cpu_snapshot(self.router_prior)
        self.router_weights = self._cpu_snapshot(self.router_weights)
        self.mixture_prediction = self._cpu_snapshot(self.mixture_prediction)
        if self.capability_sketch is not None:
            self.capability_sketch = self._cpu_snapshot(self.capability_sketch)

        if self.expert_predictions.ndim != 3:
            raise ValueError(
                "expert_predictions must have shape [H, C, E], got "
                f"{tuple(self.expert_predictions.shape)}"
            )
        horizon, channels, experts = self.expert_predictions.shape
        expected_hce = (horizon, channels, experts)
        if tuple(self.router_prior.shape) != expected_hce:
            raise ValueError(
                f"router_prior must have shape {expected_hce}, got "
                f"{tuple(self.router_prior.shape)}"
            )
        if tuple(self.router_weights.shape) != expected_hce:
            raise ValueError(
                f"router_weights must have shape {expected_hce}, got "
                f"{tuple(self.router_weights.shape)}"
            )
        if tuple(self.mixture_prediction.shape) != (horizon, channels):
            raise ValueError(
                f"mixture_prediction must have shape {(horizon, channels)}, got "
                f"{tuple(self.mixture_prediction.shape)}"
            )
        if self.capability_sketch is not None:
            if self.capability_sketch.ndim != 2:
                raise ValueError("capability_sketch must have shape [E,D]")
            if self.capability_sketch.shape[0] != experts:
                raise ValueError("capability_sketch Expert dimension mismatch")

        target_dtype = self.mixture_prediction.dtype
        self.matured_targets = torch.full(
            (horizon, channels), float("nan"), dtype=target_dtype
        )
        self.matured_mask = torch.zeros(horizon, dtype=torch.bool)
        self.expert_squared_error = torch.zeros(
            experts, dtype=self.expert_predictions.dtype
        )
        self.local_responsibility = torch.full(
            (horizon, channels, experts), float("nan"), dtype=target_dtype
        )
        self.local_confidence = torch.full(
            (horizon, channels), float("nan"), dtype=target_dtype
        )
        self.sample_responsibility = torch.full(
            (experts,), 1.0 / experts, dtype=target_dtype
        )

    @staticmethod
    def _cpu_snapshot(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("forecast record fields must be torch.Tensor values")
        return tensor.detach().to(device="cpu").clone()

    @property
    def horizon(self) -> int:
        return int(self.expert_predictions.shape[0])

    @property
    def channels(self) -> int:
        return int(self.expert_predictions.shape[1])

    @property
    def experts(self) -> int:
        return int(self.expert_predictions.shape[2])

    @property
    def is_complete(self) -> bool:
        return self.num_matured == self.horizon

    @property
    def num_matured_values(self) -> int:
        return self.num_matured * self.channels

    # Compatibility/readability aliases for downstream stages.
    @property
    def expert_outputs(self) -> torch.Tensor:
        return self.expert_predictions

    @property
    def mix_pred(self) -> torch.Tensor:
        return self.mixture_prediction


@dataclass(frozen=True)
class ProgressiveFeedbackEvent:
    """One newly observed target attached to one historical forecast."""

    record: ProgressiveForecastRecord
    horizon_index: int
    target: torch.Tensor
    local_loss: torch.Tensor
    local_responsibility: torch.Tensor
    local_confidence: torch.Tensor


class ProgressiveFeedbackManager:
    """Tracks rolling-origin forecasts without retaining unrevealed targets."""

    def __init__(
        self,
        pred_len: int,
        c_out: int,
        local_credit_temperature: float = 1.0,
        sample_credit_temperature: float = 1.0,
        min_credit_eps: float = 1e-8,
    ) -> None:
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if c_out <= 0:
            raise ValueError("c_out must be positive")
        self.pred_len = int(pred_len)
        self.c_out = int(c_out)
        self.local_credit_temperature = float(local_credit_temperature)
        self.sample_credit_temperature = float(sample_credit_temperature)
        self.min_credit_eps = float(min_credit_eps)
        self._records: Dict[int, ProgressiveForecastRecord] = {}
        self._last_release_origin: int | None = None

    def reset(self) -> None:
        self._records.clear()
        self._last_release_origin = None

    def add_record(self, record: ProgressiveForecastRecord) -> None:
        if record.horizon != self.pred_len or record.channels != self.c_out:
            raise ValueError(
                "record shape does not match manager: "
                f"record={(record.horizon, record.channels)}, "
                f"manager={(self.pred_len, self.c_out)}"
            )
        if record.origin in self._records:
            raise ValueError(f"duplicate forecast origin {record.origin}")
        if self._last_release_origin is not None and record.origin < self._last_release_origin:
            raise ValueError("cannot add a record from an already released origin")
        self._records[record.origin] = record

    def release(
        self, origin: int, observation: torch.Tensor
    ) -> Tuple[List[ProgressiveFeedbackEvent], List[ProgressiveForecastRecord]]:
        """Release exactly the horizon maturing at ``origin`` for each record."""

        origin = int(origin)
        if self._last_release_origin is not None and origin <= self._last_release_origin:
            raise ValueError("release origins must be strictly increasing")
        self._last_release_origin = origin

        target = observation.detach().to(device="cpu").reshape(-1).clone()
        if tuple(target.shape) != (self.c_out,):
            raise ValueError(
                f"observation must contain c_out={self.c_out} values, got "
                f"{tuple(target.shape)}"
            )
        if not torch.isfinite(target).all():
            raise ValueError("newly observed target contains NaN or Inf")

        events: List[ProgressiveFeedbackEvent] = []
        completed: List[ProgressiveForecastRecord] = []
        completed_origins: List[int] = []

        for record_origin, record in list(self._records.items()):
            horizon_index = matured_horizon_index(
                record_origin, origin, self.pred_len
            )
            if horizon_index is None:
                continue
            if bool(record.matured_mask[horizon_index].item()):
                raise RuntimeError(
                    f"origin {record_origin} horizon {horizon_index + 1} "
                    "was released twice"
                )

            record.matured_targets[horizon_index].copy_(
                target.to(dtype=record.matured_targets.dtype)
            )
            record.matured_mask[horizon_index] = True
            error = (
                record.expert_predictions[horizon_index]
                - target.to(dtype=record.expert_predictions.dtype).unsqueeze(-1)
            ).pow(2)
            record.expert_squared_error.add_(error.sum(dim=0))
            record.num_matured += 1
            local_credit = compute_local_credit(
                record.expert_predictions[horizon_index],
                target.to(dtype=record.expert_predictions.dtype),
                temperature=self.local_credit_temperature,
                eps=self.min_credit_eps,
            )
            record.local_responsibility[horizon_index].copy_(
                local_credit.responsibility
            )
            record.local_confidence[horizon_index].copy_(
                local_credit.confidence
            )
            sample_credit = compute_sample_credit(
                accumulated_loss=record.expert_squared_error,
                num_matured_values=record.num_matured_values,
                matured_horizons=record.num_matured,
                total_horizons=record.horizon,
                temperature=self.sample_credit_temperature,
                eps=self.min_credit_eps,
            )
            record.sample_responsibility.copy_(sample_credit.responsibility)
            record.sample_confidence = sample_credit.confidence
            events.append(
                ProgressiveFeedbackEvent(
                    record=record,
                    horizon_index=horizon_index,
                    target=target.clone(),
                    local_loss=local_credit.loss.clone(),
                    local_responsibility=local_credit.responsibility.clone(),
                    local_confidence=local_credit.confidence.clone(),
                )
            )

            if record.is_complete:
                if not bool(record.matured_mask.all().item()):
                    raise RuntimeError("record maturity count and mask disagree")
                completed.append(record)
                completed_origins.append(record_origin)

        for record_origin in completed_origins:
            del self._records[record_origin]
        return events, completed

    @property
    def pending_records(self) -> Tuple[ProgressiveForecastRecord, ...]:
        return tuple(self._records.values())

    def __len__(self) -> int:
        return len(self._records)
