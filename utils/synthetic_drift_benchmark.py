"""Small progressive benchmark for synthetic drift and DRSE mechanisms."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from data_provider.synthetic_drift import (
    SyntheticDriftDataset,
    parse_drift_channels,
)
from utils.credit_assignment import partial_router_objective
from utils.drift_metrics import (
    DriftMetricTracker,
    MemoryTimelineRecorder,
    compute_drift_metrics,
    rolling_mean,
)
from utils.expert_memory import (
    ExpertMemoryManager,
    VersionedMemoryItem,
    classify_capability_evolution,
)
from utils.online_routing import OnlineRoutingCorrection
from utils.progressive_feedback import (
    ProgressiveFeedbackManager,
    ProgressiveForecastRecord,
)


_STRATEGIES = {"plain", "tsb", "subspace", "hybrid"}


@dataclass
class SyntheticBenchmarkConfig:
    seq_len: int = 12
    pred_len: int = 3
    channels: int = 3
    total_length: int = 120
    seed: int = 0
    noise_std: float = 0.05
    transition_window: int | None = None
    shock_duration: int | None = None
    drift_type: str = "recurring"
    regime_separation: float = 1.0
    a1_length: int | None = None
    b_length: int | None = None
    a2_length: int | None = None
    drift_channels: tuple[int, ...] | None = None
    num_experts: int = 3
    strategy: str = "hybrid"
    expert_lr: float = 0.02
    router_lr: float = 0.05
    recovery_window: int = 4
    recovery_tolerance: float = 0.2
    recovery_hold_steps: int = 2
    rolling_window: int = 8
    early_window: int = 8
    pre_window: int = 16
    stable_capacity: int = 16
    recovery_capacity: int = 16
    max_recovery_attempts: int = 4
    memory_refresh_interval: int = 2
    disable_recovery: bool = False
    disable_directional_recovery: bool = False
    disable_version_awareness: bool = False
    disable_z_correction: bool = False

    def validate(self) -> None:
        self.strategy = str(self.strategy).lower()
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(_STRATEGIES)}")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.expert_lr <= 0.0 or self.router_lr <= 0.0:
            raise ValueError("online learning rates must be positive")
        if self.memory_refresh_interval <= 0:
            raise ValueError("memory_refresh_interval must be positive")
        if self.recovery_hold_steps <= 0 or self.rolling_window <= 0:
            raise ValueError(
                "recovery_hold_steps and rolling_window must be positive"
            )
        lengths = (self.a1_length, self.b_length, self.a2_length)
        if any(value is not None for value in lengths):
            if not all(value is not None for value in lengths):
                raise ValueError(
                    "a1_length, b_length, and a2_length must be set together"
                )
            if sum(int(value) for value in lengths) != self.total_length:
                raise ValueError(
                    "A1/B/A2 lengths must sum to total_length"
                )


@dataclass
class SyntheticBenchmarkResult:
    predictions: np.ndarray
    targets: np.ndarray
    mse_timeline: np.ndarray
    horizon_mse_timeline: np.ndarray
    series: np.ndarray
    regime_id: np.ndarray
    transition_alpha: np.ndarray
    drift_metrics: dict[str, Any]
    memory_diagnostics: dict[str, Any]
    event_metadata: dict[str, Any]
    memory_arrays: dict[str, np.ndarray]
    regime_diagnostics: dict[str, Any]
    origin: np.ndarray
    origin_regime_id: np.ndarray
    origin_phase: np.ndarray
    rolling_mse: np.ndarray
    completed_records: int
    protocol_trace: tuple[tuple[int, str], ...]


class SyntheticProgressiveBenchmark:
    """A tiny adaptive MoE that follows the production feedback ordering."""

    def __init__(self, config: SyntheticBenchmarkConfig) -> None:
        config.validate()
        self.config = config
        torch.manual_seed(config.seed)
        self.dataset = SyntheticDriftDataset(
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            channels=config.channels,
            total_length=config.total_length,
            seed=config.seed,
            noise_std=config.noise_std,
            transition_window=config.transition_window,
            shock_duration=config.shock_duration,
            drift_type=config.drift_type,
            drift_channels=config.drift_channels,
            regime_separation=config.regime_separation,
            a1_length=config.a1_length,
            b_length=config.b_length,
            a2_length=config.a2_length,
        )
        experts, channels = config.num_experts, config.channels
        identity = torch.eye(channels).unsqueeze(0).repeat(experts, 1, 1)
        scales = torch.linspace(0.35, 0.9, experts).reshape(experts, 1, 1)
        signs = torch.ones(experts, 1, 1)
        if experts > 1:
            signs[1::2] = -1.0
        self.expert_weight = torch.nn.Parameter(identity * scales * signs)
        self.expert_bias = torch.nn.Parameter(
            torch.linspace(-0.1, 0.1, experts).reshape(experts, 1).expand(-1, channels).clone()
        )
        self.router_logits = torch.nn.Parameter(
            torch.zeros(config.pred_len, channels, experts)
        )
        self.expert_optimizer = torch.optim.SGD(
            [self.expert_weight, self.expert_bias], lr=config.expert_lr
        )
        self.router_optimizer = torch.optim.SGD(
            [self.router_logits], lr=config.router_lr
        )
        self.feedback = ProgressiveFeedbackManager(
            pred_len=config.pred_len,
            c_out=channels,
            local_credit_temperature=0.7,
            sample_credit_temperature=0.7,
        )
        self.correction = OnlineRoutingCorrection(
            pred_len=config.pred_len,
            c_out=channels,
            num_experts=experts,
            device=torch.device("cpu"),
            correction_lr=0.15,
            correction_decay=0.01,
            correction_grad_clip=10.0,
            correction_logit_clip=5.0,
        )
        self.memory = ExpertMemoryManager(
            num_experts=experts,
            stable_capacity=config.stable_capacity,
            recovery_capacity=(0 if config.disable_recovery else config.recovery_capacity),
            responsibility_threshold=0.0,
            alignment_threshold=0.985,
            duplicate_threshold=1.1,
            failure_penalty=0.5,
            max_recovery_attempts=config.max_recovery_attempts,
            storage_dtype="fp32",
            promote_alignment_threshold=0.99,
            promote_loss_threshold=1.5,
            version_awareness_enabled=not config.disable_version_awareness,
            directional_recovery_enabled=(
                not config.disable_directional_recovery
                and not config.disable_version_awareness
            ),
            recovery_enabled=not config.disable_recovery,
            capability_rebase_enabled=(
                not config.disable_directional_recovery
                and not config.disable_version_awareness
            ),
            recovery_degradation_margin=0.0,
        )
        self.metric_tracker = DriftMetricTracker()
        self.memory_recorder = MemoryTimelineRecorder(experts)
        self._initial_expert_weight = self.expert_weight.detach().clone()
        self._initial_expert_bias = self.expert_bias.detach().clone()
        self._initial_router_logits = self.router_logits.detach().clone()
        self.previous_gradients: tuple[torch.Tensor, torch.Tensor] | None = None
        self.sample_regimes: dict[int, int] = {}
        self.protocol_trace: list[tuple[int, str]] = []
        self.completed_records = 0

    def reset(self) -> None:
        with torch.no_grad():
            self.expert_weight.copy_(self._initial_expert_weight)
            self.expert_bias.copy_(self._initial_expert_bias)
            self.router_logits.copy_(self._initial_router_logits)
        self.expert_optimizer.zero_grad(set_to_none=True)
        self.router_optimizer.zero_grad(set_to_none=True)
        self.feedback.reset()
        self.correction.reset()
        self.memory.clear()
        self.metric_tracker.reset()
        self.memory_recorder.reset()
        self.previous_gradients = None
        self.sample_regimes.clear()
        self.protocol_trace.clear()
        self.completed_records = 0

    def _forecast_experts(self, context: torch.Tensor) -> torch.Tensor:
        """Return [H,C,E] recursive forecasts from context-only state."""

        states = context[-1].reshape(1, -1).expand(self.config.num_experts, -1)
        forecasts = []
        for _ in range(self.config.pred_len):
            states = torch.einsum("eoc,ec->eo", self.expert_weight, states)
            states = states + self.expert_bias
            forecasts.append(states.transpose(0, 1))
        return torch.stack(forecasts, dim=0)

    def _capability_sketch(self) -> torch.Tensor:
        flattened = torch.cat(
            [
                self.expert_weight.reshape(self.config.num_experts, -1),
                self.expert_bias.reshape(self.config.num_experts, -1),
            ],
            dim=1,
        )
        return F.normalize(flattened, p=2, dim=-1, eps=1e-8)

    def _alignment(self, old_sketch: torch.Tensor, expert_id: int) -> float:
        if self.config.disable_version_awareness:
            return 1.0
        current = self._capability_sketch()[expert_id]
        cosine = torch.dot(old_sketch.float(), current).clamp(-1.0, 1.0)
        return float(((cosine + 1.0) / 2.0).item())

    def _update_router(self, event: Any) -> None:
        horizon = int(event.horizon_index)
        if not self.config.disable_z_correction:
            self.correction.update(
                horizon,
                event.record.expert_predictions[horizon],
                event.record.mixture_prediction[horizon],
                event.target,
            )
        self.router_optimizer.zero_grad(set_to_none=True)
        current_prior = torch.softmax(self.router_logits[horizon], dim=-1).unsqueeze(0)
        correction = (
            self.correction.z[horizon].unsqueeze(0)
            if not self.config.disable_z_correction
            else torch.zeros_like(current_prior)
        )
        loss, _ = partial_router_objective(
            current_prior=current_prior,
            correction=correction,
            expert_prediction=event.record.expert_predictions[horizon].unsqueeze(0),
            target=event.target.unsqueeze(0),
            local_responsibility=event.local_responsibility.unsqueeze(0),
            local_confidence=event.local_confidence.unsqueeze(0),
            local_credit_weight=0.1,
            entropy_weight=0.001,
            top_k=self.config.num_experts,
        )
        loss.backward()
        self.router_optimizer.step()

    def _apply_tsb(
        self,
        weight_gradient: torch.Tensor,
        bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.previous_gradients is None:
            return weight_gradient, bias_gradient
        filtered = []
        for current, reference in zip(
            (weight_gradient, bias_gradient), self.previous_gradients
        ):
            base = 0.75 * current + 0.25 * reference
            dot = torch.sum(base * reference)
            if float(dot.item()) < 0.0:
                base = base - dot / (torch.sum(reference * reference) + 1e-8) * reference
            filtered.append(base)
        return filtered[0], filtered[1]

    def _apply_subspace(self, gradient: torch.Tensor) -> torch.Tensor:
        filtered = gradient.clone()
        for expert_id, buffer in enumerate(self.memory.stable_buffers):
            if not buffer.items:
                continue
            features = torch.stack(
                [item.x.float()[0, -1] for item in buffer.items], dim=0
            )
            _, singular, right = torch.linalg.svd(features, full_matrices=False)
            if singular.numel() == 0 or float(singular[0].item()) <= 1e-8:
                continue
            rank = min(2, int((singular > singular[0] * 1e-4).sum().item()))
            basis = right[:rank].transpose(0, 1)
            rows = filtered[expert_id]
            filtered[expert_id] = rows - (rows @ basis) @ basis.transpose(0, 1)
        return filtered

    def _memory_evaluator(
        self, expert_id: int, item: VersionedMemoryItem
    ) -> tuple[float, float, torch.Tensor]:
        alignment = self._alignment(item.normalized_sketch, expert_id)
        prediction = self._forecast_experts(item.x.float()[0])[..., expert_id]
        target = item.target.float()[0]
        loss = float((prediction - target).pow(2).mean().item())
        return alignment, loss, self._capability_sketch()[expert_id]

    def _candidate_for_record(
        self, record: ProgressiveForecastRecord, timestamp: int
    ) -> VersionedMemoryItem:
        owner = int(record.sample_responsibility.argmax().item())
        alignment = self._alignment(record.capability_sketch[owner], owner)
        current = self._forecast_experts(record.x.float()[0])[..., owner]
        current_loss = float(
            (current - record.matured_targets).pow(2).mean().item()
        )
        prediction_time_loss = float(
            (
                record.expert_predictions[..., owner]
                - record.matured_targets
            ).pow(2).mean().item()
        )
        responsibility = float(record.sample_responsibility[owner].item())
        reference_sketch = record.capability_sketch[owner]
        reference_loss = prediction_time_loss
        admission_alignment = alignment
        recovery_eligible = not self.config.disable_recovery
        capability_evolution = "retained"
        if self.memory.direction_awareness_enabled:
            decision = classify_capability_evolution(
                alignment=alignment,
                prediction_loss=prediction_time_loss,
                current_loss=current_loss,
                alignment_threshold=self.memory.alignment_threshold,
                degradation_margin=0.0,
            )
            capability_evolution = decision.category
            low_alignment = alignment < self.memory.alignment_threshold
            recovery_eligible = bool(
                not self.config.disable_recovery
                and low_alignment
                and decision.category == "harmful_drift"
            )
            if (
                low_alignment
                and decision.category
                == "beneficial_or_neutral_evolution"
            ):
                reference_sketch = self._capability_sketch()[owner]
                reference_loss = current_loss
                admission_alignment = 1.0
        return VersionedMemoryItem(
            sample_id=record.origin,
            origin=record.origin,
            expert_id=owner,
            x=record.x,
            x_mark=record.x_mark,
            target=record.matured_targets.unsqueeze(0),
            prediction_capability_sketch=record.capability_sketch[owner],
            normalized_sketch=reference_sketch,
            sample_responsibility=responsibility,
            sample_confidence=float(record.sample_confidence),
            last_alignment=admission_alignment,
            stable_credit=responsibility * admission_alignment,
            recovery_credit=responsibility * (1.0 - admission_alignment),
            timestamp=timestamp,
            recent_prediction_loss=current_loss,
            prediction_time_loss=prediction_time_loss,
            reference_capability_loss=reference_loss,
            last_observed_alignment=alignment,
            recovery_eligible=recovery_eligible,
            capability_evolution=capability_evolution,
        )

    @staticmethod
    def _empty_update_stats() -> dict[str, int]:
        return {
            "retained": 0,
            "low_alignment": 0,
            "harmful_drift": 0,
            "beneficial_evolution": 0,
            "capability_rebase": 0,
            "stable_to_recovery": 0,
            "recovery_admission": 0,
            "recovery_attempt": 0,
            "performance_recovery": 0,
            "recovery_to_stable": 0,
            "recovery_failed": 0,
            "recovery_attempt_exhausted": 0,
            "recovery_dropped_after_success": 0,
            "recovery_skipped_non_degraded": 0,
            "matured_records": 0,
        }

    @staticmethod
    def _merge_update_stats(
        destination: dict[str, int], source: dict[str, int]
    ) -> None:
        for key in destination:
            destination[key] += int(source.get(key, 0))

    def _candidate_stats(
        self, candidate: VersionedMemoryItem
    ) -> dict[str, int]:
        stats = self._empty_update_stats()
        stats["matured_records"] = 1
        low_alignment = bool(
            self.memory.version_awareness_enabled
            and candidate.last_observed_alignment
            < self.memory.alignment_threshold
        )
        stats["low_alignment"] = int(low_alignment)
        stats["retained"] = int(not low_alignment)
        if candidate.capability_evolution == "harmful_drift":
            stats["harmful_drift"] = 1
        elif (
            candidate.capability_evolution
            == "beneficial_or_neutral_evolution"
        ):
            stats["beneficial_evolution"] = 1
            if low_alignment and self.memory.direction_awareness_enabled:
                stats["capability_rebase"] = 1
                stats["recovery_skipped_non_degraded"] = 1
        return stats

    def _update_experts(
        self, record: ProgressiveForecastRecord, timestamp: int
    ) -> dict[str, int]:
        candidate = self._candidate_for_record(record, timestamp)
        stats = self._candidate_stats(candidate)
        replay_items: list[tuple[int, VersionedMemoryItem]] = []
        if self.memory.recovery_enabled:
            excluded: set[int] = set()
            for expert_id in range(self.config.num_experts):
                replay_items.extend(
                    (expert_id, item)
                    for item in self.memory.sample_recovery(
                        expert_id, 1, excluded_sample_ids=excluded
                    )
                )
        stats["recovery_attempt"] += len(replay_items)

        self.expert_optimizer.zero_grad(set_to_none=True)
        prediction = self._forecast_experts(record.x.float()[0])
        loss = (prediction - record.matured_targets.unsqueeze(-1)).pow(2).mean()
        for expert_id, item in replay_items:
            replay_prediction = self._forecast_experts(
                item.x.float()[0]
            )[..., expert_id]
            loss = loss + 0.1 * (
                replay_prediction - item.target.float()[0]
            ).pow(2).mean()
        loss.backward()
        raw_weight = self.expert_weight.grad.detach().clone()
        raw_bias = self.expert_bias.grad.detach().clone()
        filtered_weight, filtered_bias = raw_weight, raw_bias
        if self.config.strategy in {"tsb", "hybrid"}:
            filtered_weight, filtered_bias = self._apply_tsb(
                filtered_weight, filtered_bias
            )
        if self.config.strategy in {"subspace", "hybrid"}:
            filtered_weight = self._apply_subspace(filtered_weight)
        self.previous_gradients = (raw_weight, raw_bias)
        self.expert_weight.grad.copy_(filtered_weight)
        self.expert_bias.grad.copy_(filtered_bias)
        torch.nn.utils.clip_grad_norm_(
            [self.expert_weight, self.expert_bias], 5.0
        )
        self.expert_optimizer.step()

        for expert_id, item in replay_items:
            original_prediction_loss = item.prediction_time_loss
            alignment, replay_loss, current_sketch = self._memory_evaluator(
                expert_id, item
            )
            status = self.memory.update_recovery_result(
                expert_id,
                item.sample_id,
                alignment,
                replay_loss,
                current_sketch=current_sketch,
            )
            if (
                self.memory.version_awareness_enabled
                and alignment < self.memory.alignment_threshold
            ):
                stats["low_alignment"] += 1
            else:
                stats["retained"] += 1
            if item.capability_evolution == "harmful_drift":
                stats["harmful_drift"] += 1
            elif (
                item.capability_evolution
                == "beneficial_or_neutral_evolution"
            ):
                stats["beneficial_evolution"] += 1
            if item.prediction_time_loss != original_prediction_loss:
                raise RuntimeError("prediction_time_loss changed during rebase")
            if status == "promoted_performance":
                stats["performance_recovery"] += 1
                stats["capability_rebase"] += 1
                stats["recovery_to_stable"] += 1
            elif status == "promoted":
                stats["recovery_to_stable"] += 1
            elif status == "dropped_after_performance_recovery":
                stats["performance_recovery"] += 1
                stats["capability_rebase"] += 1
                stats["recovery_dropped_after_success"] += 1
            elif status == "dropped_after_recovery":
                stats["recovery_dropped_after_success"] += 1
            elif status == "dropped":
                stats["recovery_failed"] += 1
                stats["recovery_attempt_exhausted"] += 1
            elif status == "recovery":
                stats["recovery_failed"] += 1

        kind, admission = self.memory.add_candidate_with_result(candidate)
        if kind == "recovery" and admission.accepted:
            stats["recovery_admission"] += 1
        self.sample_regimes[record.origin] = self.dataset.target_regime_at(
            record.origin
        )
        self.completed_records += 1
        if self.completed_records % self.config.memory_refresh_interval == 0:
            def tracked_evaluator(
                expert_id: int, item: VersionedMemoryItem
            ) -> tuple[float, float, torch.Tensor]:
                evaluation = self._memory_evaluator(expert_id, item)
                alignment = evaluation[0]
                if (
                    self.memory.version_awareness_enabled
                    and alignment < self.memory.alignment_threshold
                ):
                    stats["low_alignment"] += 1
                else:
                    stats["retained"] += 1
                return evaluation

            refresh = self.memory.refresh(
                tracked_evaluator,
                timestamp=timestamp,
                count_recovery_attempts=False,
            )
            self._merge_update_stats(stats, refresh)
        return stats

    def _predict_record(self, origin: int) -> tuple[ProgressiveForecastRecord, torch.Tensor]:
        context = self.dataset.context_at(origin)
        x_mark = self.dataset.context_marks_at(origin)
        with torch.no_grad():
            expert_prediction = self._forecast_experts(context)
            prior = torch.softmax(self.router_logits, dim=-1)
            weights = (
                self.correction.effective_weights(
                    prior, top_k=self.config.num_experts
                )
                if not self.config.disable_z_correction
                else prior
            )
            mixture = (weights * expert_prediction).sum(dim=-1)
            sketch = self._capability_sketch()
        record = ProgressiveForecastRecord(
            origin=origin,
            x=context.unsqueeze(0),
            x_mark=x_mark.unsqueeze(0),
            expert_predictions=expert_prediction,
            router_prior=prior,
            router_weights=weights,
            mixture_prediction=mixture,
            capability_sketch=sketch,
        )
        return record, mixture

    def _regime_diagnostics(
        self,
        mse: np.ndarray,
        phases: np.ndarray,
        memory_arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "rate_definition": "raw events / matured records",
            "events_per_1000_definition": (
                "1000 * raw events / matured records"
            ),
            "regimes": {},
        }
        event_names = (
            "harmful_drift",
            "beneficial_evolution",
            "stable_to_recovery",
            "recovery_admission",
            "recovery_attempt",
            "performance_recovery",
            "recovery_to_stable",
            "recovery_failed",
            "recovery_attempt_exhausted",
            "recovery_dropped_after_success",
            "recovery_skipped_non_degraded",
            "capability_rebase",
        )
        stable_size = memory_arrays["stable_occupancy"].sum(axis=1)
        recovery_size = memory_arrays["recovery_occupancy"].sum(axis=1)
        matured_events = memory_arrays["matured_records_events"]
        for phase in ("A1", "B", "A2"):
            mask = phases == phase
            indices = np.flatnonzero(mask)
            matured = int(matured_events[mask].sum())
            phase_result: dict[str, Any] = {
                "origin_start": (
                    int(indices[0]) if indices.size else None
                ),
                "origin_end": (
                    int(indices[-1] + 1) if indices.size else None
                ),
                "num_origins": int(indices.size),
                "matured_records": matured,
                "mean_mse": (
                    float(mse[mask].mean()) if indices.size else None
                ),
                "mean_stable_occupancy": (
                    float(stable_size[mask].mean())
                    if indices.size
                    else None
                ),
                "mean_recovery_occupancy": (
                    float(recovery_size[mask].mean())
                    if indices.size
                    else None
                ),
                "events": {},
            }
            for name in event_names:
                count = int(memory_arrays[f"{name}_events"][mask].sum())
                phase_result["events"][name] = {
                    "count": count,
                    "rate": (
                        float(count / matured) if matured else None
                    ),
                    "events_per_1000_matured_records": (
                        float(1000.0 * count / matured)
                        if matured
                        else None
                    ),
                }
            diagnostics["regimes"][phase] = phase_result
        return diagnostics

    def run(self, output_directory: str | None = None) -> SyntheticBenchmarkResult:
        self.reset()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        horizon_mse: list[float] = []
        for origin in range(len(self.dataset)):
            self.protocol_trace.append((origin, "release"))
            if not self.config.disable_z_correction:
                self.correction.begin_origin(origin)
            events, completed = self.feedback.release(
                origin, self.dataset.observation_at(origin)
            )
            for event in events:
                self._update_router(event)
            transition_stats = None
            for record in completed:
                self.protocol_trace.append((origin, "completed_update"))
                current_stats = self._update_experts(record, origin)
                if transition_stats is None:
                    transition_stats = current_stats
                else:
                    for key, value in current_stats.items():
                        transition_stats[key] += value

            self.protocol_trace.append((origin, "predict"))
            record, prediction = self._predict_record(origin)
            if not torch.isnan(record.matured_targets).all():
                raise RuntimeError("future target leaked into a new forecast record")
            self.feedback.add_record(record)
            self.protocol_trace.append((origin, "evaluate"))
            target = self.dataset.evaluation_target_at(origin)
            first_step_mse = float((prediction[0] - target[0]).pow(2).mean().item())
            full_mse = float((prediction - target).pow(2).mean().item())
            self.metric_tracker.update(first_step_mse)
            horizon_mse.append(full_mse)
            predictions.append(prediction.numpy())
            targets.append(target.numpy())
            self.memory_recorder.record(
                origin,
                self.memory,
                transition_stats=transition_stats,
                sample_regimes=self.sample_regimes,
            )

        metrics = compute_drift_metrics(
            self.metric_tracker.mse,
            self.dataset.events,
            pre_window=self.config.pre_window,
            early_window=self.config.early_window,
            recovery_window=self.config.recovery_window,
            recovery_tolerance=self.config.recovery_tolerance,
            recovery_hold_steps=self.config.recovery_hold_steps,
            recurring_intervals=self.dataset.intervals,
            seq_len=self.config.seq_len,
        )
        mse_timeline = np.asarray(
            self.metric_tracker.mse, dtype=np.float64
        )
        memory_arrays = self.memory_recorder.arrays()
        origins = np.arange(len(self.dataset), dtype=np.int64)
        origin_regime_id = np.asarray(
            [self.dataset.target_regime_at(i) for i in origins],
            dtype=np.int64,
        )
        origin_phase = np.asarray(
            [self.dataset.target_phase_at(i) for i in origins],
            dtype="<U2",
        )
        rolling_mse = rolling_mean(
            mse_timeline, self.config.rolling_window
        )
        regime_diagnostics = self._regime_diagnostics(
            mse_timeline, origin_phase, memory_arrays
        )
        memory_diagnostics = self.memory_recorder.summary()
        memory_diagnostics["regime_diagnostics"] = regime_diagnostics
        result = SyntheticBenchmarkResult(
            predictions=np.asarray(predictions, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            mse_timeline=mse_timeline,
            horizon_mse_timeline=np.asarray(horizon_mse, dtype=np.float64),
            series=self.dataset.series.copy(),
            regime_id=self.dataset.regime_id.copy(),
            transition_alpha=self.dataset.transition_alpha.copy(),
            drift_metrics=metrics,
            memory_diagnostics=memory_diagnostics,
            event_metadata=dict(self.dataset.metadata),
            memory_arrays=memory_arrays,
            regime_diagnostics=regime_diagnostics,
            origin=origins,
            origin_regime_id=origin_regime_id,
            origin_phase=origin_phase,
            rolling_mse=rolling_mse,
            completed_records=self.completed_records,
            protocol_trace=tuple(self.protocol_trace),
        )
        if output_directory is not None:
            self.save(result, output_directory)
        return result

    @staticmethod
    def save(result: SyntheticBenchmarkResult, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            os.path.join(directory, "predictions_and_mse.npz"),
            predictions=result.predictions,
            targets=result.targets,
            mse_timeline=result.mse_timeline,
            horizon_mse_timeline=result.horizon_mse_timeline,
        )
        with open(
            os.path.join(directory, "drift_events.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(result.event_metadata, handle, indent=2, ensure_ascii=False)
        with open(
            os.path.join(directory, "drift_metrics.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(result.drift_metrics, handle, indent=2, ensure_ascii=False)
        with open(
            os.path.join(directory, "memory_diagnostics.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(result.memory_diagnostics, handle, indent=2, ensure_ascii=False)
        with open(
            os.path.join(directory, "regime_diagnostics.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(result.regime_diagnostics, handle, indent=2, ensure_ascii=False)
        np.savez_compressed(
            os.path.join(directory, "synthetic_timeline.npz"),
            series=result.series,
            regime_id=result.regime_id,
            transition_alpha=result.transition_alpha,
            **result.memory_arrays,
        )
        np.savez_compressed(
            os.path.join(directory, "timeline.npz"),
            origin=result.origin,
            regime_id=result.origin_regime_id,
            phase=result.origin_phase,
            mse=result.mse_timeline,
            rolling_mse=result.rolling_mse,
            stable_size=result.memory_arrays["stable_occupancy"].sum(axis=1),
            recovery_size=result.memory_arrays[
                "recovery_occupancy"
            ].sum(axis=1),
            harmful_drift_events=result.memory_arrays[
                "harmful_drift_events"
            ],
            beneficial_evolution_events=result.memory_arrays[
                "beneficial_evolution_events"
            ],
            capability_rebase_events=result.memory_arrays[
                "capability_rebase_events"
            ],
            stable_to_recovery_events=result.memory_arrays[
                "stable_to_recovery_events"
            ],
            recovery_admission_events=result.memory_arrays[
                "recovery_admission_events"
            ],
            recovery_attempt_events=result.memory_arrays[
                "recovery_attempt_events"
            ],
            recovery_to_stable_events=result.memory_arrays[
                "recovery_to_stable_events"
            ],
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="result/synthetic_drift")
    parser.add_argument(
        "--drift_type", choices=sorted({"abrupt", "gradual", "recurring", "shock"}), default="recurring"
    )
    parser.add_argument("--strategy", choices=sorted(_STRATEGIES), default="hybrid")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=3)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--total_length", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--regime_separation", type=float, default=1.0)
    parser.add_argument("--a1_length", type=int, default=None)
    parser.add_argument("--b_length", type=int, default=None)
    parser.add_argument("--a2_length", type=int, default=None)
    parser.add_argument("--transition_window", type=int, default=None)
    parser.add_argument("--shock_duration", type=int, default=None)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--synthetic_drift_channels", default="")
    parser.add_argument("--recovery_window", type=int, default=4)
    parser.add_argument("--recovery_tolerance", type=float, default=0.2)
    parser.add_argument("--recovery_hold_steps", type=int, default=2)
    parser.add_argument("--rolling_window", type=int, default=8)
    parser.add_argument("--disable_recovery", action="store_true")
    parser.add_argument("--disable_directional_recovery", action="store_true")
    parser.add_argument("--disable_version_awareness", action="store_true")
    parser.add_argument("--disable_z_correction", action="store_true")
    return parser


def result_directory(
    output_root: str, config: SyntheticBenchmarkConfig
) -> str:
    variant_tags = [
        config.drift_type,
        config.strategy,
        f"seed{config.seed}",
    ]
    if config.disable_recovery:
        variant_tags.append("no_recovery")
    if config.disable_directional_recovery:
        variant_tags.append("no_direction")
    if config.disable_version_awareness:
        variant_tags.append("no_version")
    if config.disable_z_correction:
        variant_tags.append("no_z")
    return os.path.join(output_root, "_".join(variant_tags))


def main() -> None:
    arguments = _parser().parse_args()
    explicit_lengths = (
        arguments.a1_length,
        arguments.b_length,
        arguments.a2_length,
    )
    total_length = arguments.total_length
    if all(value is not None for value in explicit_lengths):
        total_length = sum(int(value) for value in explicit_lengths)
    config = SyntheticBenchmarkConfig(
        seq_len=arguments.seq_len,
        pred_len=arguments.pred_len,
        channels=arguments.channels,
        total_length=total_length,
        seed=arguments.seed,
        noise_std=arguments.noise_std,
        regime_separation=arguments.regime_separation,
        a1_length=arguments.a1_length,
        b_length=arguments.b_length,
        a2_length=arguments.a2_length,
        transition_window=arguments.transition_window,
        shock_duration=arguments.shock_duration,
        drift_type=arguments.drift_type,
        drift_channels=parse_drift_channels(arguments.synthetic_drift_channels),
        num_experts=arguments.num_experts,
        strategy=arguments.strategy,
        recovery_window=arguments.recovery_window,
        recovery_tolerance=arguments.recovery_tolerance,
        recovery_hold_steps=arguments.recovery_hold_steps,
        rolling_window=arguments.rolling_window,
        disable_recovery=arguments.disable_recovery,
        disable_directional_recovery=arguments.disable_directional_recovery,
        disable_version_awareness=arguments.disable_version_awareness,
        disable_z_correction=arguments.disable_z_correction,
    )
    output = result_directory(arguments.output_dir, config)
    result = SyntheticProgressiveBenchmark(config).run(output)
    print(
        json.dumps(
            {
                "output": output,
                "origins": int(result.mse_timeline.size),
                "completed_records": result.completed_records,
                "mean_mse": float(result.mse_timeline.mean()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
