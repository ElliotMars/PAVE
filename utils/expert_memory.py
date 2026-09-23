import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Set, Tuple

import torch
import torch.nn.functional as F


MemoryKind = Literal["stable", "recovery"]
CapabilityEvolution = Literal[
    "retained", "harmful_drift", "beneficial_or_neutral_evolution"
]


@dataclass(frozen=True)
class CapabilityEvolutionDecision:
    category: CapabilityEvolution
    degraded: bool
    relative_degradation: float


def classify_capability_evolution(
    alignment: float,
    prediction_loss: float,
    current_loss: float,
    alignment_threshold: float,
    degradation_margin: float = 0.0,
    eps: float = 1e-8,
) -> CapabilityEvolutionDecision:
    """Classify representation drift by its historical performance direction."""

    values = (
        float(alignment),
        float(prediction_loss),
        float(current_loss),
        float(alignment_threshold),
        float(degradation_margin),
        float(eps),
    )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("capability evolution inputs contain NaN or Inf")
    if prediction_loss < 0.0 or current_loss < 0.0:
        raise ValueError("capability losses must be non-negative")
    if degradation_margin < 0.0 or eps <= 0.0:
        raise ValueError(
            "degradation margin must be non-negative and eps positive"
        )
    degraded = (
        current_loss
        > prediction_loss * (1.0 + degradation_margin) + eps
    )
    relative = (current_loss - prediction_loss) / (prediction_loss + eps)
    if alignment >= alignment_threshold:
        category: CapabilityEvolution = "retained"
    elif degraded:
        category = "harmful_drift"
    else:
        category = "beneficial_or_neutral_evolution"
    return CapabilityEvolutionDecision(category, degraded, relative)



def _storage_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError("buffer_storage_dtype must be fp16 or fp32")


@dataclass
class VersionedMemoryItem:
    sample_id: int
    origin: int
    expert_id: int
    x: torch.Tensor
    x_mark: torch.Tensor
    target: torch.Tensor
    prediction_capability_sketch: torch.Tensor
    sample_responsibility: float
    last_alignment: float
    stable_credit: float
    recovery_credit: float
    timestamp: int
    age: int = 0
    recovery_attempts: int = 0
    recent_prediction_loss: float = float("inf")
    normalized_sketch: Optional[torch.Tensor] = None
    sample_confidence: float = 1.0
    prediction_time_loss: float = float("inf")
    reference_capability_loss: Optional[float] = None
    last_observed_alignment: float = 1.0
    recovery_eligible: bool = True
    capability_evolution: CapabilityEvolution = "retained"

    def __post_init__(self) -> None:
        self.sample_confidence = float(self.sample_confidence)
        self.prediction_time_loss = float(self.prediction_time_loss)
        if self.reference_capability_loss is None:
            self.reference_capability_loss = self.prediction_time_loss
        else:
            self.reference_capability_loss = float(
                self.reference_capability_loss
            )
        self.last_observed_alignment = float(self.last_observed_alignment)
        self.recovery_eligible = bool(self.recovery_eligible)
        if not math.isfinite(self.sample_confidence):
            raise FloatingPointError("sample_confidence contains NaN or Inf")
        if not 0.0 <= self.sample_confidence <= 1.0:
            raise ValueError("sample_confidence must be in [0,1]")
        self.refresh_credit()

    def get_reference_capability_loss(self) -> float:
        """Return the current loss reference, migrating old items lazily."""

        value = getattr(self, "reference_capability_loss", None)
        if value is None:
            value = getattr(self, "prediction_time_loss", float("inf"))
            self.reference_capability_loss = float(value)
        return float(value)

    def refresh_credit(self) -> None:
        """Recompute confidence-aware long-term credit at current alignment."""

        self.stable_credit = (
            self.sample_confidence
            * self.sample_responsibility
            * self.last_alignment
        )
        self.recovery_credit = (
            self.sample_confidence
            * self.sample_responsibility
            * (1.0 - self.last_alignment)
            if self.recovery_eligible
            else 0.0
        )
        if not (
            math.isfinite(self.stable_credit)
            and math.isfinite(self.recovery_credit)
        ):
            raise FloatingPointError("memory credit contains NaN or Inf")

    def to_cpu_storage(self, dtype: torch.dtype) -> "VersionedMemoryItem":
        def snapshot(tensor: torch.Tensor) -> torch.Tensor:
            stored = tensor.detach().to(device="cpu").clone()
            return stored.to(dtype=dtype) if stored.is_floating_point() else stored

        self.x = snapshot(self.x)
        self.x_mark = snapshot(self.x_mark)
        self.target = snapshot(self.target)
        self.prediction_capability_sketch = snapshot(
            self.prediction_capability_sketch
        )
        sketch = (
            self.prediction_capability_sketch
            if self.normalized_sketch is None
            else snapshot(self.normalized_sketch)
        )
        self.normalized_sketch = F.normalize(
            sketch.float(), p=2, dim=-1, eps=1e-8
        ).to(dtype=dtype)
        return self

    def rebase_capability_reference(
        self,
        current_sketch: torch.Tensor,
        current_loss: float,
        dtype: torch.dtype,
    ) -> None:
        """Atomically rebase the sketch and loss to one capability version."""

        stored = current_sketch.detach().to(device="cpu").clone().float()
        if (
            stored.ndim != 1
            or not bool(torch.isfinite(stored).all().item())
        ):
            raise ValueError(
                "current capability sketch must be finite and one-dimensional"
            )
        reference_loss = float(current_loss)
        if not math.isfinite(reference_loss) or reference_loss < 0.0:
            raise ValueError(
                "current capability loss must be finite and non-negative"
            )
        normalized = F.normalize(stored, p=2, dim=-1, eps=1e-8)
        self.prediction_capability_sketch = normalized.to(dtype=dtype)
        self.normalized_sketch = normalized.to(dtype=dtype)
        self.reference_capability_loss = reference_loss

    def rebase_capability_sketch(
        self, current_sketch: torch.Tensor, dtype: torch.dtype
    ) -> None:
        """Backward-compatible synchronized reference rebase."""

        self.rebase_capability_reference(
            current_sketch, self.recent_prediction_loss, dtype
        )

    @property
    def stable_score(self) -> float:
        return self.stable_credit

    def recovery_score(self, failure_penalty: float) -> float:
        return (
            self.recovery_credit
            / (1.0 + failure_penalty * self.recovery_attempts)
        )


@dataclass(frozen=True)
class BufferAddResult:
    """Detailed outcome of one fixed-capacity buffer insertion."""

    accepted: bool
    replaced_item: Optional[VersionedMemoryItem]
    reason: str


class FixedCapacityExpertBuffer:
    def __init__(
        self,
        capacity: int,
        kind: MemoryKind,
        duplicate_threshold: float,
        failure_penalty: float,
    ) -> None:
        if capacity < 0:
            raise ValueError("buffer capacity must be non-negative")
        if kind not in {"stable", "recovery"}:
            raise ValueError("invalid buffer kind")
        self.capacity = int(capacity)
        self.kind = kind
        self.duplicate_threshold = float(duplicate_threshold)
        self.failure_penalty = float(failure_penalty)
        self._items: List[VersionedMemoryItem] = []

    def _score(self, item: VersionedMemoryItem) -> float:
        if self.kind == "stable":
            return item.stable_score
        return item.recovery_score(self.failure_penalty)

    def add_with_result(self, item: VersionedMemoryItem) -> BufferAddResult:
        """Insert an item and report whether an existing item was replaced."""

        if self.capacity == 0:
            return BufferAddResult(False, None, "rejected")

        same_sample = next(
            (i for i, old in enumerate(self._items) if old.sample_id == item.sample_id),
            None,
        )
        if same_sample is not None:
            if self._score(item) > self._score(self._items[same_sample]):
                replaced = self._items[same_sample]
                self._items[same_sample] = item
                return BufferAddResult(True, replaced, "duplicate_replaced")
            return BufferAddResult(False, None, "duplicate_rejected")

        if self.kind == "stable" and self._items:
            query = item.normalized_sketch.float()
            similarities = torch.tensor(
                [
                    float(
                        torch.dot(query, old.normalized_sketch.float()).clamp(-1, 1)
                    )
                    for old in self._items
                ]
            )
            duplicate_index = int(similarities.argmax().item())
            if float(similarities[duplicate_index].item()) > self.duplicate_threshold:
                if self._score(item) > self._score(self._items[duplicate_index]):
                    replaced = self._items[duplicate_index]
                    self._items[duplicate_index] = item
                    return BufferAddResult(True, replaced, "duplicate_replaced")
                return BufferAddResult(False, None, "duplicate_rejected")

        if len(self._items) < self.capacity:
            self._items.append(item)
            return BufferAddResult(True, None, "inserted")
        lowest_index = min(
            range(len(self._items)), key=lambda index: self._score(self._items[index])
        )
        if self._score(item) > self._score(self._items[lowest_index]):
            replaced = self._items[lowest_index]
            self._items[lowest_index] = item
            return BufferAddResult(True, replaced, "replaced")
        return BufferAddResult(False, None, "rejected")

    def add(self, item: VersionedMemoryItem) -> bool:
        """Backward-compatible boolean insertion API."""

        return self.add_with_result(item).accepted

    def remove(self, sample_id: int) -> Optional[VersionedMemoryItem]:
        for index, item in enumerate(self._items):
            if item.sample_id == sample_id:
                return self._items.pop(index)
        return None

    def contains(self, sample_id: int) -> bool:
        return any(item.sample_id == sample_id for item in self._items)

    def clear(self) -> None:
        self._items.clear()

    @property
    def items(self) -> Tuple[VersionedMemoryItem, ...]:
        return tuple(self._items)

    def get(self, sample_id: int) -> Optional[VersionedMemoryItem]:
        for item in self._items:
            if item.sample_id == sample_id:
                return item
        return None

    def __len__(self) -> int:
        return len(self._items)


class ExpertMemoryManager:
    def __init__(
        self,
        num_experts: int,
        stable_capacity: int,
        recovery_capacity: int,
        responsibility_threshold: float,
        alignment_threshold: float,
        duplicate_threshold: float,
        failure_penalty: float,
        max_recovery_attempts: int,
        storage_dtype: str = "fp16",
        promote_alignment_threshold: float = 0.9,
        promote_loss_threshold: float = 1.0,
        directional_recovery_enabled: bool = False,
        version_awareness_enabled: bool = True,
        recovery_degradation_margin: float = 0.0,
        degradation_eps: float = 1e-8,
        capability_rebase_enabled: bool = True,
        recovery_enabled: bool = True,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts must be non-negative")
        self.num_experts = int(num_experts)
        self.responsibility_threshold = float(responsibility_threshold)
        self.alignment_threshold = float(alignment_threshold)
        self.max_recovery_attempts = int(max_recovery_attempts)
        self.storage_dtype = _storage_dtype(storage_dtype)
        self.promote_alignment_threshold = float(promote_alignment_threshold)
        self.promote_loss_threshold = float(promote_loss_threshold)
        self.version_awareness_enabled = bool(version_awareness_enabled)
        self.direction_awareness_enabled = bool(
            directional_recovery_enabled
            and self.version_awareness_enabled
        )
        # Backward-compatible name for existing diagnostics and callers.
        self.directional_recovery_enabled = self.direction_awareness_enabled
        self.recovery_enabled = bool(recovery_enabled)
        self.recovery_degradation_margin = float(
            recovery_degradation_margin
        )
        self.degradation_eps = float(degradation_eps)
        self.capability_rebase_enabled = bool(capability_rebase_enabled)
        if (
            not math.isfinite(self.recovery_degradation_margin)
            or self.recovery_degradation_margin < 0.0
        ):
            raise ValueError(
                "recovery_degradation_margin must be finite and non-negative"
            )
        if not math.isfinite(self.degradation_eps) or self.degradation_eps <= 0.0:
            raise ValueError("degradation_eps must be finite and positive")
        self.stable_buffers = [
            FixedCapacityExpertBuffer(
                stable_capacity, "stable", duplicate_threshold, failure_penalty
            )
            for _ in range(num_experts)
        ]
        self.recovery_buffers = [
            FixedCapacityExpertBuffer(
                recovery_capacity, "recovery", duplicate_threshold, failure_penalty
            )
            for _ in range(num_experts)
        ]

    def add_candidate_with_result(
        self, item: VersionedMemoryItem
    ) -> tuple[Optional[MemoryKind], BufferAddResult]:
        """Admit a candidate while preserving the buffer insertion outcome."""

        expert_id = int(item.expert_id)
        if not 0 <= expert_id < self.num_experts:
            raise IndexError("invalid expert_id")
        if item.sample_responsibility < self.responsibility_threshold:
            return None, BufferAddResult(False, None, "rejected")
        item.to_cpu_storage(self.storage_dtype)
        kind: MemoryKind = (
            "stable"
            if item.last_alignment >= self.alignment_threshold
            else "recovery"
        )
        if kind == "recovery" and not self.recovery_enabled:
            return None, BufferAddResult(
                False, None, "recovery_disabled"
            )
        if self.direction_awareness_enabled and kind == "stable":
            item.recovery_eligible = False
            item.refresh_credit()
        if (
            kind == "recovery"
            and self.direction_awareness_enabled
            and not item.recovery_eligible
        ):
            return None, BufferAddResult(
                False, None, "directional_rejected"
            )
        if (
            self.direction_awareness_enabled
            and not math.isfinite(item.get_reference_capability_loss())
        ):
            raise ValueError(
                "directional Recovery requires finite reference_capability_loss"
            )
        target = (
            self.stable_buffers[expert_id]
            if kind == "stable"
            else self.recovery_buffers[expert_id]
        )
        other = (
            self.recovery_buffers[expert_id]
            if kind == "stable"
            else self.stable_buffers[expert_id]
        )
        result = target.add_with_result(item)
        if result.accepted:
            other.remove(item.sample_id)
            return kind, result
        return None, result

    def add_candidate(self, item: VersionedMemoryItem) -> Optional[MemoryKind]:
        """Backward-compatible candidate admission API."""

        kind, _ = self.add_candidate_with_result(item)
        return kind

    @staticmethod
    def _unpack_evaluation(
        evaluation: tuple,
    ) -> tuple[float, float, Optional[torch.Tensor]]:
        if len(evaluation) not in {2, 3}:
            raise ValueError(
                "memory evaluator must return alignment, loss, and optional sketch"
            )
        alignment = float(evaluation[0])
        current_loss = float(evaluation[1])
        current_sketch = evaluation[2] if len(evaluation) == 3 else None
        values = torch.tensor(
            [alignment, current_loss], dtype=torch.float64
        )
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError(
                "memory evaluation contains NaN or Inf"
            )
        return alignment, current_loss, current_sketch

    def _direction_decision(
        self,
        item: VersionedMemoryItem,
        alignment: float,
        current_loss: float,
    ) -> Optional[CapabilityEvolutionDecision]:
        reference_loss = item.get_reference_capability_loss()
        if not math.isfinite(reference_loss):
            if self.direction_awareness_enabled:
                raise ValueError(
                    "directional Recovery requires finite reference_capability_loss"
                )
            return None
        return classify_capability_evolution(
            alignment=alignment,
            prediction_loss=reference_loss,
            current_loss=current_loss,
            alignment_threshold=self.alignment_threshold,
            degradation_margin=self.recovery_degradation_margin,
            eps=self.degradation_eps,
        )

    def _maybe_rebase(
        self,
        item: VersionedMemoryItem,
        current_sketch: Optional[torch.Tensor],
        current_loss: float,
    ) -> bool:
        if not self.capability_rebase_enabled or current_sketch is None:
            return False
        item.rebase_capability_reference(
            current_sketch, current_loss, dtype=self.storage_dtype
        )
        return True

    def refresh(
        self,
        evaluator: Callable[[int, VersionedMemoryItem], tuple],
        timestamp: int,
        count_recovery_attempts: bool = True,
    ) -> dict[str, int]:
        """Re-evaluate snapshots and apply direction-aware migrations."""

        stats = {
            "stable_to_recovery": 0,
            "recovery_to_stable": 0,
            "recovery_dropped_after_success": 0,
            "recovery_failed": 0,
            "recovery_evicted": 0,
            "recovery_attempt_exhausted": 0,
            "evicted": 0,
            "harmful_drift": 0,
            "beneficial_evolution": 0,
            "recovery_skipped_non_degraded": 0,
            "performance_recovery": 0,
            "capability_rebase": 0,
        }
        stable_snapshots = [
            (expert_id, item)
            for expert_id, buffer in enumerate(self.stable_buffers)
            for item in buffer.items
        ]
        recovery_snapshots = (
            [
                (expert_id, item)
                for expert_id, buffer in enumerate(self.recovery_buffers)
                for item in buffer.items
            ]
            if self.recovery_enabled
            else []
        )

        demotions: List[Tuple[int, VersionedMemoryItem]] = []
        for expert_id, item in stable_snapshots:
            if not self.stable_buffers[expert_id].contains(item.sample_id):
                continue
            alignment, current_loss, current_sketch = self._unpack_evaluation(
                evaluator(expert_id, item)
            )
            decision = self._direction_decision(
                item, alignment, current_loss
            )
            if decision is not None:
                item.capability_evolution = decision.category
                if decision.category == "harmful_drift":
                    stats["harmful_drift"] += 1
                elif decision.category == "beneficial_or_neutral_evolution":
                    stats["beneficial_evolution"] += 1
            effective_alignment = (
                alignment if self.version_awareness_enabled else 1.0
            )
            item.last_observed_alignment = alignment
            item.last_alignment = effective_alignment
            item.recent_prediction_loss = current_loss
            item.age = max(0, int(timestamp) - item.timestamp)
            low_alignment = (
                self.version_awareness_enabled
                and effective_alignment < self.alignment_threshold
            )
            if self.direction_awareness_enabled:
                item.recovery_eligible = bool(
                    self.recovery_enabled
                    and low_alignment
                    and decision is not None
                    and decision.category == "harmful_drift"
                )
            else:
                item.recovery_eligible = self.recovery_enabled
            item.refresh_credit()
            if not low_alignment:
                continue
            if (
                self.direction_awareness_enabled
                and decision is not None
                and decision.category
                == "beneficial_or_neutral_evolution"
            ):
                stats["recovery_skipped_non_degraded"] += 1
                if self._maybe_rebase(item, current_sketch, current_loss):
                    stats["capability_rebase"] += 1
                continue
            if not self.recovery_enabled:
                continue
            demotions.append((expert_id, item))

        # Remove first, then migrate. If Recovery rejects the item, it is
        # evicted rather than incorrectly remaining in Stable.
        for expert_id, item in demotions:
            removed = self.stable_buffers[expert_id].remove(item.sample_id)
            if removed is None:
                continue
            self.recovery_buffers[expert_id].remove(item.sample_id)
            result = self.recovery_buffers[expert_id].add_with_result(removed)
            if result.accepted:
                stats["stable_to_recovery"] += 1
                if result.reason == "replaced":
                    stats["recovery_evicted"] += 1
                    stats["evicted"] += 1
            else:
                stats["evicted"] += 1

        demoted_keys = {
            (expert_id, item.sample_id) for expert_id, item in demotions
        }
        for expert_id, item in recovery_snapshots:
            if (expert_id, item.sample_id) in demoted_keys:
                continue
            if not self.recovery_buffers[expert_id].contains(item.sample_id):
                continue
            alignment, current_loss, current_sketch = self._unpack_evaluation(
                evaluator(expert_id, item)
            )
            decision = self._direction_decision(
                item, alignment, current_loss
            )
            if decision is not None:
                item.capability_evolution = decision.category
                if decision.category == "harmful_drift":
                    stats["harmful_drift"] += 1
                elif decision.category == "beneficial_or_neutral_evolution":
                    stats["beneficial_evolution"] += 1
            effective_alignment = (
                alignment if self.version_awareness_enabled else 1.0
            )
            item.last_observed_alignment = alignment
            item.last_alignment = effective_alignment
            item.recent_prediction_loss = current_loss
            item.age = max(0, int(timestamp) - item.timestamp)
            if count_recovery_attempts:
                item.recovery_attempts += 1

            performance_recovered = bool(
                self.direction_awareness_enabled
                and decision is not None
                and not decision.degraded
            )
            alignment_recovered = bool(
                effective_alignment >= self.promote_alignment_threshold
                and current_loss <= self.promote_loss_threshold
            )
            item.recovery_eligible = not performance_recovered
            item.refresh_credit()
            if performance_recovered or alignment_recovered:
                removed = self.recovery_buffers[expert_id].remove(
                    item.sample_id
                )
                if removed is None:
                    continue
                if performance_recovered:
                    stats["performance_recovery"] += 1
                    if self._maybe_rebase(removed, current_sketch, current_loss):
                        stats["capability_rebase"] += 1
                if self.stable_buffers[expert_id].add(removed):
                    stats["recovery_to_stable"] += 1
                else:
                    stats["recovery_dropped_after_success"] += 1
                    stats["evicted"] += 1
                continue
            if count_recovery_attempts:
                stats["recovery_failed"] += 1
            if (
                count_recovery_attempts
                and item.recovery_attempts >= self.max_recovery_attempts
            ):
                self.recovery_buffers[expert_id].remove(item.sample_id)
                stats["recovery_attempt_exhausted"] += 1
                stats["evicted"] += 1
        return stats

    def sample_recovery(
        self,
        expert_id: int,
        batch_size: int,
        excluded_sample_ids: Optional[Set[int]] = None,
    ) -> Tuple[VersionedMemoryItem, ...]:
        """Select high-priority Recovery items without replacement."""

        if not 0 <= expert_id < self.num_experts:
            raise IndexError("invalid expert_id")
        if not self.recovery_enabled or batch_size <= 0:
            return ()
        excluded = excluded_sample_ids if excluded_sample_ids is not None else set()
        buffer = self.recovery_buffers[expert_id]
        candidates = [
            item for item in buffer.items if item.sample_id not in excluded
        ]
        candidates.sort(
            key=lambda item: item.recovery_score(buffer.failure_penalty),
            reverse=True,
        )
        selected = tuple(candidates[:batch_size])
        excluded.update(item.sample_id for item in selected)
        return selected

    def update_recovery_result(
        self,
        expert_id: int,
        sample_id: int,
        alignment: float,
        prediction_loss: float,
        current_sketch: Optional[torch.Tensor] = None,
    ) -> Literal[
        "recovery",
        "promoted",
        "promoted_performance",
        "dropped",
        "dropped_after_recovery",
        "dropped_after_performance_recovery",
        "recovery_disabled",
        "missing",
    ]:
        """Apply one post-update Recovery result and lifecycle transition."""

        if not 0 <= expert_id < self.num_experts:
            raise IndexError("invalid expert_id")
        if not self.recovery_enabled:
            return "recovery_disabled"
        buffer = self.recovery_buffers[expert_id]
        item = buffer.get(sample_id)
        if item is None:
            return "missing"
        values = torch.tensor(
            [alignment, prediction_loss], dtype=torch.float64
        )
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError("Recovery result contains NaN or Inf")
        current_loss = float(prediction_loss)
        decision = self._direction_decision(
            item, float(alignment), current_loss
        )
        effective_alignment = (
            float(alignment) if self.version_awareness_enabled else 1.0
        )
        item.recovery_attempts += 1
        item.last_observed_alignment = float(alignment)
        item.last_alignment = effective_alignment
        item.recent_prediction_loss = current_loss
        if decision is not None:
            item.capability_evolution = decision.category
        performance_recovered = bool(
            self.direction_awareness_enabled
            and decision is not None
            and not decision.degraded
        )
        alignment_recovered = bool(
            item.last_alignment >= self.promote_alignment_threshold
            and item.recent_prediction_loss <= self.promote_loss_threshold
        )
        item.recovery_eligible = not performance_recovered
        item.refresh_credit()
        if performance_recovered or alignment_recovered:
            removed = buffer.remove(sample_id)
            if removed is None:
                return "missing"
            if performance_recovered:
                self._maybe_rebase(removed, current_sketch, current_loss)
            if self.stable_buffers[expert_id].add(removed):
                return (
                    "promoted_performance"
                    if performance_recovered
                    else "promoted"
                )
            return (
                "dropped_after_performance_recovery"
                if performance_recovered
                else "dropped_after_recovery"
            )
        if item.recovery_attempts >= self.max_recovery_attempts:
            buffer.remove(sample_id)
            return "dropped"
        return "recovery"

    def buffer_sizes(self) -> tuple[list[int], list[int]]:
        """Return Stable and Recovery sizes for every Expert."""

        return (
            [len(buffer) for buffer in self.stable_buffers],
            [len(buffer) for buffer in self.recovery_buffers],
        )

    def all_items(self) -> Iterable[VersionedMemoryItem]:
        for buffers in (self.stable_buffers, self.recovery_buffers):
            for buffer in buffers:
                yield from buffer.items

    def clear(self) -> None:
        for buffers in (self.stable_buffers, self.recovery_buffers):
            for buffer in buffers:
                buffer.clear()
