"""Empirical constrained-comparator evaluation for progressive routing."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ComparatorStatistics:
    """Global sufficient statistics for cumulative mean mixture MSE."""

    gram: np.ndarray
    expert_target_cross: np.ndarray
    target_square_sum: float
    router_cumulative_loss: float
    num_origins: int


@dataclass(frozen=True)
class HorizonChannelStatistics:
    """Per-(h,c) sufficient statistics, scaled to mean over H*C."""

    gram: np.ndarray
    expert_target_cross: np.ndarray
    target_square_sum: np.ndarray
    router_cumulative_loss: float
    num_origins: int


@dataclass(frozen=True)
class ComparatorResult:
    """Result for the Global Static Convex Comparator."""

    comparator_type: str
    router_cumulative_loss: float
    comparator_loss: float
    regret: float
    average_regret: float
    num_origins: int
    comparator_weights: np.ndarray
    converged: bool
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        weights = self.comparator_weights.tolist()
        return {
            "comparator_type": self.comparator_type,
            "router_cumulative_loss": self.router_cumulative_loss,
            "comparator_loss": self.comparator_loss,
            "regret": self.regret,
            "average_regret": self.average_regret,
            "num_origins": self.num_origins,
            "comparator_weights": weights,
            "converged": self.converged,
            "iterations": self.iterations,
            # Backward-compatible fields.
            "static_comparator_loss": self.comparator_loss,
            "static_regret": self.regret,
            "average_static_regret": self.average_regret,
            "static_comparator_weights": weights,
            # Explicit paper terminology.
            "global_static_comparator_loss": self.comparator_loss,
            "global_static_regret": self.regret,
            "global_average_static_regret": self.average_regret,
            "global_static_comparator_weights": weights,
        }


@dataclass(frozen=True)
class HorizonChannelComparatorResult:
    comparator_weights: np.ndarray
    comparator_loss: float
    regret: float
    average_regret: float
    router_cumulative_loss: float
    num_origins: int
    converged: bool
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "hc_comparator_type": "horizon_channel_static_convex_comparator",
            "hc_static_comparator_loss": self.comparator_loss,
            "hc_static_regret": self.regret,
            "hc_average_static_regret": self.average_regret,
            "hc_static_comparator_weights": self.comparator_weights.tolist(),
            "hc_static_converged": self.converged,
            "hc_static_iterations": self.iterations,
        }


@dataclass(frozen=True)
class KSwitchComparatorResult:
    comparator_loss: float
    regret: float
    average_regret: float
    router_cumulative_loss: float
    num_origins: int
    max_switches: int
    num_comparator_points: int
    switch_points: tuple[int, ...]
    segment_weights: np.ndarray
    converged: bool
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dynamic_comparator_type": "empirical_k_switch_convex_comparator",
            "dynamic_comparator_loss": self.comparator_loss,
            "dynamic_regret": self.regret,
            "average_dynamic_regret": self.average_regret,
            "dynamic_max_switches": self.max_switches,
            "dynamic_num_comparator_points": self.num_comparator_points,
            "switch_points": list(self.switch_points),
            "segment_weights": self.segment_weights.tolist(),
            "dynamic_converged": self.converged,
            "dynamic_iterations": self.iterations,
        }


@dataclass(frozen=True)
class ComparatorBlock:
    start_origin: int
    end_origin: int
    statistics: ComparatorStatistics


def _validated_origin_arrays(
    expert_predictions: torch.Tensor,
    router_prediction: torch.Tensor,
    target: torch.Tensor,
    expected_experts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if expert_predictions.ndim != 3:
        raise ValueError("expert_predictions must have shape [H,C,E]")
    horizon, channels, experts = expert_predictions.shape
    if experts != expected_experts:
        raise ValueError("Expert dimension does not match accumulator")
    if tuple(router_prediction.shape) != (horizon, channels):
        raise ValueError("router_prediction must have shape [H,C]")
    if tuple(target.shape) != (horizon, channels):
        raise ValueError("target must have shape [H,C]")
    predictions = expert_predictions.detach().double().cpu().numpy()
    router = router_prediction.detach().double().cpu().numpy()
    target_value = target.detach().double().cpu().numpy()
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Expert predictions contain NaN or Inf")
    if not np.isfinite(router).all():
        raise FloatingPointError("Router prediction contains NaN or Inf")
    if not np.isfinite(target_value).all():
        raise FloatingPointError("evaluation target contains NaN or Inf")
    return predictions, router, target_value


def _origin_global_statistics(
    predictions: np.ndarray,
    router: np.ndarray,
    target: np.ndarray,
) -> ComparatorStatistics:
    experts = predictions.shape[-1]
    flat_predictions = predictions.reshape(-1, experts)
    flat_router = router.reshape(-1)
    flat_target = target.reshape(-1)
    scale = 1.0 / float(flat_target.size)
    return ComparatorStatistics(
        gram=scale * (flat_predictions.T @ flat_predictions),
        expert_target_cross=scale * (flat_predictions.T @ flat_target),
        target_square_sum=scale * float(flat_target @ flat_target),
        router_cumulative_loss=float(np.mean((flat_router - flat_target) ** 2)),
        num_origins=1,
    )


def _combine_statistics(
    first: ComparatorStatistics,
    second: ComparatorStatistics,
) -> ComparatorStatistics:
    return ComparatorStatistics(
        gram=first.gram + second.gram,
        expert_target_cross=(
            first.expert_target_cross + second.expert_target_cross
        ),
        target_square_sum=first.target_square_sum + second.target_square_sum,
        router_cumulative_loss=(
            first.router_cumulative_loss + second.router_cumulative_loss
        ),
        num_origins=first.num_origins + second.num_origins,
    )


class StaticComparatorAccumulator:
    """Stream global O(E^2) and structured O(H*C*E^2) statistics."""

    def __init__(
        self,
        num_experts: int,
        horizon: int | None = None,
        channels: int | None = None,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if (horizon is None) != (channels is None):
            raise ValueError("horizon and channels must be provided together")
        if horizon is not None and (horizon <= 0 or channels <= 0):
            raise ValueError("horizon and channels must be positive")
        self.num_experts = int(num_experts)
        self._configured_shape = (
            None if horizon is None else (int(horizon), int(channels))
        )
        self.reset()

    def _initialize_hc(self, horizon: int, channels: int) -> None:
        if self.hc_gram is not None:
            if self.hc_gram.shape[:2] != (horizon, channels):
                raise ValueError("Horizon/channel shape changed within one stream")
            return
        experts = self.num_experts
        self.hc_gram = np.zeros(
            (horizon, channels, experts, experts), dtype=np.float64
        )
        self.hc_cross = np.zeros(
            (horizon, channels, experts), dtype=np.float64
        )
        self.hc_target_square_sum = np.zeros(
            (horizon, channels), dtype=np.float64
        )

    def reset(self) -> None:
        self.gram = np.zeros(
            (self.num_experts, self.num_experts), dtype=np.float64
        )
        self.expert_target_cross = np.zeros(
            self.num_experts, dtype=np.float64
        )
        self.target_square_sum = 0.0
        self.router_cumulative_loss = 0.0
        self.num_origins = 0
        self.hc_gram: np.ndarray | None = None
        self.hc_cross: np.ndarray | None = None
        self.hc_target_square_sum: np.ndarray | None = None
        if self._configured_shape is not None:
            self._initialize_hc(*self._configured_shape)

    @property
    def state_nbytes(self) -> int:
        arrays = [self.gram, self.expert_target_cross]
        arrays.extend(
            array
            for array in (
                self.hc_gram,
                self.hc_cross,
                self.hc_target_square_sum,
            )
            if array is not None
        )
        return int(sum(array.nbytes for array in arrays))

    @torch.no_grad()
    def update(
        self,
        expert_predictions: torch.Tensor,
        router_prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        predictions, router, target_value = _validated_origin_arrays(
            expert_predictions,
            router_prediction,
            target,
            self.num_experts,
        )
        horizon, channels, _ = predictions.shape
        self._initialize_hc(horizon, channels)
        origin = _origin_global_statistics(predictions, router, target_value)
        self.gram += origin.gram
        self.expert_target_cross += origin.expert_target_cross
        self.target_square_sum += origin.target_square_sum
        self.router_cumulative_loss += origin.router_cumulative_loss
        self.num_origins += 1

        scale = 1.0 / float(horizon * channels)
        self.hc_gram += scale * np.einsum(
            "hce,hcf->hcef", predictions, predictions
        )
        self.hc_cross += scale * predictions * target_value[..., None]
        self.hc_target_square_sum += scale * target_value**2

    def statistics(self) -> ComparatorStatistics:
        return ComparatorStatistics(
            gram=self.gram.copy(),
            expert_target_cross=self.expert_target_cross.copy(),
            target_square_sum=float(self.target_square_sum),
            router_cumulative_loss=float(self.router_cumulative_loss),
            num_origins=int(self.num_origins),
        )

    def horizon_channel_statistics(self) -> HorizonChannelStatistics | None:
        if self.hc_gram is None:
            return None
        return HorizonChannelStatistics(
            gram=self.hc_gram.copy(),
            expert_target_cross=self.hc_cross.copy(),
            target_square_sum=self.hc_target_square_sum.copy(),
            router_cumulative_loss=float(self.router_cumulative_loss),
            num_origins=int(self.num_origins),
        )


class KSwitchComparatorAccumulator:
    """Retain at most max_points ordered blocks for offline K-switch DP."""

    def __init__(self, num_experts: int, max_points: int = 64) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        self.num_experts = int(num_experts)
        self.max_points = int(max_points)
        self.reset()

    def reset(self) -> None:
        self._blocks: list[ComparatorBlock] = []
        self.num_origins = 0

    @property
    def blocks(self) -> tuple[ComparatorBlock, ...]:
        return tuple(self._blocks)

    @property
    def state_nbytes(self) -> int:
        return int(
            sum(
                block.statistics.gram.nbytes
                + block.statistics.expert_target_cross.nbytes
                for block in self._blocks
            )
        )

    @torch.no_grad()
    def update(
        self,
        expert_predictions: torch.Tensor,
        router_prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        predictions, router, target_value = _validated_origin_arrays(
            expert_predictions,
            router_prediction,
            target,
            self.num_experts,
        )
        origin = self.num_origins
        statistics = _origin_global_statistics(predictions, router, target_value)
        self._blocks.append(ComparatorBlock(origin, origin + 1, statistics))
        self.num_origins += 1
        self._compress()

    def _compress(self) -> None:
        while len(self._blocks) > self.max_points:
            pair_sizes = [
                first.statistics.num_origins + second.statistics.num_origins
                for first, second in zip(self._blocks, self._blocks[1:])
            ]
            index = int(np.argmin(pair_sizes))
            first, second = self._blocks[index : index + 2]
            merged = ComparatorBlock(
                start_origin=first.start_origin,
                end_origin=second.end_origin,
                statistics=_combine_statistics(
                    first.statistics, second.statistics
                ),
            )
            self._blocks[index : index + 2] = [merged]

    def statistics(self) -> ComparatorStatistics:
        if not self._blocks:
            return ComparatorStatistics(
                gram=np.zeros(
                    (self.num_experts, self.num_experts), dtype=np.float64
                ),
                expert_target_cross=np.zeros(
                    self.num_experts, dtype=np.float64
                ),
                target_square_sum=0.0,
                router_cumulative_loss=0.0,
                num_origins=0,
            )
        combined = self._blocks[0].statistics
        for block in self._blocks[1:]:
            combined = _combine_statistics(combined, block.statistics)
        return combined


def _project_simplex_rows(values: np.ndarray) -> np.ndarray:
    """Project vectors in the final dimension onto the probability simplex."""

    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1, original_shape[-1])
    sorted_values = np.sort(flat, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_values, axis=1)
    indices = np.arange(1, flat.shape[1] + 1, dtype=np.float64)
    positive = sorted_values - (cumulative - 1.0) / indices > 0.0
    rho = np.maximum(positive.sum(axis=1) - 1, 0)
    threshold = (
        cumulative[np.arange(flat.shape[0]), rho] - 1.0
    ) / (rho + 1.0)
    projected = np.maximum(flat - threshold[:, None], 0.0)
    projected /= projected.sum(axis=1, keepdims=True)
    return projected.reshape(original_shape)


def _solve_simplex_quadratics(
    gram: np.ndarray,
    cross: np.ndarray,
    target_square: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """Solve a batch of small simplex quadratics with projected gradient."""

    grams = np.asarray(gram, dtype=np.float64)
    crosses = np.asarray(cross, dtype=np.float64)
    targets = np.asarray(target_square, dtype=np.float64)
    if grams.ndim != 3 or grams.shape[-1] != grams.shape[-2]:
        raise ValueError("gram batch must have shape [N,E,E]")
    count, experts, _ = grams.shape
    if crosses.shape != (count, experts) or targets.shape != (count,):
        raise ValueError("quadratic batch shapes do not match")
    if count == 0:
        return (
            np.empty((0, experts), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            True,
            0,
        )
    weights = np.full((count, experts), 1.0 / experts, dtype=np.float64)
    uniform = weights.copy()
    eigenvalues = np.linalg.eigvalsh((grams + grams.transpose(0, 2, 1)) / 2.0)
    step = 1.0 / np.maximum(2.0 * np.maximum(eigenvalues[:, -1], 0.0), eps)
    converged = False
    iterations = max_iterations
    for iteration in range(1, max_iterations + 1):
        gradient = 2.0 * (
            np.einsum("nij,nj->ni", grams, weights) - crosses
        )
        updated = _project_simplex_rows(weights - step[:, None] * gradient)
        if float(np.max(np.linalg.norm(updated - weights, axis=1))) <= tolerance:
            weights = updated
            converged = True
            iterations = iteration
            break
        weights = updated

    def losses(candidate: np.ndarray) -> np.ndarray:
        return (
            np.einsum("ni,nij,nj->n", candidate, grams, candidate)
            - 2.0 * np.einsum("ni,ni->n", crosses, candidate)
            + targets
        )

    final_loss = np.maximum(losses(weights), 0.0)
    uniform_loss = np.maximum(losses(uniform), 0.0)
    worse = final_loss > uniform_loss + 1e-10
    weights[worse] = uniform[worse]
    final_loss[worse] = uniform_loss[worse]
    return weights, final_loss, converged, iterations


def _validated_statistics(
    source: ComparatorStatistics | StaticComparatorAccumulator,
) -> ComparatorStatistics:
    statistics = source.statistics() if hasattr(source, "statistics") else source
    gram = np.asarray(statistics.gram, dtype=np.float64)
    cross = np.asarray(statistics.expert_target_cross, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must have shape [E,E]")
    if cross.shape != (gram.shape[0],):
        raise ValueError("expert_target_cross must have shape [E]")
    if not np.isfinite(gram).all() or not np.isfinite(cross).all():
        raise FloatingPointError("comparator statistics contain NaN or Inf")
    scalar = np.asarray(
        [statistics.target_square_sum, statistics.router_cumulative_loss]
    )
    if not np.isfinite(scalar).all():
        raise FloatingPointError("comparator scalar statistics contain NaN or Inf")
    if statistics.num_origins < 0:
        raise ValueError("num_origins must be non-negative")
    return ComparatorStatistics(
        gram=(gram + gram.T) / 2.0,
        expert_target_cross=cross,
        target_square_sum=float(statistics.target_square_sum),
        router_cumulative_loss=float(statistics.router_cumulative_loss),
        num_origins=int(statistics.num_origins),
    )


def fixed_mixture_cumulative_loss(
    source: ComparatorStatistics | StaticComparatorAccumulator,
    weights: np.ndarray,
) -> float:
    statistics = _validated_statistics(source)
    mixture = np.asarray(weights, dtype=np.float64)
    experts = statistics.gram.shape[0]
    if mixture.shape != (experts,):
        raise ValueError(f"weights must have shape {(experts,)}")
    if not np.isfinite(mixture).all():
        raise FloatingPointError("mixture weights contain NaN or Inf")
    if np.any(mixture < -1e-12) or not np.isclose(
        mixture.sum(), 1.0, atol=1e-10
    ):
        raise ValueError("weights must belong to the probability simplex")
    loss = float(
        mixture @ statistics.gram @ mixture
        - 2.0 * statistics.expert_target_cross @ mixture
        + statistics.target_square_sum
    )
    return max(loss, 0.0)


def evaluate_static_comparator(
    source: ComparatorStatistics | StaticComparatorAccumulator,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    eps: float = 1e-12,
) -> ComparatorResult:
    """Evaluate the Global Static Convex Comparator."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    statistics = _validated_statistics(source)
    weights, losses, converged, iterations = _solve_simplex_quadratics(
        statistics.gram[None, ...],
        statistics.expert_target_cross[None, ...],
        np.asarray([statistics.target_square_sum]),
        max_iterations=max_iterations,
        tolerance=tolerance,
        eps=eps,
    )
    comparator_loss = float(losses[0])
    router_loss = statistics.router_cumulative_loss
    regret = router_loss - comparator_loss
    average = regret / statistics.num_origins if statistics.num_origins else 0.0
    return ComparatorResult(
        comparator_type="global_static_convex_comparator",
        router_cumulative_loss=router_loss,
        comparator_loss=comparator_loss,
        regret=regret,
        average_regret=average,
        num_origins=statistics.num_origins,
        comparator_weights=weights[0],
        converged=converged,
        iterations=iterations,
    )


def evaluate_horizon_channel_static_comparator(
    source: HorizonChannelStatistics | StaticComparatorAccumulator,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    eps: float = 1e-12,
) -> HorizonChannelComparatorResult:
    """Fit one across-time fixed simplex mixture for every (h,c)."""

    statistics = (
        source.horizon_channel_statistics()
        if isinstance(source, StaticComparatorAccumulator)
        else source
    )
    if statistics is None:
        raise ValueError("horizon/channel shape is unavailable before any update")
    gram = np.asarray(statistics.gram, dtype=np.float64)
    cross = np.asarray(statistics.expert_target_cross, dtype=np.float64)
    target_square = np.asarray(statistics.target_square_sum, dtype=np.float64)
    if gram.ndim != 4 or gram.shape[-1] != gram.shape[-2]:
        raise ValueError("HC gram must have shape [H,C,E,E]")
    horizon, channels, experts, _ = gram.shape
    if cross.shape != (horizon, channels, experts):
        raise ValueError("HC cross must have shape [H,C,E]")
    if target_square.shape != (horizon, channels):
        raise ValueError("HC target squares must have shape [H,C]")
    weights, losses, converged, iterations = _solve_simplex_quadratics(
        gram.reshape(-1, experts, experts),
        cross.reshape(-1, experts),
        target_square.reshape(-1),
        max_iterations=max_iterations,
        tolerance=tolerance,
        eps=eps,
    )
    comparator_loss = float(losses.sum())
    router_loss = float(statistics.router_cumulative_loss)
    regret = router_loss - comparator_loss
    average = regret / statistics.num_origins if statistics.num_origins else 0.0
    return HorizonChannelComparatorResult(
        comparator_weights=weights.reshape(horizon, channels, experts),
        comparator_loss=comparator_loss,
        regret=regret,
        average_regret=average,
        router_cumulative_loss=router_loss,
        num_origins=int(statistics.num_origins),
        converged=converged,
        iterations=iterations,
    )


def evaluate_k_switch_dynamic_comparator(
    source: KSwitchComparatorAccumulator,
    max_switches: int,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    eps: float = 1e-12,
) -> KSwitchComparatorResult:
    """Offline DP for a global piecewise-constant mixture with <=K switches."""

    if not isinstance(source, KSwitchComparatorAccumulator):
        raise TypeError("K-switch evaluation requires KSwitchComparatorAccumulator")
    if max_switches < 0:
        raise ValueError("max_switches must be non-negative")
    blocks = source.blocks
    experts = source.num_experts
    if not blocks:
        return KSwitchComparatorResult(
            comparator_loss=0.0,
            regret=0.0,
            average_regret=0.0,
            router_cumulative_loss=0.0,
            num_origins=0,
            max_switches=int(max_switches),
            num_comparator_points=0,
            switch_points=(),
            segment_weights=np.empty((0, experts), dtype=np.float64),
            converged=True,
            iterations=0,
        )

    count = len(blocks)
    cumulative_gram = np.zeros((count + 1, experts, experts), dtype=np.float64)
    cumulative_cross = np.zeros((count + 1, experts), dtype=np.float64)
    cumulative_target = np.zeros(count + 1, dtype=np.float64)
    for index, block in enumerate(blocks, start=1):
        cumulative_gram[index] = (
            cumulative_gram[index - 1] + block.statistics.gram
        )
        cumulative_cross[index] = (
            cumulative_cross[index - 1]
            + block.statistics.expert_target_cross
        )
        cumulative_target[index] = (
            cumulative_target[index - 1]
            + block.statistics.target_square_sum
        )

    segment_pairs: list[tuple[int, int]] = []
    segment_grams = []
    segment_crosses = []
    segment_targets = []
    lookup = np.full((count, count), -1, dtype=np.int64)
    for start in range(count):
        for end in range(start, count):
            lookup[start, end] = len(segment_pairs)
            segment_pairs.append((start, end))
            segment_grams.append(
                cumulative_gram[end + 1] - cumulative_gram[start]
            )
            segment_crosses.append(
                cumulative_cross[end + 1] - cumulative_cross[start]
            )
            segment_targets.append(
                cumulative_target[end + 1] - cumulative_target[start]
            )
    segment_weights, segment_losses, converged, iterations = (
        _solve_simplex_quadratics(
            np.asarray(segment_grams),
            np.asarray(segment_crosses),
            np.asarray(segment_targets),
            max_iterations=max_iterations,
            tolerance=tolerance,
            eps=eps,
        )
    )

    max_segments = min(count, int(max_switches) + 1)
    dynamic = np.full((max_segments + 1, count + 1), np.inf)
    previous = np.full((max_segments + 1, count + 1), -1, dtype=np.int64)
    dynamic[0, 0] = 0.0
    for segments in range(1, max_segments + 1):
        for end_exclusive in range(segments, count + 1):
            best_value = np.inf
            best_start = -1
            for start in range(segments - 1, end_exclusive):
                candidate = dynamic[segments - 1, start] + segment_losses[
                    lookup[start, end_exclusive - 1]
                ]
                if candidate < best_value:
                    best_value = candidate
                    best_start = start
            dynamic[segments, end_exclusive] = best_value
            previous[segments, end_exclusive] = best_start

    segment_count = int(np.argmin(dynamic[1:, count]) + 1)
    comparator_loss = float(dynamic[segment_count, count])
    boundaries: list[tuple[int, int]] = []
    end_exclusive = count
    for segments in range(segment_count, 0, -1):
        start = int(previous[segments, end_exclusive])
        boundaries.append((start, end_exclusive - 1))
        end_exclusive = start
    boundaries.reverse()
    selected_weights = np.stack(
        [segment_weights[lookup[start, end]] for start, end in boundaries]
    )
    switch_points = tuple(
        blocks[start].start_origin for start, _ in boundaries[1:]
    )
    totals = source.statistics()
    router_loss = totals.router_cumulative_loss
    regret = router_loss - comparator_loss
    average = regret / totals.num_origins if totals.num_origins else 0.0
    return KSwitchComparatorResult(
        comparator_loss=comparator_loss,
        regret=regret,
        average_regret=average,
        router_cumulative_loss=router_loss,
        num_origins=totals.num_origins,
        max_switches=int(max_switches),
        num_comparator_points=count,
        switch_points=switch_points,
        segment_weights=selected_weights,
        converged=converged,
        iterations=iterations,
    )


def evaluate_dynamic_comparator(
    source: Any,
    comparator_class: str | None = None,
    *,
    max_switches: int = 0,
) -> KSwitchComparatorResult:
    """Dispatch only the implemented empirical K-switch comparator."""

    if comparator_class not in {"k_switch", "k-switch"}:
        raise NotImplementedError(
            "Specify comparator class 'k_switch' to evaluate the empirical "
            "K-switch comparator; path-length evaluation remains a TODO"
        )
    return evaluate_k_switch_dynamic_comparator(source, max_switches)


def evaluate_path_length_comparator(*args: Any, **kwargs: Any) -> None:
    """Reserved path-length interface; no approximate value is fabricated."""

    del args, kwargs
    raise NotImplementedError(
        "TODO: implement and validate a reliable path-length comparator solver"
    )


def save_comparator_diagnostics(
    result: ComparatorResult,
    directory: str,
    hc_result: HorizonChannelComparatorResult | None = None,
    dynamic_result: KSwitchComparatorResult | None = None,
) -> tuple[str, str]:
    os.makedirs(directory, exist_ok=True)
    values = result.as_dict()
    arrays: dict[str, np.ndarray] = {
        "comparator_type": np.asarray(result.comparator_type),
        "router_cumulative_loss": np.asarray(result.router_cumulative_loss),
        "static_comparator_loss": np.asarray(result.comparator_loss),
        "static_regret": np.asarray(result.regret),
        "average_static_regret": np.asarray(result.average_regret),
        "num_origins": np.asarray(result.num_origins, dtype=np.int64),
        "static_comparator_weights": np.asarray(
            result.comparator_weights, dtype=np.float64
        ),
        "global_static_comparator_loss": np.asarray(result.comparator_loss),
        "global_static_regret": np.asarray(result.regret),
        "global_average_static_regret": np.asarray(result.average_regret),
        "global_static_comparator_weights": np.asarray(
            result.comparator_weights, dtype=np.float64
        ),
        "converged": np.asarray(result.converged),
        "iterations": np.asarray(result.iterations, dtype=np.int64),
    }
    if hc_result is not None:
        values.update(hc_result.as_dict())
        arrays.update(
            {
                "hc_static_comparator_loss": np.asarray(
                    hc_result.comparator_loss
                ),
                "hc_static_regret": np.asarray(hc_result.regret),
                "hc_average_static_regret": np.asarray(
                    hc_result.average_regret
                ),
                "hc_static_comparator_weights": np.asarray(
                    hc_result.comparator_weights, dtype=np.float64
                ),
                "hc_static_converged": np.asarray(hc_result.converged),
                "hc_static_iterations": np.asarray(
                    hc_result.iterations, dtype=np.int64
                ),
            }
        )
    if dynamic_result is not None:
        values.update(dynamic_result.as_dict())
        arrays.update(
            {
                "dynamic_comparator_loss": np.asarray(
                    dynamic_result.comparator_loss
                ),
                "dynamic_regret": np.asarray(dynamic_result.regret),
                "average_dynamic_regret": np.asarray(
                    dynamic_result.average_regret
                ),
                "dynamic_max_switches": np.asarray(
                    dynamic_result.max_switches, dtype=np.int64
                ),
                "dynamic_num_comparator_points": np.asarray(
                    dynamic_result.num_comparator_points, dtype=np.int64
                ),
                "switch_points": np.asarray(
                    dynamic_result.switch_points, dtype=np.int64
                ),
                "segment_weights": np.asarray(
                    dynamic_result.segment_weights, dtype=np.float64
                ),
            }
        )
    npz_path = os.path.join(directory, "comparator_diagnostics.npz")
    json_path = os.path.join(directory, "comparator_diagnostics.json")
    np.savez_compressed(npz_path, **arrays)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2)
    return npz_path, json_path
