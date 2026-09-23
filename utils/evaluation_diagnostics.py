"""Evaluation-only oracle and streaming specialization diagnostics."""

from __future__ import annotations

import itertools
import math
import os
from typing import Any

import numpy as np
import torch


def _project_simplex(vector: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of one vector onto the probability simplex."""

    sorted_values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0)
    indices = torch.arange(
        1, vector.numel() + 1, dtype=vector.dtype, device=vector.device
    )
    candidates = sorted_values - (cumulative - 1.0) / indices
    positive = torch.nonzero(candidates > 0, as_tuple=False).flatten()
    if positive.numel() == 0:
        return torch.full_like(vector, 1.0 / vector.numel())
    rho = int(positive[-1].item())
    threshold = (cumulative[rho] - 1.0) / float(rho + 1)
    projected = torch.clamp(vector - threshold, min=0.0)
    return projected / projected.sum().clamp_min(
        torch.finfo(projected.dtype).eps
    )


def _all_expert_convex_oracle(
    predictions: torch.Tensor,
    target: torch.Tensor,
    initial_weights: torch.Tensor,
    eps: float,
    max_iterations: int = 5000,
    tolerance: float = 1e-12,
) -> tuple[float, torch.Tensor]:
    """Solve one hindsight simplex least-squares problem without autograd."""

    experts = predictions.shape[-1]
    design = predictions.reshape(-1, experts).double()
    target_vector = target.reshape(-1).double()
    scale = 1.0 / float(target_vector.numel())
    gram = scale * (design.T @ design)
    cross = scale * (design.T @ target_vector)
    weights = initial_weights.detach().double().clone()
    initial_prediction = design @ weights
    initial_mse = float(
        (initial_prediction - target_vector).pow(2).mean().item()
    )
    largest = float(torch.linalg.eigvalsh(gram).max().clamp_min(0.0).item())
    if largest > eps:
        step_size = 1.0 / max(2.0 * largest, eps)
        for _ in range(max_iterations):
            gradient = 2.0 * (gram @ weights - cross)
            updated = _project_simplex(weights - step_size * gradient)
            if (
                float(torch.linalg.vector_norm(updated - weights).item())
                <= tolerance
            ):
                weights = updated
                break
            weights = updated
    prediction = design @ weights
    mse = float((prediction - target_vector).pow(2).mean().item())
    if mse > initial_mse + 1e-10:
        return initial_mse, initial_weights.detach().double().clone()
    return mse, weights


@torch.no_grad()
def compute_router_oracle_diagnostics(
    expert_predictions: torch.Tensor,
    mixture_prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Evaluate historical predictions against a complete target.

    This helper is diagnostic-only: every input is detached and no returned
    value participates in model updates.
    """

    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if expert_predictions.ndim != 3:
        raise ValueError("expert_predictions must have shape [H,C,E]")
    horizon, channels, experts = expert_predictions.shape
    if experts <= 0:
        raise ValueError("at least one Expert is required")
    if tuple(mixture_prediction.shape) != (horizon, channels):
        raise ValueError("mixture_prediction must have shape [H,C]")
    if tuple(target.shape) != (horizon, channels):
        raise ValueError("target must have shape [H,C]")

    predictions = expert_predictions.detach()
    mixture = mixture_prediction.detach().to(predictions)
    target_value = target.detach().to(predictions)
    if not bool(torch.isfinite(predictions).all().item()):
        raise FloatingPointError("Expert predictions contain NaN or Inf")
    if not bool(torch.isfinite(mixture).all().item()):
        raise FloatingPointError("mixture prediction contains NaN or Inf")
    if not bool(torch.isfinite(target_value).all().item()):
        raise FloatingPointError("evaluation target contains NaN or Inf")

    expert_mse = (predictions - target_value.unsqueeze(-1)).pow(2).mean(
        dim=(0, 1)
    )
    hard_expert_id = int(expert_mse.argmin().item())
    hard_mse = float(expert_mse[hard_expert_id].item())

    if experts == 1:
        top2_pair = (0, 0)
        top2_alpha = 1.0
        top2_mse = hard_mse
    else:
        if hard_expert_id == 0:
            top2_pair = (0, 1)
            top2_alpha = 1.0
        else:
            top2_pair = (0, hard_expert_id)
            top2_alpha = 0.0
        top2_mse = hard_mse
        for first, second in itertools.combinations(range(experts), 2):
            prediction_first = predictions[..., first]
            prediction_second = predictions[..., second]
            direction = prediction_first - prediction_second
            numerator = torch.sum(
                (target_value - prediction_second) * direction
            )
            denominator = torch.sum(direction * direction) + eps
            alpha = torch.clamp(numerator / denominator, 0.0, 1.0)
            convex_prediction = (
                alpha * prediction_first
                + (1.0 - alpha) * prediction_second
            )
            mse = float(
                (convex_prediction - target_value).pow(2).mean().item()
            )
            if mse < top2_mse:
                top2_pair = (first, second)
                top2_alpha = float(alpha.item())
                top2_mse = mse
    top2_weights = torch.zeros(
        experts, dtype=predictions.dtype, device=predictions.device
    )
    top2_weights[top2_pair[0]] += top2_alpha
    top2_weights[top2_pair[1]] += 1.0 - top2_alpha
    all_expert_mse, all_expert_weights = _all_expert_convex_oracle(
        predictions=predictions,
        target=target_value,
        initial_weights=top2_weights,
        eps=eps,
    )


    router_mse = float((mixture - target_value).pow(2).mean().item())
    return {
        "expert_mse": expert_mse.detach().cpu(),
        "oracle_hard_expert_id": hard_expert_id,
        "oracle_hard_mse": hard_mse,
        "oracle_top2_mse": top2_mse,
        "oracle_top2_pair": top2_pair,
        "oracle_top2_alpha": top2_alpha,
        "oracle_all_expert_mse": all_expert_mse,
        "oracle_all_expert_weights": all_expert_weights.detach().cpu(),
        "router_mse": router_mse,
        "gap_to_hard_oracle": router_mse - hard_mse,
        "gap_to_top2_oracle": router_mse - top2_mse,
        "gap_to_all_expert_oracle": router_mse - all_expert_mse,
    }


class SpecializationDiagnosticsAggregator:
    """Fixed-size streaming horizon/channel specialization statistics."""

    def __init__(self, pred_len: int, c_out: int, num_experts: int) -> None:
        if pred_len <= 0 or c_out <= 0 or num_experts <= 0:
            raise ValueError("pred_len, c_out, and num_experts must be positive")
        self.pred_len = int(pred_len)
        self.c_out = int(c_out)
        self.num_experts = int(num_experts)
        self.reset()

    def reset(self) -> None:
        horizon_shape = (self.pred_len, self.num_experts)
        channel_shape = (self.c_out, self.num_experts)
        self.router_weight_horizon_sum = np.zeros(horizon_shape, dtype=np.float64)
        self.router_weight_channel_sum = np.zeros(channel_shape, dtype=np.float64)
        self.expert_error_horizon_sum = np.zeros(horizon_shape, dtype=np.float64)
        self.expert_error_channel_sum = np.zeros(channel_shape, dtype=np.float64)
        self.winner_horizon_sum = np.zeros(horizon_shape, dtype=np.float64)
        self.winner_channel_sum = np.zeros(channel_shape, dtype=np.float64)
        self.horizon_count = np.zeros(self.pred_len, dtype=np.int64)
        self.channel_count = np.zeros(self.c_out, dtype=np.int64)
        self.num_updates = 0

    @property
    def state_nbytes(self) -> int:
        return int(
            sum(
                value.nbytes
                for value in self.__dict__.values()
                if isinstance(value, np.ndarray)
            )
        )

    @torch.no_grad()
    def update(
        self,
        router_weights: torch.Tensor,
        expert_predictions: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        expected_hce = (self.pred_len, self.c_out, self.num_experts)
        if tuple(router_weights.shape) != expected_hce:
            raise ValueError(
                f"router_weights must have shape {expected_hce}, got "
                f"{tuple(router_weights.shape)}"
            )
        if tuple(expert_predictions.shape) != expected_hce:
            raise ValueError(
                f"expert_predictions must have shape {expected_hce}, got "
                f"{tuple(expert_predictions.shape)}"
            )
        if tuple(target.shape) != (self.pred_len, self.c_out):
            raise ValueError(
                "target must have shape "
                f"{(self.pred_len, self.c_out)}, got {tuple(target.shape)}"
            )

        weights = router_weights.detach().double().cpu().numpy()
        predictions = expert_predictions.detach().double().cpu().numpy()
        target_value = target.detach().double().cpu().numpy()
        if not np.isfinite(weights).all():
            raise FloatingPointError("router weights contain NaN or Inf")
        if not np.isfinite(predictions).all():
            raise FloatingPointError("Expert predictions contain NaN or Inf")
        if not np.isfinite(target_value).all():
            raise FloatingPointError("evaluation target contains NaN or Inf")

        squared_error = (predictions - target_value[..., None]) ** 2
        winners = squared_error.argmin(axis=-1)
        self.router_weight_horizon_sum += weights.sum(axis=1)
        self.router_weight_channel_sum += weights.sum(axis=0)
        self.expert_error_horizon_sum += squared_error.sum(axis=1)
        self.expert_error_channel_sum += squared_error.sum(axis=0)
        for horizon in range(self.pred_len):
            self.winner_horizon_sum[horizon] += np.bincount(
                winners[horizon], minlength=self.num_experts
            )
        for channel in range(self.c_out):
            self.winner_channel_sum[channel] += np.bincount(
                winners[:, channel], minlength=self.num_experts
            )
        self.horizon_count += self.c_out
        self.channel_count += self.pred_len
        self.num_updates += 1

    @staticmethod
    def _mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
        result = np.zeros_like(total, dtype=np.float64)
        np.divide(total, count[:, None], out=result, where=count[:, None] > 0)
        return result

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "mean_router_weight_by_horizon": self._mean(
                self.router_weight_horizon_sum, self.horizon_count
            ),
            "mean_router_weight_by_channel": self._mean(
                self.router_weight_channel_sum, self.channel_count
            ),
            "expert_mse_by_horizon": self._mean(
                self.expert_error_horizon_sum, self.horizon_count
            ),
            "expert_mse_by_channel": self._mean(
                self.expert_error_channel_sum, self.channel_count
            ),
            "winning_expert_rate_by_horizon": self._mean(
                self.winner_horizon_sum, self.horizon_count
            ),
            "winning_expert_rate_by_channel": self._mean(
                self.winner_channel_sum, self.channel_count
            ),
            "horizon_count": self.horizon_count.copy(),
            "channel_count": self.channel_count.copy(),
            "num_updates": np.asarray(self.num_updates, dtype=np.int64),
        }

    def save(self, path: str) -> str:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        return path
