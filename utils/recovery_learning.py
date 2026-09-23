"""Differentiable objectives used by Recovery replay.

The historical capability sketch is always treated as a prediction-time
snapshot.  It is detached inside the objective so callers cannot
accidentally backpropagate into stored memory.
"""

from __future__ import annotations

import torch


def recovery_replay_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current_sketch: torch.Tensor,
    prediction_time_sketch: torch.Tensor,
    responsibility: torch.Tensor,
    sketch_weight: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a responsibility-weighted Recovery replay loss.

    Args:
        prediction: Current Expert prediction, shape ``[N, D]``.
        target: Complete historical target, shape ``[N, D]``.
        current_sketch: Current differentiable sketch, shape ``[N, S]``.
        prediction_time_sketch: Stored sketch, shape ``[N, S]``.
        responsibility: Recovery responsibility per sample, shape ``[N]``.
        sketch_weight: Weight of sketch distillation inside each sample loss.
        eps: Positive numerical-validity bound retained for API consistency.
    """

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [N,D]")
    if (
        current_sketch.shape != prediction_time_sketch.shape
        or current_sketch.ndim != 2
        or current_sketch.shape[0] != prediction.shape[0]
    ):
        raise ValueError("current and historical sketches must share shape [N,S]")
    if responsibility.ndim != 1 or responsibility.shape[0] != prediction.shape[0]:
        raise ValueError("responsibility must have shape [N]")
    if sketch_weight < 0:
        raise ValueError("sketch_weight must be non-negative")

    historical = prediction_time_sketch.detach().to(current_sketch)
    prediction_loss = (prediction - target.to(prediction)).pow(2).mean(dim=-1)
    sketch_loss = (current_sketch - historical).pow(2).mean(dim=-1)
    per_sample = prediction_loss + float(sketch_weight) * sketch_loss
    weights = responsibility.detach().to(per_sample).clamp_min(0.0)
    if eps <= 0:
        raise ValueError("eps must be positive")
    loss = (weights * per_sample).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Recovery replay loss is not finite")
    return loss, {
        "prediction_loss": prediction_loss.detach().mean(),
        "sketch_loss": sketch_loss.detach().mean(),
        "weight_mass": weights.detach().sum(),
    }
