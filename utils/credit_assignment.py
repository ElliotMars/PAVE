import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalCredit:
    loss: torch.Tensor
    responsibility: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class SampleCredit:
    mean_loss: torch.Tensor
    responsibility: torch.Tensor
    confidence: float
    matured_fraction: float


def normalized_responsibility_confidence(
    responsibility: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Return normalized inverse entropy along the Expert dimension."""

    if responsibility.ndim < 1:
        raise ValueError("responsibility must have an Expert dimension")
    experts = responsibility.shape[-1]
    if experts <= 0:
        raise ValueError("responsibility must contain at least one Expert")
    if experts == 1:
        return torch.ones_like(responsibility[..., 0])
    entropy = -(
        responsibility.clamp_min(eps)
        * responsibility.clamp_min(eps).log()
    ).sum(dim=-1)
    return (1.0 - entropy / math.log(experts)).clamp(0.0, 1.0)


def compute_local_credit(
    expert_prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    eps: float = 1e-8,
) -> LocalCredit:
    """Compute per-channel responsibility for one newly matured horizon."""

    if temperature <= 0:
        raise ValueError("local credit temperature must be positive")
    if expert_prediction.ndim != 2:
        raise ValueError("expert_prediction must have shape [C,E]")
    if target.ndim != 1 or target.shape[0] != expert_prediction.shape[0]:
        raise ValueError("target must have shape [C]")
    local_loss = (
        expert_prediction - target.to(expert_prediction).unsqueeze(-1)
    ).pow(2)
    responsibility = torch.softmax(-local_loss / temperature, dim=-1)
    confidence = normalized_responsibility_confidence(responsibility, eps=eps)
    return LocalCredit(local_loss, responsibility, confidence)


def compute_sample_credit(
    accumulated_loss: torch.Tensor,
    num_matured_values: int,
    matured_horizons: int,
    total_horizons: int,
    temperature: float,
    eps: float = 1e-8,
) -> SampleCredit:
    """Compute record ownership from mean counterfactual Expert error."""

    if accumulated_loss.ndim != 1:
        raise ValueError("accumulated_loss must have shape [E]")
    if num_matured_values <= 0:
        raise ValueError("num_matured_values must be positive")
    if total_horizons <= 0 or not 0 <= matured_horizons <= total_horizons:
        raise ValueError("invalid matured horizon count")
    if temperature <= 0:
        raise ValueError("sample credit temperature must be positive")

    mean_loss = accumulated_loss / float(num_matured_values)
    responsibility = torch.softmax(-mean_loss / temperature, dim=-1)
    matured_fraction = matured_horizons / float(total_horizons)
    specialization = normalized_responsibility_confidence(
        responsibility, eps=eps
    )
    confidence = float((matured_fraction * specialization).item())
    return SampleCredit(
        mean_loss=mean_loss,
        responsibility=responsibility,
        confidence=confidence,
        matured_fraction=matured_fraction,
    )


def full_router_objective(
    dense_prior: torch.Tensor,
    effective_weights: torch.Tensor,
    expert_prediction: torch.Tensor,
    target: torch.Tensor,
    entropy_weight: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full-feedback Router objective with detached Expert predictions."""

    if dense_prior.ndim != 4:
        raise ValueError("dense_prior must have shape [B,H,C,E]")
    if effective_weights.shape != dense_prior.shape:
        raise ValueError("effective_weights must match dense_prior")
    if expert_prediction.shape != dense_prior.shape:
        raise ValueError("expert_prediction must match dense_prior")
    if target.shape != dense_prior.shape[:-1]:
        raise ValueError("target must have shape [B,H,C]")
    prediction = (effective_weights * expert_prediction.detach()).sum(dim=-1)
    mixture_loss = (prediction - target).pow(2).mean()
    if entropy_weight != 0.0:
        entropy = -(
            dense_prior.clamp_min(eps)
            * dense_prior.clamp_min(eps).log()
        ).sum(dim=-1).mean()
    else:
        with torch.no_grad():
            detached = dense_prior.detach().clamp_min(eps)
            entropy = -(detached * detached.log()).sum(dim=-1).mean()
    total = mixture_loss - float(entropy_weight) * entropy
    return total, {
        "mixture_loss": mixture_loss,
        "entropy": entropy,
        "prediction": prediction,
    }


def partial_router_objective(
    current_prior: torch.Tensor,
    correction: torch.Tensor,
    expert_prediction: torch.Tensor,
    target: torch.Tensor,
    local_responsibility: torch.Tensor,
    local_confidence: torch.Tensor,
    local_credit_weight: float,
    entropy_weight: float,
    eps: float = 1e-8,
    top_k: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Router loss over only the horizon positions that matured now."""

    expected = current_prior.shape
    if current_prior.ndim != 3:
        raise ValueError("current_prior must have shape [N,C,E]")
    for name, tensor in (
        ("correction", correction),
        ("expert_prediction", expert_prediction),
        ("local_responsibility", local_responsibility),
    ):
        if tuple(tensor.shape) != tuple(expected):
            raise ValueError(
                f"{name} must have shape {tuple(expected)}, got "
                f"{tuple(tensor.shape)}"
            )
    if target.shape != current_prior.shape[:2]:
        raise ValueError("target must have shape [N,C]")
    if local_confidence.shape != current_prior.shape[:2]:
        raise ValueError("local_confidence must have shape [N,C]")

    effective_weights = torch.softmax(
        torch.log(current_prior.clamp_min(eps)) + correction.detach(), dim=-1
    )
    if top_k is not None and top_k < current_prior.shape[-1]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        values, indices = torch.topk(effective_weights, k=top_k, dim=-1)
        sparse = torch.zeros_like(effective_weights).scatter(
            -1, indices, values
        )
        effective_weights = sparse / sparse.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
    mixture = (effective_weights * expert_prediction.detach()).sum(dim=-1)
    partial_mix_loss = (mixture - target).pow(2).mean()
    local_ce = -(
        local_responsibility.detach()
        * torch.log(current_prior.clamp_min(eps))
    ).sum(dim=-1)
    local_credit_loss = (local_confidence.detach() * local_ce).mean()
    if entropy_weight != 0.0:
        entropy = -(
            current_prior.clamp_min(eps)
            * torch.log(current_prior.clamp_min(eps))
        ).sum(dim=-1).mean()
    else:
        with torch.no_grad():
            detached = current_prior.detach().clamp_min(eps)
            entropy = -(detached * detached.log()).sum(dim=-1).mean()
    total = (
        partial_mix_loss
        + local_credit_weight * local_credit_loss
        - entropy_weight * entropy
    )
    return total, {
        "partial_mix_loss": partial_mix_loss,
        "local_credit_loss": local_credit_loss,
        "entropy": entropy,
    }


def jensen_shannon_divergence(
    first: torch.Tensor, second: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("responsibilities must have the same shape [E]")
    first = first / first.sum().clamp_min(eps)
    second = second / second.sum().clamp_min(eps)
    midpoint = 0.5 * (first + second)
    first_kl = (
        first.clamp_min(eps)
        * (first.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    ).sum()
    second_kl = (
        second.clamp_min(eps)
        * (second.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    ).sum()
    return 0.5 * (first_kl + second_kl)
