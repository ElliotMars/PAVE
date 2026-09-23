"""Responsibility-conditioned subspace protection for prediction heads.

Only the final regressor weight is protected.  Feature observations always
live in the 320-dimensional input space of that prediction head, so the
largest matrix constructed here is a ``[320, 320]`` covariance matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ExpertSubspaceState:
    """Persistent subspace state for one Expert."""

    basis: torch.Tensor
    eigenvalues: torch.Tensor
    effective_rank: int = 0
    stable_evidence_mass: float = 0.0
    protection_mass: float = 0.0
    gamma: float = 1.0
    captured_energy: float = 0.0
    basis_drift: float = 0.0
    last_refresh_step: int = -1


@dataclass(frozen=True)
class GradientProjectionStats:
    """Diagnostics from one regressor gradient projection."""

    parallel_norm: float
    perpendicular_norm: float
    gamma: float
    rank: int


class RegressorSubspaceProtector:
    """Maintain and apply per-Expert prediction-head capability subspaces."""

    def __init__(
        self,
        num_experts: int,
        feature_dim: int = 320,
        rank: int = 0,
        max_rank: int = 32,
        energy_threshold: float = 0.95,
        min_samples: int = 4,
        eps: float = 1e-8,
        subspace_lambda: float = 1e4,
        gamma_min: float = 0.0,
        gamma_max: float = 1.0,
        evidence_mass_scale: float = 8.0,
    ) -> None:
        if num_experts <= 0 or feature_dim <= 0:
            raise ValueError("num_experts and feature_dim must be positive")
        if rank < 0 or max_rank <= 0:
            raise ValueError("rank must be non-negative and max_rank positive")
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0,1]")
        if min_samples <= 0 or eps <= 0 or subspace_lambda < 0:
            raise ValueError("invalid subspace scalar configuration")
        if not math.isfinite(evidence_mass_scale) or evidence_mass_scale <= 0:
            raise ValueError("evidence_mass_scale must be finite and positive")
        if not 0.0 <= gamma_min <= gamma_max <= 1.0:
            raise ValueError("gamma bounds must satisfy 0 <= min <= max <= 1")

        self.num_experts = int(num_experts)
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.max_rank = min(int(max_rank), self.feature_dim)
        self.energy_threshold = float(energy_threshold)
        self.min_samples = int(min_samples)
        self.eps = float(eps)
        self.subspace_lambda = float(subspace_lambda)
        self.evidence_mass_scale = float(evidence_mass_scale)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.last_covariance_shape: Optional[tuple[int, int]] = None
        self.states = [
            ExpertSubspaceState(
                basis=torch.empty(self.feature_dim, 0),
                eigenvalues=torch.empty(0),
            )
            for _ in range(self.num_experts)
        ]

    def reset(self) -> None:
        """Clear all online bases while preserving configuration."""

        self.last_covariance_shape = None
        self.states = [
            ExpertSubspaceState(
                basis=torch.empty(self.feature_dim, 0),
                eigenvalues=torch.empty(0),
            )
            for _ in range(self.num_experts)
        ]

    def _validate_expert(self, expert_id: int) -> ExpertSubspaceState:
        if not 0 <= expert_id < self.num_experts:
            raise IndexError("invalid expert_id")
        return self.states[expert_id]

    def deactivate_protection(
        self, expert_id: int, step: Optional[int] = None
    ) -> None:
        """Disable protection while retaining the cached basis geometry."""

        state = self._validate_expert(expert_id)
        self._update_protection_strength(
            state, evidence_mass=0.0, current_lr=0.0
        )
        if step is not None:
            state.last_refresh_step = int(step)

    def _update_protection_strength(
        self,
        state: ExpertSubspaceState,
        evidence_mass: float,
        current_lr: float,
    ) -> None:
        """Update evidence strength independently of cached basis geometry."""

        raw_mass = float(evidence_mass)
        learning_rate = float(current_lr)
        if not math.isfinite(raw_mass) or raw_mass < 0.0:
            raise ValueError(
                "stable_evidence_mass must be finite and non-negative"
            )
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("current_lr must be finite and non-negative")
        protection_mass = (
            raw_mass / (raw_mass + self.evidence_mass_scale)
            if raw_mass > 0.0
            else 0.0
        )
        state.stable_evidence_mass = raw_mass
        state.protection_mass = protection_mass
        if protection_mass == 0.0:
            state.gamma = 1.0
            return
        lambda_eff = self.subspace_lambda * protection_mass
        gamma = 1.0 / (1.0 + learning_rate * lambda_eff)
        if not math.isfinite(gamma):
            raise FloatingPointError("subspace gamma is not finite")
        state.gamma = max(self.gamma_min, min(self.gamma_max, gamma))

    def _select_rank(self, eigenvalues: torch.Tensor) -> int:
        positive = int((eigenvalues > self.eps).sum().item())
        available = min(positive, self.max_rank)
        if available == 0:
            return 0
        if self.rank > 0:
            return min(self.rank, available)
        positive_energy = eigenvalues[:positive].clamp_min(0.0)
        total = positive_energy.sum()
        if not bool(torch.isfinite(total).item()) or float(total.item()) <= self.eps:
            return 0
        cumulative = torch.cumsum(
            positive_energy[:available], dim=0
        ) / total
        threshold = torch.tensor(
            self.energy_threshold, device=cumulative.device
        )
        reached = torch.nonzero(cumulative >= threshold, as_tuple=False)
        if reached.numel() == 0:
            return available
        return int(reached[0].item()) + 1

    @staticmethod
    def _basis_drift(old_basis: torch.Tensor, new_basis: torch.Tensor) -> float:
        if old_basis.numel() == 0 or new_basis.numel() == 0:
            return 0.0
        overlap = old_basis.T @ new_basis
        shared = min(old_basis.shape[1], new_basis.shape[1])
        similarity = overlap.pow(2).sum() / float(shared)
        return float((1.0 - similarity.clamp(0.0, 1.0)).item())

    def refresh_expert(
        self,
        expert_id: int,
        features: torch.Tensor,
        observation_weights: torch.Tensor,
        sample_count: int,
        step: int,
        stable_evidence_mass: Optional[float] = None,
        current_lr: float = 0.0,
    ) -> bool:
        """Refresh one basis from weighted ``[N,320]`` head features.

        ``observation_weights`` may contain per-channel weights for an
        FSNet-Time Expert, but their sum must represent sample-level evidence.
        Evidence strength is refreshed even when basis geometry cannot be.
        """

        state = self._validate_expert(expert_id)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [N,{self.feature_dim}]"
            )
        if observation_weights.shape != (features.shape[0],):
            raise ValueError("observation_weights must have shape [N]")
        weights = observation_weights.to(features)
        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError("observation_weights must be finite")
        if bool((weights < 0).any().item()):
            raise ValueError("observation_weights must be non-negative")
        evidence_mass = weights.sum()
        raw_mass = (
            float(evidence_mass.item())
            if stable_evidence_mass is None
            else float(stable_evidence_mass)
        )
        self._update_protection_strength(
            state, evidence_mass=raw_mass, current_lr=current_lr
        )

        if sample_count < self.min_samples or features.shape[0] == 0:
            return False
        if not bool(torch.isfinite(features).all().item()):
            return False
        if float(evidence_mass.item()) <= self.eps:
            return False

        # Preserve repeated activation directions with an uncentered weighted
        # second moment; mean subtraction would erase identical features.
        weighted_features = features * weights.unsqueeze(-1)
        covariance = features.T @ weighted_features
        covariance = covariance / evidence_mass.clamp_min(self.eps)
        covariance = 0.5 * (covariance + covariance.T)
        self.last_covariance_shape = tuple(covariance.shape)
        if self.last_covariance_shape != (self.feature_dim, self.feature_dim):
            raise RuntimeError("subspace covariance has an invalid shape")
        if not bool(torch.isfinite(covariance).all().item()):
            return False
        if not torch.allclose(
            covariance,
            covariance.T,
            atol=max(10.0 * self.eps, 1e-7),
            rtol=1e-5,
        ):
            return False

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        except RuntimeError:
            return False
        eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
        eigenvectors = eigenvectors.flip(1)
        if not (
            bool(torch.isfinite(eigenvalues).all().item())
            and bool(torch.isfinite(eigenvectors).all().item())
        ):
            return False
        selected_rank = self._select_rank(eigenvalues)
        if selected_rank == 0:
            return False

        new_basis = eigenvectors[:, :selected_rank]
        selected_energy = eigenvalues[:selected_rank].sum()
        total_energy = eigenvalues.sum().clamp_min(self.eps)
        captured = float((selected_energy / total_energy).clamp(0.0, 1.0).item())
        old_basis = state.basis.to(new_basis)
        drift = self._basis_drift(old_basis, new_basis)
        state.basis = new_basis.detach().cpu()
        state.eigenvalues = eigenvalues[:selected_rank].detach().cpu()
        state.effective_rank = selected_rank
        state.captured_energy = captured
        state.basis_drift = drift
        state.last_refresh_step = int(step)
        return True

    def gamma(self, expert_id: int, current_lr: float) -> float:
        """Compute soft protection strength from LR and normalized evidence."""

        state = self._validate_expert(expert_id)
        if not math.isfinite(current_lr) or current_lr < 0:
            raise ValueError("current_lr must be finite and non-negative")
        self._update_protection_strength(
            state,
            evidence_mass=state.stable_evidence_mass,
            current_lr=current_lr,
        )
        return state.gamma

    @staticmethod
    def project_with_basis(
        gradient: torch.Tensor,
        basis: torch.Tensor,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter the final feature dimension of ``gradient`` using ``basis``."""

        if gradient.ndim < 2:
            raise ValueError("prediction-head gradient must have at least 2 dims")
        if basis.ndim != 2 or basis.shape[0] != gradient.shape[-1]:
            raise ValueError("basis must have shape [feature_dim,rank]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0,1]")
        basis = basis.to(device=gradient.device, dtype=gradient.dtype)
        parallel = (gradient @ basis) @ basis.T
        perpendicular = gradient - parallel
        filtered = perpendicular + float(gamma) * parallel
        return filtered, parallel, perpendicular

    def filter_gradient(
        self,
        expert_id: int,
        gradient: torch.Tensor,
        current_lr: float,
        gamma_override: Optional[float] = None,
    ) -> tuple[torch.Tensor, GradientProjectionStats]:
        """Apply soft filtering to one regressor weight gradient."""

        state = self._validate_expert(expert_id)
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError("regressor gradient contains NaN or Inf")
        if state.effective_rank == 0:
            stats = GradientProjectionStats(
                parallel_norm=0.0,
                perpendicular_norm=float(torch.linalg.vector_norm(gradient).item()),
                gamma=1.0,
                rank=0,
            )
            return gradient, stats
        if state.protection_mass == 0.0:
            stats = GradientProjectionStats(
                parallel_norm=0.0,
                perpendicular_norm=float(torch.linalg.vector_norm(gradient).item()),
                gamma=1.0,
                rank=state.effective_rank,
            )
            return gradient, stats
        gamma = (
            self.gamma(expert_id, current_lr)
            if gamma_override is None
            else float(gamma_override)
        )
        filtered, parallel, perpendicular = self.project_with_basis(
            gradient, state.basis, gamma
        )
        if not bool(torch.isfinite(filtered).all().item()):
            raise FloatingPointError("filtered regressor gradient is not finite")
        stats = GradientProjectionStats(
            parallel_norm=float(torch.linalg.vector_norm(parallel).item()),
            perpendicular_norm=float(torch.linalg.vector_norm(perpendicular).item()),
            gamma=gamma,
            rank=state.effective_rank,
        )
        return filtered, stats

    def metrics(self) -> dict[str, list[float]]:
        """Return fixed-size per-Expert diagnostics."""

        return {
            "subspace_rank": [float(s.effective_rank) for s in self.states],
            "subspace_captured_energy": [s.captured_energy for s in self.states],
            "subspace_basis_drift": [s.basis_drift for s in self.states],
            "subspace_evidence_mass": [
                s.stable_evidence_mass for s in self.states
            ],
            "subspace_protection_mass": [s.protection_mass for s in self.states],
            "subspace_gamma": [s.gamma for s in self.states],
        }
