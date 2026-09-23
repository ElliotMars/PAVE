from typing import Optional

import torch


class OnlineRoutingCorrection:
    """Fast exponentiated-gradient correction over horizon/channel/expert."""

    def __init__(
        self,
        pred_len: int,
        c_out: int,
        num_experts: int,
        device: torch.device,
        correction_lr: float,
        correction_decay: float,
        correction_grad_clip: float,
        correction_logit_clip: float,
        eps: float = 1e-8,
    ) -> None:
        if pred_len <= 0 or c_out <= 0 or num_experts <= 0:
            raise ValueError("pred_len, c_out, and num_experts must be positive")
        if correction_lr < 0:
            raise ValueError("correction_lr must be non-negative")
        if not 0.0 <= correction_decay <= 1.0:
            raise ValueError("correction_decay must be in [0, 1]")
        if correction_grad_clip < 0 or correction_logit_clip < 0:
            raise ValueError("correction clips must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.shape = (int(pred_len), int(c_out), int(num_experts))
        self.device = torch.device(device)
        self.correction_lr = float(correction_lr)
        self.correction_decay = float(correction_decay)
        self.grad_clip = float(correction_grad_clip)
        self.logit_clip = float(correction_logit_clip)
        self.eps = float(eps)
        self.z = torch.zeros(self.shape, device=self.device)
        self._last_decay_origin: Optional[int] = None

    def reset(self) -> None:
        self.z.zero_()
        self._last_decay_origin = None

    def begin_origin(self, origin: int) -> None:
        """Apply decay once, even if several feedback events mature now."""

        origin = int(origin)
        if self._last_decay_origin == origin:
            return
        if self._last_decay_origin is not None and origin < self._last_decay_origin:
            raise ValueError("origins must be non-decreasing")
        self.z.mul_(1.0 - self.correction_decay)
        self._last_decay_origin = origin
        self._check_finite("decayed correction")

    def effective_weights(
        self, prior: torch.Tensor, top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Apply dense correction before optional top-k sparsification."""

        prior_device = prior.to(self.device)
        if tuple(prior_device.shape) == self.shape:
            correction = self.z
        elif (
            prior_device.ndim == 4
            and tuple(prior_device.shape[1:]) == self.shape
        ):
            correction = self.z.unsqueeze(0)
        else:
            raise ValueError(
                f"prior must have shape {self.shape} or [B, {self.shape}], got "
                f"{tuple(prior_device.shape)}"
            )
        self._check_tensor_finite(prior_device, "router prior")
        corrected = torch.softmax(
            torch.log(prior_device.clamp_min(self.eps)) + correction, dim=-1
        )
        if top_k is None or top_k >= self.shape[-1]:
            return corrected
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        values, indices = torch.topk(corrected, k=int(top_k), dim=-1)
        sparse = torch.zeros_like(corrected).scatter(-1, indices, values)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def update(
        self,
        horizon_index: int,
        expert_prediction: torch.Tensor,
        mixture_prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not 0 <= horizon_index < self.shape[0]:
            raise IndexError(f"horizon_index {horizon_index} is out of range")

        expert_prediction = expert_prediction.to(self.device)
        mixture_prediction = mixture_prediction.to(self.device)
        target = target.to(self.device)
        expected_expert_shape = (self.shape[1], self.shape[2])
        expected_channel_shape = (self.shape[1],)
        if tuple(expert_prediction.shape) != expected_expert_shape:
            raise ValueError(
                f"expert_prediction must have shape {expected_expert_shape}, got "
                f"{tuple(expert_prediction.shape)}"
            )
        if tuple(mixture_prediction.shape) != expected_channel_shape:
            raise ValueError(
                f"mixture_prediction must have shape {expected_channel_shape}, got "
                f"{tuple(mixture_prediction.shape)}"
            )
        if tuple(target.shape) != expected_channel_shape:
            raise ValueError(
                f"target must have shape {expected_channel_shape}, got "
                f"{tuple(target.shape)}"
            )
        self._check_tensor_finite(expert_prediction, "expert prediction")
        self._check_tensor_finite(mixture_prediction, "mixture prediction")
        self._check_tensor_finite(target, "target")

        gradient = 2.0 * (mixture_prediction - target).unsqueeze(-1) * (
            expert_prediction - mixture_prediction.unsqueeze(-1)
        )
        if self.grad_clip > 0:
            gradient = gradient.clamp(-self.grad_clip, self.grad_clip)
        self._check_tensor_finite(gradient, "correction gradient")

        self.z[horizon_index].add_(gradient, alpha=-self.correction_lr)
        self.z.sub_(self.z.mean(dim=-1, keepdim=True))
        if self.logit_clip > 0:
            self.z.clamp_(-self.logit_clip, self.logit_clip)
        self._check_finite("updated correction")
        return gradient.detach().clone()

    def _check_finite(self, name: str) -> None:
        self._check_tensor_finite(self.z, name)

    @staticmethod
    def _check_tensor_finite(tensor: torch.Tensor, name: str) -> None:
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
