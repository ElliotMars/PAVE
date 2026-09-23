import copy
import json
import math
import os
import time
import warnings
from collections import defaultdict
from collections import deque
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.fsnet import TSEncoder
from utils.comparator_evaluation import (
    KSwitchComparatorAccumulator,
    StaticComparatorAccumulator,
    evaluate_horizon_channel_static_comparator,
    evaluate_k_switch_dynamic_comparator,
    evaluate_static_comparator,
    save_comparator_diagnostics,
)
from utils.credit_assignment import (
    compute_sample_credit,
    full_router_objective,
    jensen_shannon_divergence,
    partial_router_objective,
)
from utils.expert_memory import (
    ExpertMemoryManager,
    VersionedMemoryItem,
    classify_capability_evolution,
)
from utils.evaluation_diagnostics import (
    SpecializationDiagnosticsAggregator,
    compute_router_oracle_diagnostics,
)
from utils.metrics import cumavg, metric
from utils.online_checks import StrictOnlineChecker
from utils.online_diagnostics import (
    OnlineDiagnosticsRecorder,
    StreamingTSBGradientDiagnostics,
)
from utils.online_routing import OnlineRoutingCorrection
from utils.progressive_feedback import (
    ProgressiveFeedbackEvent,
    ProgressiveFeedbackManager,
    ProgressiveForecastRecord,
)
from utils.recovery_learning import recovery_replay_objective
from utils.subspace_protection import RegressorSubspaceProtector
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings("ignore")


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)


class ExpertNet(nn.Module):
    """
    Single FSNet expert built on the same structure used by run.sh.
    """

    expert_type = "fsnet"

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.input_dim = args.enc_in + 7
        self.hidden_dim = 320
        self.output_dim = args.c_out * args.pred_len

        encoder = TSEncoder(
            input_dims=self.input_dim,
            output_dims=self.hidden_dim,
            hidden_dims=64,
            depth=10,
            device=self.device,
        )
        self.encoder = TS2VecEncoderWrapper(encoder, mask="all_true").to(self.device)
        self.regressor = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)

    def _encode(self, x, x_mark):
        x_with_mark = torch.cat([x, x_mark], dim=-1)  # [B, T, enc_in + 7]
        # FSNet's dilated conv stack can fail on some CUDA/cuDNN combos;
        # run this block without cuDNN to keep training stable.
        with torch.backends.cudnn.flags(enabled=False):
            h = self.encoder(x_with_mark)  # [B, T, H]
        return h

    def forward(self, x, x_mark, return_repr=False):
        h = self._encode(x, x_mark)
        z = h.mean(dim=1)  # [B, H]
        y_hat = self.regressor(h[:, -1, :])  # [B, D]
        if return_repr:
            return y_hat, z
        return y_hat

    def extract_head_features(self, x, x_mark):
        """Return the exact ``[B,320]`` input consumed by ``regressor``."""

        h = self._encode(x, x_mark)
        return h[:, -1, :]

    def encode_feature(self, x, x_mark):
        """Backward-compatible alias for the prediction-head feature."""

        return self.extract_head_features(x, x_mark)

    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if "PadConv" in type(layer).__name__:
                layer.store_grad()


class FSNetTimeExpertNet(nn.Module):
    """
    FSNet-Time expert built on the same structure used by run.sh.
    """

    expert_type = "fsnet_time"

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.input_dim = args.seq_len
        self.hidden_dim = 320
        self.pred_len = args.pred_len
        self.output_dim = args.c_out * args.pred_len

        encoder = TSEncoder(
            input_dims=self.input_dim,
            output_dims=self.hidden_dim,
            hidden_dims=64,
            depth=10,
            device=self.device,
        )
        self.encoder_time = TS2VecEncoderWrapper(encoder, mask="all_true").to(self.device)
        self.regressor_time = nn.Linear(self.hidden_dim, self.pred_len).to(self.device)

    def _encode(self, x, x_mark):
        del x_mark
        # Matches exp_onenet_fsnet: call forward_time without an explicit mask.
        with torch.backends.cudnn.flags(enabled=False):
            h = self.encoder_time.encoder.forward_time(x)  # [B, C, H]
        return h

    def forward(self, x, x_mark, return_repr=False):
        h = self._encode(x, x_mark)
        y_hat = self.regressor_time(h).transpose(1, 2)  # [B, pred_len, C]
        y_hat = rearrange(y_hat, "b t d -> b (t d)")
        z = h.mean(dim=1)  # [B, H]
        if return_repr:
            return y_hat, z
        return y_hat

    def extract_head_features(self, x, x_mark):
        """Return the exact ``[B,C,320]`` input of ``regressor_time``."""

        return self._encode(x, x_mark)

    def encode_feature(self, x, x_mark):
        """Backward-compatible pooled feature used by older callers."""

        return self.extract_head_features(x, x_mark).mean(dim=1)

    def store_grad(self):
        for name, layer in self.encoder_time.named_modules():
            if "PadConv" in type(layer).__name__:
                layer.store_grad()


class RoutingFeatureEncoder(nn.Module):
    """
    Shared feature encoder for routing. It produces a sequence feature from the
    original series; the router consumes the pooled feature to assign experts.
    """

    def __init__(self, args, hidden_dim, device):
        super().__init__()
        self.device = device
        self.input_dim = args.enc_in + 7
        self.input_proj = nn.Linear(self.input_dim, hidden_dim).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=args.dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)

    def forward(self, x, x_mark):
        x_with_mark = torch.cat([x, x_mark], dim=-1)  # [B, T, enc_in + 7]
        h = self.input_proj(x_with_mark)  # [B, T, H]
        h = h.transpose(0, 1)  # [T, B, H]
        h = self.encoder(h)
        h = h.transpose(0, 1)  # [B, T, H]
        return h, h[:, -1, :]


class net(nn.Module):
    """
    Multi-expert version with mixed FSNet and FSNet-Time experts.
    Output shape without router: [batch_size, num_experts, output_dim].
    """

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.num_experts = max(1, int(getattr(args, "num_experts", 4)))
        self.top_k = max(1, min(int(getattr(args, "top_k", self.num_experts)), self.num_experts))
        self.expert_composition = str(
            getattr(args, "expert_composition", "mixed")
        ).lower()
        if self.expert_composition not in {"mixed", "fsnet", "fsnet_time"}:
            raise ValueError("invalid expert_composition")
        if self.expert_composition == "fsnet":
            self.num_fsnet_experts = self.num_experts
            self.num_time_experts = 0
        elif self.expert_composition == "fsnet_time":
            self.num_fsnet_experts = 0
            self.num_time_experts = self.num_experts
        else:
            self.num_time_experts = min(2, self.num_experts)
            self.num_fsnet_experts = self.num_experts - self.num_time_experts
        self.hidden_dim = 320
        self.c_out = args.c_out
        self.pred_len = args.pred_len
        self.router_temperature = max(float(getattr(args, "router_temperature", 2.0)), 1e-3)
        self.router_granularity = getattr(args, "router_granularity", "channel")
        self.capability_sketch_dim = int(
            getattr(args, "capability_sketch_dim", 32)
        )
        self.capability_sketch_seed = int(
            getattr(args, "capability_sketch_seed", 2025)
        )
        if self.capability_sketch_dim <= 0:
            raise ValueError("capability_sketch_dim must be positive")
        if self.router_granularity not in {"channel", "horizon_channel"}:
            raise ValueError(
                "router_granularity must be either 'channel' or 'horizon_channel'"
            )

        experts = []
        for _ in range(self.num_fsnet_experts):
            experts.append(ExpertNet(args, device=self.device))
        for _ in range(self.num_time_experts):
            experts.append(FSNetTimeExpertNet(args, device=self.device))
        self.experts = nn.ModuleList(experts)
        projection_generator = torch.Generator(device="cpu")
        projection_generator.manual_seed(self.capability_sketch_seed)
        capability_projection = torch.randn(
            self.num_experts,
            self.hidden_dim,
            self.capability_sketch_dim,
            generator=projection_generator,
        ) / math.sqrt(self.hidden_dim)
        self.register_buffer(
            "capability_projection", capability_projection.to(self.device)
        )

        self.feature_encoder = RoutingFeatureEncoder(args, self.hidden_dim, self.device)
        self.router = nn.Linear(self.hidden_dim, self.c_out * self.num_experts).to(self.device)
        # Start from a safe uniform ensemble.
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        if self.router_granularity == "horizon_channel":
            self.horizon_head = nn.Linear(
                self.hidden_dim, self.pred_len * self.num_experts
            ).to(self.device)
            self.horizon_bias = nn.Parameter(
                torch.zeros(self.pred_len, self.num_experts, device=self.device)
            )
            nn.init.zeros_(self.horizon_head.weight)
            nn.init.zeros_(self.horizon_head.bias)
        else:
            # Do not register extra parameters in legacy mode, so old
            # checkpoints retain exactly the same state_dict structure.
            self.horizon_head = None
            self.register_parameter("horizon_bias", None)
        self.use_router = False

    def router_parameters(self) -> List[nn.Parameter]:
        parameters = list(self.feature_encoder.parameters()) + list(self.router.parameters())
        if self.horizon_head is not None:
            parameters += list(self.horizon_head.parameters()) + [self.horizon_bias]
        return parameters

    def set_router_mode(self, use_router: bool) -> None:
        self.use_router = use_router

    def _sparsify_prior(self, prior: torch.Tensor) -> torch.Tensor:
        if self.top_k >= self.num_experts:
            return prior
        _, topk_idx = torch.topk(prior, k=self.top_k, dim=-1)
        mask = torch.zeros_like(prior)
        mask.scatter_(-1, topk_idx, 1.0)
        prior = prior * mask
        return prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _compute_prior(
        self, x: torch.Tensor, x_mark: torch.Tensor
    ) -> torch.Tensor:
        _, route_feature = self.feature_encoder(x, x_mark)
        channel_logits = self.router(route_feature).reshape(
            x.shape[0], self.c_out, self.num_experts
        )
        if self.router_granularity == "channel":
            channel_prior = torch.softmax(
                channel_logits / self.router_temperature, dim=-1
            )
            prior = channel_prior.unsqueeze(1).expand(
                x.shape[0], self.pred_len, self.c_out, self.num_experts
            )
            assert prior.shape == (
                x.shape[0], self.pred_len, self.c_out, self.num_experts
            )
            return prior

        assert self.horizon_head is not None and self.horizon_bias is not None
        horizon_logits = self.horizon_head(route_feature).reshape(
            x.shape[0], self.pred_len, self.num_experts
        )
        scores = (
            channel_logits.unsqueeze(1)
            + horizon_logits.unsqueeze(2)
            + self.horizon_bias.view(1, self.pred_len, 1, self.num_experts)
        )
        prior = torch.softmax(scores / self.router_temperature, dim=-1)
        assert prior.shape == (
            x.shape[0], self.pred_len, self.c_out, self.num_experts
        )
        return prior

    def _apply_online_correction(
        self,
        prior: torch.Tensor,
        correction: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if prior.ndim == 3:
            if correction.ndim == 3:
                correction = correction.unsqueeze(0)
            if correction.ndim != 4:
                raise ValueError(
                    "channel prior correction must have shape [H,C,E] or [B,H,C,E]"
                )
            prior = prior.unsqueeze(1).expand(
                prior.shape[0], correction.shape[-3], self.c_out, self.num_experts
            )
        elif prior.ndim == 4 and correction.ndim == 3:
            correction = correction.unsqueeze(0)
        if prior.ndim != 4 or correction.ndim != 4:
            raise ValueError("prior and correction must resolve to [B,H,C,E]")
        if prior.shape[-3:] != correction.shape[-3:]:
            raise ValueError(
                f"prior/correction shape mismatch: {tuple(prior.shape)} vs "
                f"{tuple(correction.shape)}"
            )
        corrected = torch.softmax(
            torch.log(prior.clamp_min(eps)) + correction, dim=-1
        )
        return self._sparsify_prior(corrected)

    def _compute_gates(
        self, x: torch.Tensor, x_mark: torch.Tensor
    ) -> torch.Tensor:
        """Return effective gates while preserving legacy channel shape."""
        effective = self._sparsify_prior(self._compute_prior(x, x_mark))
        if self.router_granularity == "channel":
            return effective[:, 0]
        return effective

    def _prepare_expert_input(self, expert, x, x_mark):
        # FSNet-Time applies masks in-place, so each expert gets an isolated input.
        del expert
        return x.clone(), x_mark

    def compute_capability_sketch(
        self, representations: torch.Tensor
    ) -> torch.Tensor:
        expected = (
            representations.shape[0],
            self.num_experts,
            self.hidden_dim,
        )
        if representations.ndim != 3 or tuple(representations.shape) != expected:
            raise ValueError(
                f"representations must have shape {expected}, got "
                f"{tuple(representations.shape)}"
            )
        projected = torch.einsum(
            "beh,ehd->bed", representations, self.capability_projection
        )
        return F.normalize(projected, p=2, dim=-1, eps=1e-8)

    def forward_experts(self, x, x_mark, return_repr=False):
        if return_repr:
            ys, zs = [], []
            for expert in self.experts:
                expert_x, expert_x_mark = self._prepare_expert_input(expert, x, x_mark)
                y_i, z_i = expert(expert_x, expert_x_mark, return_repr=True)
                ys.append(y_i)
                zs.append(z_i)
            outputs = torch.stack(ys, dim=1)  # [B, E, D]
            reps = torch.stack(zs, dim=1)  # [B, E, H]
            return outputs, reps
        expert_outputs = []
        for expert in self.experts:
            expert_x, expert_x_mark = self._prepare_expert_input(expert, x, x_mark)
            expert_outputs.append(expert(expert_x, expert_x_mark))
        outputs = torch.stack(expert_outputs, dim=1)  # [B, E, D]
        return outputs

    def aggregate_with_gates(
        self, gates: torch.Tensor, outputs: torch.Tensor
    ) -> torch.Tensor:
        if outputs.ndim != 3:
            raise ValueError(
                f"outputs must have shape [B,E,H*C], got {tuple(outputs.shape)}"
            )
        batch_size = outputs.shape[0]
        expected_outputs = (batch_size, self.num_experts, self.pred_len * self.c_out)
        if tuple(outputs.shape) != expected_outputs:
            raise ValueError(
                f"outputs must have shape {expected_outputs}, got {tuple(outputs.shape)}"
            )
        outputs = outputs.reshape(
            batch_size, self.num_experts, self.pred_len, self.c_out
        )
        if gates.ndim == 3:
            expected_gates = (batch_size, self.c_out, self.num_experts)
            if tuple(gates.shape) != expected_gates:
                raise ValueError(
                    f"channel gates must have shape {expected_gates}, got "
                    f"{tuple(gates.shape)}"
                )
            weights = gates.permute(0, 2, 1).unsqueeze(2)
        elif gates.ndim == 4:
            expected_gates = (
                batch_size, self.pred_len, self.c_out, self.num_experts
            )
            if tuple(gates.shape) != expected_gates:
                raise ValueError(
                    f"horizon-channel gates must have shape {expected_gates}, got "
                    f"{tuple(gates.shape)}"
                )
            weights = gates.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "gates must have shape [B,C,E] or [B,H,C,E], got "
                f"{tuple(gates.shape)}"
            )
        prediction = torch.sum(weights * outputs, dim=1)
        return rearrange(prediction, "b t d -> b (t d)")

    def route_and_aggregate(self, x, x_mark, outputs=None):
        gates = self._compute_gates(x, x_mark)
        if outputs is None:
            outputs = self.forward_experts(x, x_mark, return_repr=False)
        return self.aggregate_with_gates(gates, outputs)

    def forward(self, x, x_mark):
        if self.use_router:
            return self.route_and_aggregate(x, x_mark)
        return self.forward_experts(x, x_mark, return_repr=False)

    def store_grad(self):
        for expert in self.experts:
            expert.store_grad()


class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ["none", "full", "regressor"]
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = net(args, device=self.device)
        self.lambda_div = float(getattr(args, "lambda_div", 0.0))
        self.tsb_alpha = float(getattr(args, "tsb_alpha", 0.5))
        self.tsb_eps = float(getattr(args, "tsb_eps", 1e-8))
        self.tsb_buffer_size = int(getattr(args, "tsb_buffer_size", 8))
        requested_strategy = str(
            getattr(args, "expert_update_strategy", "tsb")
        ).lower()
        if requested_strategy not in {"plain", "tsb", "subspace", "hybrid"}:
            raise ValueError("invalid expert_update_strategy")
        disable_tsb = bool(getattr(args, "disable_tsb", False))
        if disable_tsb and requested_strategy in {"tsb", "hybrid"}:
            replacement = "plain" if requested_strategy == "tsb" else "subspace"
            print(
                "[WARNING] --disable_tsb conflicts with "
                f"--expert_update_strategy={requested_strategy}; using {replacement}"
            )
            requested_strategy = replacement
        self.expert_update_strategy = requested_strategy
        self.use_tsb = requested_strategy in {"tsb", "hybrid"}
        self.use_subspace = requested_strategy in {"subspace", "hybrid"}
        self.tsb_smoothing_enabled = self.use_tsb and not bool(
            getattr(args, "disable_tsb_smoothing", False)
        )
        self.tsb_conflict_filter_enabled = self.use_tsb and not bool(
            getattr(args, "disable_tsb_conflict_filter", False)
        )
        self.tsb_reference_enabled = (
            self.tsb_smoothing_enabled or self.tsb_conflict_filter_enabled
        )
        self.adaptive_controller = str(
            getattr(args, "adaptive_controller", "dynamic")
        ).lower()
        if self.adaptive_controller not in {"fixed", "dynamic"}:
            raise ValueError("invalid adaptive_controller")
        self.online_buffer = deque(maxlen=self.tsb_buffer_size)
        self.expert_params = list(self.model.experts.parameters())
        self.router_params = self.model.router_parameters()
        self.base_learning_rate_expert = float(self.args.online_lr_expert)
        self.base_learning_rate_router = float(self.args.online_lr_router)
        self.expert_grad_clip = float(getattr(args, "expert_grad_clip", 1.0))
        self.router_grad_clip = float(getattr(args, "router_grad_clip", 0.5))
        self.router_entropy_weight = float(getattr(args, "router_entropy_weight", 1e-3))
        self.online_log_interval = int(getattr(args, "online_log_interval", 500))
        self.max_online_steps = int(getattr(args, "max_online_steps", -1))
        self.strict_online_checks = bool(
            getattr(args, "strict_online_checks", False)
        )
        self.online_step = 0
        self.fallback_count = 0
        self.fallback_channel_count = 0
        self.prev_online_mse = None
        self.online_mse_ema = None
        self.online_mse_ema_beta = 0.9
        self.progressive_fb = bool(getattr(args, "progressive_fb", False))
        self.online_correction_enabled = not bool(
            getattr(args, "disable_online_correction", False)
        )
        self.version_awareness_enabled = not bool(
            getattr(args, "disable_version_awareness", False)
        )
        self.recovery_enabled = not bool(
            getattr(args, "disable_recovery", False)
        )
        self.direction_awareness_enabled = (
            self.version_awareness_enabled
            and not bool(
                getattr(args, "disable_directional_recovery", False)
            )
        )
        # Backward-compatible diagnostics/config name. This gate represents
        # direction awareness and is intentionally independent of Recovery.
        self.directional_recovery_enabled = self.direction_awareness_enabled
        self.capability_rebase_enabled = self.direction_awareness_enabled
        self.expert_online_update_enabled = not bool(
            getattr(args, "disable_expert_online_update", False)
        )
        self.credit_weighted_subspace = not bool(
            getattr(args, "disable_credit_weighted_subspace", False)
        )
        self.routing_correction = OnlineRoutingCorrection(
            pred_len=args.pred_len,
            c_out=args.c_out,
            num_experts=self.model.num_experts,
            device=self.device,
            correction_lr=float(getattr(args, "correction_lr", 0.1)),
            correction_decay=float(getattr(args, "correction_decay", 0.01)),
            correction_grad_clip=float(
                getattr(args, "correction_grad_clip", 10.0)
            ),
            correction_logit_clip=float(
                getattr(args, "correction_logit_clip", 5.0)
            ),
        )
        self.progressive_origin = 0
        self.local_credit_temperature = float(
            getattr(args, "local_credit_temperature", 1.0)
        )
        self.sample_credit_temperature = float(
            getattr(args, "sample_credit_temperature", 1.0)
        )
        self.local_credit_weight = float(
            getattr(args, "local_credit_weight", 0.1)
        )
        self.min_credit_eps = float(getattr(args, "min_credit_eps", 1e-8))
        self.recovery_degradation_margin = float(
            getattr(args, "recovery_degradation_margin", 0.0)
        )
        if (
            not math.isfinite(self.recovery_degradation_margin)
            or self.recovery_degradation_margin < 0.0
        ):
            raise ValueError(
                "recovery_degradation_margin must be finite and non-negative"
            )
        self.responsibility_threshold = float(
            getattr(args, "responsibility_threshold", 0.3)
        )
        self.alignment_threshold = float(
            getattr(args, "alignment_threshold", 0.8)
        )
        self.credit_top_k = max(
            1,
            min(
                int(getattr(args, "credit_top_k", 1)),
                self.model.num_experts,
            ),
        )
        self.memory_refresh_interval = int(
            getattr(args, "memory_refresh_interval", 100)
        )
        self.subspace_refresh_interval = int(
            getattr(args, "subspace_refresh_interval", 100)
        )
        self.recovery_batch_size = int(
            getattr(args, "recovery_batch_size", 2)
        )
        self.recovery_loss_weight = float(
            getattr(args, "recovery_loss_weight", 0.1)
        )
        self.recovery_sketch_weight = float(
            getattr(args, "recovery_sketch_weight", 1.0)
        )
        stable_capacity = int(getattr(args, "stable_buffer_size", 32))
        recovery_capacity = int(getattr(args, "recovery_buffer_size", 32))
        if not self.recovery_enabled:
            recovery_capacity = 0
        self.memory_manager = ExpertMemoryManager(
            num_experts=self.model.num_experts,
            stable_capacity=stable_capacity,
            recovery_capacity=recovery_capacity,
            responsibility_threshold=self.responsibility_threshold,
            alignment_threshold=self.alignment_threshold,
            duplicate_threshold=float(
                getattr(args, "buffer_duplicate_threshold", 0.98)
            ),
            failure_penalty=float(
                getattr(args, "recovery_failure_penalty", 0.5)
            ),
            max_recovery_attempts=int(
                getattr(args, "max_recovery_attempts", 3)
            ),
            storage_dtype=str(getattr(args, "buffer_storage_dtype", "fp16")),
            promote_alignment_threshold=float(
                getattr(args, "promote_alignment_threshold", 0.9)
            ),
            promote_loss_threshold=float(
                getattr(args, "promote_loss_threshold", 1.0)
            ),
            directional_recovery_enabled=self.directional_recovery_enabled,
            version_awareness_enabled=self.version_awareness_enabled,
            recovery_degradation_margin=self.recovery_degradation_margin,
            degradation_eps=self.min_credit_eps,
            capability_rebase_enabled=self.capability_rebase_enabled,
            recovery_enabled=self.recovery_enabled,
        )
        self.subspace_scope = str(getattr(args, "subspace_scope", "regressor"))
        if self.subspace_scope != "regressor":
            raise ValueError("the first subspace implementation only supports regressor")
        self.subspace_protector = RegressorSubspaceProtector(
            num_experts=self.model.num_experts,
            feature_dim=self.model.hidden_dim,
            rank=int(getattr(args, "subspace_rank", 0)),
            max_rank=int(getattr(args, "subspace_max_rank", 32)),
            energy_threshold=float(
                getattr(args, "subspace_energy_threshold", 0.95)
            ),
            min_samples=int(getattr(args, "subspace_min_samples", 4)),
            eps=float(getattr(args, "subspace_eps", 1e-8)),
            subspace_lambda=float(getattr(args, "subspace_lambda", 1e4)),
            evidence_mass_scale=float(
                getattr(args, "subspace_evidence_mass_scale", 8.0)
            ),
            gamma_min=float(getattr(args, "subspace_gamma_min", 0.0)),
            gamma_max=float(getattr(args, "subspace_gamma_max", 1.0)),
        )
        self.diagnostics = OnlineDiagnosticsRecorder(
            self.model.num_experts, self.online_log_interval
        )
        self.tsb_gradient_diagnostics = StreamingTSBGradientDiagnostics()
        self.specialization_diagnostics = SpecializationDiagnosticsAggregator(
            pred_len=args.pred_len,
            c_out=args.c_out,
            num_experts=self.model.num_experts,
        )
        self.comparator_diagnostics = StaticComparatorAccumulator(
            num_experts=self.model.num_experts,
            horizon=args.pred_len,
            channels=args.c_out,
        )
        self.dynamic_comparator_diagnostics = (
            KSwitchComparatorAccumulator(
                num_experts=self.model.num_experts,
                max_points=int(
                    getattr(args, "dynamic_comparator_max_points", 64)
                ),
            )
            if bool(getattr(args, "dynamic_comparator", False))
            else None
        )
        self.dynamic_comparator_max_switches = int(
            getattr(args, "dynamic_comparator_max_switches", 1)
        )
        self.credit_diagnostic_buffer_size = max(
            1, int(getattr(args, "credit_diagnostic_buffer_size", 10000))
        )
        self.credit_diagnostics = deque(
            maxlen=self.credit_diagnostic_buffer_size
        )
        self.credit_diagnostic_total_count = 0
        self.expert_update_count = 0
        self.completed_record_count = 0
        self.processed_online_origins = 0
        self.online_test_early_ended = False
        self.last_expert_update_diagnostics: dict[str, Any] = {}
        self.online_checker = StrictOnlineChecker(self.strict_online_checks)
        self._test_start_model_state: dict[str, Any] | None = None
        self._test_start_optimizer_states: dict[str, Any] | None = None
        self._test_start_requires_grad: dict[str, bool] | None = None

    @staticmethod
    def _state_to_cpu(value: Any) -> Any:
        """Deep-copy nested checkpoint state without retaining GPU aliases."""

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {
                key: Exp_TS2VecSupervised._state_to_cpu(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                Exp_TS2VecSupervised._state_to_cpu(item) for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                Exp_TS2VecSupervised._state_to_cpu(item) for item in value
            )
        return copy.deepcopy(value)

    def _invalidate_test_start_state(self) -> None:
        """Forget a baseline before training or loading a new checkpoint."""

        self._test_start_model_state = None
        self._test_start_optimizer_states = None
        self._test_start_requires_grad = None

    def _capture_test_start_state(self) -> None:
        """Capture the post-checkpoint model and online optimizer baseline."""

        if not hasattr(self, "opt_expert") or not hasattr(self, "opt_router"):
            raise RuntimeError(
                "online optimizers must exist before capturing test-start state"
            )
        self._test_start_model_state = self._state_to_cpu(
            self.model.state_dict()
        )
        self._test_start_optimizer_states = {
            "expert": self._state_to_cpu(self.opt_expert.state_dict()),
            "router": self._state_to_cpu(self.opt_router.state_dict()),
        }
        self._test_start_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }

    def _restore_test_start_state(self) -> None:
        """Restore parameters, FSNet buffers, optimizer moments and flags."""

        if (
            self._test_start_model_state is None
            or self._test_start_optimizer_states is None
            or self._test_start_requires_grad is None
        ):
            raise RuntimeError("test-start state has not been captured")
        self.model.load_state_dict(self._test_start_model_state, strict=True)
        self.opt_expert.load_state_dict(
            self._test_start_optimizer_states["expert"]
        )
        self.opt_router.load_state_dict(
            self._test_start_optimizer_states["router"]
        )
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = self._test_start_requires_grad[name]
        self.opt_expert.zero_grad(set_to_none=True)
        self.opt_router.zero_grad(set_to_none=True)

    def _prepare_test_start_state(self) -> None:
        """Capture once, then restore the immutable baseline per test call."""

        if self._test_start_model_state is None:
            self._capture_test_start_state()
        else:
            self._restore_test_start_state()

    def _set_optimizer_lr(self, optimizer, lr):
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _adaptive_online_hparams(self):
        if self.adaptive_controller == "fixed":
            return (
                self.base_learning_rate_expert,
                self.base_learning_rate_router,
                self.tsb_alpha,
            )
        if self.prev_online_mse is None or self.online_mse_ema is None:
            return self.base_learning_rate_expert, self.base_learning_rate_router, self.tsb_alpha

        ratio = self.prev_online_mse / (self.online_mse_ema + self.tsb_eps)
        # A sudden error spike must make the update more conservative, not
        # increase the step size. Never exceed the configured online LR.
        lr_scale = float(np.clip(1.0 / np.sqrt(max(ratio, self.tsb_eps)), 0.1, 1.0))
        expert_lr = self.base_learning_rate_expert * lr_scale
        router_lr = self.base_learning_rate_router * lr_scale
        dynamic_tsb_alpha = self.tsb_alpha
        if self.tsb_smoothing_enabled:
            dynamic_tsb_alpha = float(
                np.clip(self.tsb_alpha / lr_scale, self.tsb_alpha, 0.95)
            )
        return expert_lr, router_lr, dynamic_tsb_alpha

    def _update_online_mse_state(self, mse_value):
        mse_value = float(mse_value)
        self.prev_online_mse = mse_value
        if self.online_mse_ema is None:
            self.online_mse_ema = mse_value
        else:
            beta = self.online_mse_ema_beta
            self.online_mse_ema = beta * self.online_mse_ema + (1.0 - beta) * mse_value

    @contextmanager
    def _fsnet_state_updates(self, enabled):
        modules = [m for m in self.model.modules() if hasattr(m, "state_updates_enabled")]
        previous = [m.state_updates_enabled for m in modules]
        for module in modules:
            module.state_updates_enabled = enabled
        try:
            yield
        finally:
            for module, old_value in zip(modules, previous):
                module.state_updates_enabled = old_value

    def prepare_without_pretraining(self):
        """Initialize online optimizers for random-init evaluation."""

        self._invalidate_test_start_state()
        self.model.set_router_mode(True)
        self.opt_expert = self._select_expert_optimizer()
        self.opt_router = self._select_router_optimizer()
        self._set_optimizer_lr(
            self.opt_expert, self.base_learning_rate_expert
        )
        self._set_optimizer_lr(
            self.opt_router, self.base_learning_rate_router
        )
        return self.model


    def _safe_prediction(self, gates, outputs):
        """Use router output unless an individual value is NaN or infinite.

        Expert disagreement is expected under distribution shift and must be
        resolved by the learned router. The old sample-wise median fallback
        bypassed all channel gates when one ECL channel shifted, so fallback is
        now limited to non-finite values and is applied coordinate-wise.
        """
        raw_prediction = self.model.aggregate_with_gates(gates, outputs)
        finite_experts = torch.isfinite(outputs)
        finite_sum = torch.where(
            finite_experts, outputs, torch.zeros_like(outputs)
        ).sum(dim=1)
        finite_count = finite_experts.sum(dim=1).clamp_min(1)
        finite_mean = finite_sum / finite_count

        unsafe_values = ~torch.isfinite(raw_prediction)
        safe_prediction = torch.where(unsafe_values, finite_mean, raw_prediction)
        safe_prediction = torch.nan_to_num(
            safe_prediction, nan=0.0, posinf=0.0, neginf=0.0
        )

        unsafe_channels = unsafe_values.view(
            outputs.shape[0], self.args.pred_len, self.args.c_out
        ).any(dim=1)
        self.fallback_count += int(unsafe_values.sum().item())
        self.fallback_channel_count += int(unsafe_channels.sum().item())
        return safe_prediction, unsafe_channels

    @staticmethod
    def _independent_expert_mse(expert_outputs, target):
        """Sum independently supervised expert MSEs, matching OneNet's scale."""
        per_expert = (expert_outputs - target.unsqueeze(1)).pow(2).mean(dim=2)
        return per_expert.sum(dim=1).mean()

    def load_pretrained(self, checkpoint_path):
        self._invalidate_test_start_state()
        state = torch.load(checkpoint_path, map_location=self.device)
        incompatible = self.model.load_state_dict(state, strict=False)
        allowed_missing = {"capability_projection"}
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            exc = RuntimeError(
                "missing={} unexpected={}".format(
                    sorted(unexpected_missing), incompatible.unexpected_keys
                )
            )
            raise RuntimeError(
                "Incompatible multi-expert checkpoint. Re-run the v2 pretraining script."
            ) from exc
        self.model.set_router_mode(True)
        self.opt_expert = self._select_expert_optimizer()
        self.opt_router = self._select_router_optimizer()
        optimizer_path = os.path.join(os.path.dirname(checkpoint_path), "optimizer.pth")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.opt_expert.load_state_dict(optimizer_state["expert"])
            self.opt_router.load_state_dict(optimizer_state["router"])
            print("Loaded optimizer state:", optimizer_path)
        else:
            print("Optimizer state not found; starting online moments from zero")
        self._set_optimizer_lr(self.opt_expert, self.base_learning_rate_expert)
        self._set_optimizer_lr(self.opt_router, self.base_learning_rate_router)
        return self.model

    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "WTH": Dataset_Custom,
            "ECL": Dataset_Custom,
            "Solar": Dataset_Custom,
            "custom": Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag == "test":
            shuffle_flag = False
            drop_last = False
            batch_size = args.test_bsz
            freq = args.freq
        elif flag == "val":
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.detail_freq
        elif flag == "pred":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            delay_fb=args.delay_fb,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    def _select_expert_optimizer(self):
        self.opt_expert = optim.AdamW(
            self.expert_params, lr=self.args.learning_rate_expert, weight_decay=self.args.weight_decay
        )
        return self.opt_expert

    def _select_router_optimizer(self):
        self.opt_router = optim.AdamW(
            self.router_params, lr=self.args.learning_rate_router, weight_decay=self.args.weight_decay
        )
        return self.opt_router

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, setting):
        self._invalidate_test_start_state()
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.model.set_router_mode(True)
        self.opt_expert = self._select_expert_optimizer()
        self.opt_router = self._select_router_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.opt_expert.zero_grad()
                self.opt_router.zero_grad()

                expert_outputs, routed_pred, expert_reps, gates, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                expert_losses = [criterion(expert_outputs[:, e, :], true) for e in range(expert_outputs.shape[1])]
                loss_pred = torch.stack(expert_losses).sum()

                # Diversity loss across experts for same input batch.
                z = F.normalize(expert_reps, p=2, dim=-1, eps=1e-8)  # [B, E, H]
                sims = []
                e_num = z.shape[1]
                for ii in range(e_num):
                    for jj in range(e_num):
                        if ii == jj:
                            continue
                        sim_ij = F.cosine_similarity(z[:, ii, :], z[:, jj, :], dim=-1).mean()
                        sims.append(sim_ij)
                if len(sims) > 0:
                    loss_div = torch.stack(sims).mean()
                else:
                    loss_div = torch.tensor(0.0, device=self.device)

                loss_expert = loss_pred + self.lambda_div * loss_div
                gate_entropy = -(gates.clamp_min(1e-8) * gates.clamp_min(1e-8).log()).sum(dim=-1).mean()
                loss_router = criterion(routed_pred, true) - self.router_entropy_weight * gate_entropy
                loss = loss_expert + loss_router
                train_loss.append(loss_expert.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss_expert.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.opt_expert)
                    scaler.unscale_(self.opt_router)
                    nn.utils.clip_grad_norm_(self.expert_params, self.expert_grad_clip)
                    nn.utils.clip_grad_norm_(self.router_params, self.router_grad_clip)
                    scaler.step(self.opt_expert)
                    scaler.step(self.opt_router)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.expert_params, self.expert_grad_clip)
                    nn.utils.clip_grad_norm_(self.router_params, self.router_grad_clip)
                    self.opt_expert.step()
                    self.opt_router.step()
                self.model.store_grad()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Vali Loss: {3:.4f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            previous_best = early_stopping.best_score
            early_stopping(vali_loss, self.model, path)
            improved = previous_best is None or -vali_loss >= previous_best + early_stopping.delta
            if improved:
                torch.save(
                    {"expert": self.opt_expert.state_dict(), "router": self.opt_router.state_dict()},
                    os.path.join(path, "optimizer.pth"),
                )
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt_expert, epoch + 1, self.args)
            adjust_learning_rate(self.opt_router, epoch + 1, self.args)

        best_model_path = path + "/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        optimizer_path = os.path.join(path, "optimizer.pth")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.opt_expert.load_state_dict(optimizer_state["expert"])
            self.opt_router.load_state_dict(optimizer_state["router"])
        self._set_optimizer_lr(self.opt_expert, self.base_learning_rate_expert)
        self._set_optimizer_lr(self.opt_router, self.base_learning_rate_router)
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        self.model.set_router_mode(True)
        total_loss = []
        with torch.no_grad(), self._fsnet_state_updates(False):
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                _, routed_pred, _, _, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="vali"
                )
                loss = criterion(routed_pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _capability_sketch_for_expert(
        self, expert_id: int, representation: torch.Tensor
    ) -> torch.Tensor:
        """Project one Expert's differentiable ``[B,320]`` representation."""

        projection = self.model.capability_projection[expert_id]
        sketch = representation @ projection
        return F.normalize(sketch, p=2, dim=-1, eps=self.min_credit_eps)

    def _sample_recovery_batches(self) -> List[dict[str, Any]]:
        """Sample each sample ID at most once across all Experts per update."""

        if (
            not self.recovery_enabled
            or self.recovery_batch_size <= 0
            or self.recovery_loss_weight <= 0
        ):
            return []
        batches: List[dict[str, Any]] = []
        excluded_sample_ids: set[int] = set()
        for expert_id in range(self.model.num_experts):
            items = self.memory_manager.sample_recovery(
                expert_id,
                self.recovery_batch_size,
                excluded_sample_ids=excluded_sample_ids,
            )
            if not items:
                continue
            batches.append(
                {
                    "expert_id": expert_id,
                    "items": items,
                    "x": torch.cat([item.x for item in items], dim=0)
                    .float()
                    .to(self.device),
                    "x_mark": torch.cat([item.x_mark for item in items], dim=0)
                    .float()
                    .to(self.device),
                    "target": torch.cat([item.target for item in items], dim=0)
                    .float()
                    .to(self.device),
                    "historical_sketch": torch.stack(
                        [item.prediction_capability_sketch for item in items]
                    )
                    .float()
                    .to(self.device)
                    .detach(),
                    "responsibility": torch.tensor(
                        [item.recovery_credit for item in items],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                }
            )
        return batches

    def _compute_recovery_replay_loss(
        self, batches: List[dict[str, Any]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute read-only-state Recovery forwards with trainable parameters."""

        losses = []
        prediction_losses = []
        sketch_losses = []
        for batch in batches:
            expert_id = int(batch["expert_id"])
            expert = self.model.experts[expert_id]
            with self._fsnet_state_updates(False):
                prediction, representation = expert(
                    batch["x"].clone(), batch["x_mark"], return_repr=True
                )
                current_sketch = self._capability_sketch_for_expert(
                    expert_id, representation
                )
                target = batch["target"].reshape(prediction.shape)
                loss, components = recovery_replay_objective(
                    prediction=prediction,
                    target=target,
                    current_sketch=current_sketch,
                    prediction_time_sketch=batch["historical_sketch"],
                    responsibility=batch["responsibility"],
                    sketch_weight=self.recovery_sketch_weight,
                    eps=self.min_credit_eps,
                )
            losses.append(loss)
            prediction_losses.append(components["prediction_loss"])
            sketch_losses.append(components["sketch_loss"])
        if not losses:
            zero = torch.zeros((), device=self.device)
            return zero, {
                "recovery_prediction_loss": 0.0,
                "recovery_sketch_loss": 0.0,
            }
        total = self.recovery_loss_weight * torch.stack(losses).sum()
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("combined Recovery loss is not finite")
        return total, {
            "recovery_prediction_loss": float(
                torch.stack(prediction_losses).mean().item()
            ),
            "recovery_sketch_loss": float(
                torch.stack(sketch_losses).mean().item()
            ),
        }

    def _finalize_recovery_replay(
        self, batches: List[dict[str, Any]]
    ) -> None:
        """Re-evaluate replayed samples after the optimizer step and migrate them."""

        for batch in batches:
            expert_id = int(batch["expert_id"])
            expert = self.model.experts[expert_id]
            with torch.no_grad(), self._fsnet_state_updates(False):
                prediction, representation = expert(
                    batch["x"].clone(), batch["x_mark"], return_repr=True
                )
                current_sketch = self._capability_sketch_for_expert(
                    expert_id, representation
                )
            target = batch["target"].reshape(prediction.shape)
            losses = (prediction - target).pow(2).mean(dim=-1)
            historical = batch["historical_sketch"].detach().to(current_sketch)
            alignments = (
                ((current_sketch * historical).sum(dim=-1).clamp(-1.0, 1.0) + 1.0)
                / 2.0
            ).clamp(0.0, 1.0)
            for item, alignment, loss, sketch in zip(
                batch["items"], alignments, losses, current_sketch
            ):
                status = self.memory_manager.update_recovery_result(
                    expert_id,
                    item.sample_id,
                    float(alignment.item()),
                    float(loss.item()),
                    current_sketch=sketch.detach(),
                )
                self.diagnostics.increment(recovery_attempts=1)
                if status in {"promoted", "promoted_performance"}:
                    counts = {
                        "recovery_successes": 1,
                        "recovery_to_stable": 1,
                        "promotion_count": 1,
                    }
                    if status == "promoted_performance":
                        counts.update(
                            performance_recovery_count=1,
                            capability_rebase_count=1,
                        )
                    self.diagnostics.increment(**counts)
                elif status in {
                    "dropped_after_recovery",
                    "dropped_after_performance_recovery",
                }:
                    counts = {
                        "recovery_successes": 1,
                        "recovery_dropped_after_success": 1,
                        "drop_count": 1,
                    }
                    if status == "dropped_after_performance_recovery":
                        counts.update(
                            performance_recovery_count=1,
                            capability_rebase_count=1,
                        )
                    self.diagnostics.increment(**counts)
                elif status == "recovery":
                    self.diagnostics.increment(recovery_failed=1)
                elif status == "dropped":
                    self.diagnostics.increment(
                        recovery_failed=1,
                        recovery_attempt_exhausted=1,
                        drop_count=1,
                    )

    def _assign_raw_expert_gradients(
        self, gradients: List[torch.Tensor | None]
    ) -> None:
        for parameter, gradient in zip(self.expert_params, gradients):
            if gradient is None:
                parameter.grad = None
                continue
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError("current Expert gradient is not finite")
            parameter.grad = gradient.clone()

    def _apply_tsb_gradient_filter(
        self,
        current: List[torch.Tensor | None],
        reference: List[torch.Tensor] | None,
        alpha: float,
    ) -> dict[str, Any]:
        """Apply TSB and compute scalar whole-Expert mechanism diagnostics."""

        conflicts = 0
        compared = 0
        diagnostic_device = self.expert_params[0].device
        current_norm_sq = torch.zeros((), device=diagnostic_device)
        reference_norm_sq = torch.zeros((), device=diagnostic_device)
        current_reference_dot = torch.zeros((), device=diagnostic_device)
        modification_norm_sq = torch.zeros((), device=diagnostic_device)
        projection_removed_norm_sq = torch.zeros((), device=diagnostic_device)
        projection_base_norm_sq = torch.zeros((), device=diagnostic_device)
        has_reference = False
        projection_applied = False
        for parameter, gradient, ref_gradient in zip(
            self.expert_params,
            current,
            reference if reference is not None else [None] * len(current),
        ):
            if gradient is None:
                parameter.grad = None
                continue
            current_norm_sq += torch.sum(gradient * gradient)
            if ref_gradient is None:
                filtered = gradient
            else:
                has_reference = True
                reference_norm_sq += torch.sum(ref_gradient * ref_gradient)
                current_reference_dot += torch.sum(gradient * ref_gradient)
                if self.tsb_smoothing_enabled:
                    base_gradient = (
                        (1.0 - alpha) * gradient + alpha * ref_gradient
                    )
                else:
                    base_gradient = gradient
                if self.tsb_conflict_filter_enabled:
                    compared += 1
                    dot = torch.sum(base_gradient * ref_gradient)
                    if float(dot.item()) < 0.0:
                        conflicts += 1
                        projection_applied = True
                        norm_sq = (
                            torch.sum(ref_gradient * ref_gradient)
                            + self.tsb_eps
                        )
                        filtered = (
                            base_gradient
                            - (dot / norm_sq) * ref_gradient
                        )
                        projection_removed_norm_sq += torch.sum(
                            (filtered - base_gradient).pow(2)
                        )
                        projection_base_norm_sq += torch.sum(
                            base_gradient.pow(2)
                        )
                    else:
                        filtered = base_gradient
                else:
                    filtered = base_gradient
            if not bool(torch.isfinite(filtered).all().item()):
                raise FloatingPointError("TSB-filtered gradient is not finite")
            modification_norm_sq += torch.sum((filtered - gradient).pow(2))
            parameter.grad = filtered.clone()

        grad_cosine = None
        gradient_conflict = None
        if has_reference:
            denominator = torch.sqrt(
                current_norm_sq * reference_norm_sq
            ).clamp_min(self.tsb_eps)
            grad_cosine = float(
                (current_reference_dot / denominator).clamp(-1.0, 1.0).item()
            )
            gradient_conflict = grad_cosine < 0.0
        modification_ratio = float(
            (
                torch.sqrt(modification_norm_sq)
                / (torch.sqrt(current_norm_sq) + self.tsb_eps)
            ).item()
        )
        projection_removal_ratio = None
        if projection_applied:
            projection_removal_ratio = float(
                (
                    torch.sqrt(projection_removed_norm_sq)
                    / (torch.sqrt(projection_base_norm_sq) + self.tsb_eps)
                ).item()
            )
        return {
            "tsb_smoothing_enabled": self.tsb_smoothing_enabled,
            "tsb_conflict_filter_enabled": (
                self.tsb_conflict_filter_enabled
            ),
            "tsb_conflict_rate": conflicts / compared if compared else 0.0,
            "tsb_reference_available": has_reference,
            "tsb_grad_cosine": grad_cosine,
            "tsb_gradient_conflict": gradient_conflict,
            "tsb_modification_ratio": modification_ratio,
            "tsb_projection_removal_ratio": projection_removal_ratio,
        }

    def _apply_subspace_gradient_filter(
        self, current_lr: float
    ) -> dict[str, Any]:
        """Filter only prediction-head weight gradients, never their biases."""

        parallel = [0.0] * self.model.num_experts
        perpendicular = [0.0] * self.model.num_experts
        gammas = [1.0] * self.model.num_experts
        for expert_id, expert in enumerate(self.model.experts):
            head = (
                expert.regressor
                if hasattr(expert, "regressor")
                else expert.regressor_time
            )
            if head.weight.grad is None:
                continue
            filtered, stats = self.subspace_protector.filter_gradient(
                expert_id, head.weight.grad, current_lr=current_lr
            )
            head.weight.grad = filtered
            parallel[expert_id] = stats.parallel_norm
            perpendicular[expert_id] = stats.perpendicular_norm
            gammas[expert_id] = stats.gamma
        return {
            "parallel_gradient_norm": parallel,
            "perpendicular_gradient_norm": perpendicular,
            "subspace_gamma": gammas,
        }

    def _apply_expert_update_strategy(
        self,
        current: List[torch.Tensor | None],
        reference: List[torch.Tensor] | None,
        alpha: float,
        current_lr: float,
    ) -> dict[str, Any]:
        """Apply plain/TSB first and optional subspace filtering second."""

        diagnostics: dict[str, Any] = {
            "tsb_smoothing_enabled": False,
            "tsb_conflict_filter_enabled": False,
            "tsb_conflict_rate": 0.0,
            "tsb_reference_available": False,
            "tsb_grad_cosine": None,
            "tsb_gradient_conflict": None,
            "tsb_modification_ratio": 0.0,
            "tsb_projection_removal_ratio": None,
        }
        if self.expert_update_strategy in {"tsb", "hybrid"}:
            diagnostics.update(
                self._apply_tsb_gradient_filter(current, reference, alpha)
            )
        else:
            self._assign_raw_expert_gradients(current)
        if self.expert_update_strategy in {"subspace", "hybrid"}:
            diagnostics.update(
                self._apply_subspace_gradient_filter(current_lr)
            )
        else:
            diagnostics.update(
                {
                    "parallel_gradient_norm": [0.0] * self.model.num_experts,
                    "perpendicular_gradient_norm": [0.0]
                    * self.model.num_experts,
                    "subspace_gamma": [1.0] * self.model.num_experts,
                }
            )
        return diagnostics

    def _delayed_online_batch(
        self, dataset, feedback_queue, batch_x, batch_y, batch_x_mark, batch_y_mark
    ):
        batch_preds, batch_trues = [], []
        for i in range(batch_x.shape[0]):
            if len(feedback_queue) >= self.args.pred_len:
                released = feedback_queue.popleft()
                self._ol_one_batch(dataset, *released)

            current = (
                batch_x[i : i + 1].detach().clone(),
                batch_y[i : i + 1].detach().clone(),
                batch_x_mark[i : i + 1].detach().clone(),
                batch_y_mark[i : i + 1].detach().clone(),
            )
            pred, true = self._predict_without_update(
                current[0], current[1], current[2]
            )
            feedback_queue.append(current)
            batch_preds.append(pred)
            batch_trues.append(true)

        return torch.cat(batch_preds, dim=0), torch.cat(batch_trues, dim=0)

    def _current_observation(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Select the one target vector that is observable at this origin."""

        if batch_x.ndim != 3 or batch_x.shape[0] != 1:
            raise ValueError(
                f"progressive samples must have shape [1,T,C], got {tuple(batch_x.shape)}"
            )
        f_dim = -1 if self.args.features == "MS" else 0
        observation = batch_x.float()[:, -1, f_dim:].reshape(-1)
        if tuple(observation.shape) != (self.args.c_out,):
            raise ValueError(
                f"observable target must have shape {(self.args.c_out,)}, got "
                f"{tuple(observation.shape)}"
            )
        return observation

    def _horizon_prior(self, prior: torch.Tensor) -> torch.Tensor:
        if prior.ndim == 3:
            expected = (
                prior.shape[0],
                self.args.c_out,
                self.model.num_experts,
            )
            if tuple(prior.shape) != expected:
                raise ValueError(
                    f"channel prior must have shape {expected}, got {tuple(prior.shape)}"
                )
            return prior.unsqueeze(1).expand(
                prior.shape[0],
                self.args.pred_len,
                self.args.c_out,
                self.model.num_experts,
            )
        expected = (
            prior.shape[0],
            self.args.pred_len,
            self.args.c_out,
            self.model.num_experts,
        )
        if prior.ndim != 4 or tuple(prior.shape) != expected:
            raise ValueError(
                f"horizon prior must have shape {expected}, got {tuple(prior.shape)}"
            )
        return prior

    def _predict_and_create_record(
        self,
        origin: int,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        ProgressiveForecastRecord,
    ]:
        """Create a read-only prediction snapshot without storing future truth."""

        if batch_x.shape[0] != 1:
            raise ValueError("_predict_and_create_record accepts one origin at a time")
        x = batch_x.float().to(self.device)
        x_mark = batch_x_mark.float().to(self.device)
        full_y = batch_y.float().to(self.device)
        f_dim = -1 if self.args.features == "MS" else 0
        full_y = full_y[:, -self.args.pred_len :, f_dim:]
        expected_y = (1, self.args.pred_len, self.args.c_out)
        if tuple(full_y.shape) != expected_y:
            raise ValueError(
                f"offline evaluation target must have shape {expected_y}, got "
                f"{tuple(full_y.shape)}"
            )
        true = rearrange(full_y, "b t d -> b (t d)")

        with torch.no_grad(), self._fsnet_state_updates(False):
            prior = self.model._compute_prior(x, x_mark)
            horizon_prior = self._horizon_prior(prior)
            effective_weights = (
                self.routing_correction.effective_weights(
                    horizon_prior, top_k=self.model.top_k
                )
                if self.online_correction_enabled
                else self.model._sparsify_prior(horizon_prior)
            )
            outputs, representations = self.model.forward_experts(
                x, x_mark, return_repr=True
            )
            capability_sketch = self.model.compute_capability_sketch(
                representations
            )
            prediction, _ = self._safe_prediction(effective_weights, outputs)

        expert_hce = outputs.reshape(
            1,
            self.model.num_experts,
            self.args.pred_len,
            self.args.c_out,
        ).permute(0, 2, 3, 1)[0]
        prior_hce = horizon_prior[0]
        weights_hce = effective_weights[0]
        mixture_hc = prediction.reshape(
            1, self.args.pred_len, self.args.c_out
        )[0]
        expected_hce = (
            self.args.pred_len,
            self.args.c_out,
            self.model.num_experts,
        )
        assert tuple(expert_hce.shape) == expected_hce
        assert tuple(prior_hce.shape) == expected_hce
        assert tuple(weights_hce.shape) == expected_hce
        checker = getattr(self, "online_checker", None)
        if checker is not None:
            checker.prediction_bundle(
                prediction,
                prior_hce,
                weights_hce,
                self.routing_correction.z,
                origin,
            )

        record = ProgressiveForecastRecord(
            origin=origin,
            x=batch_x,
            x_mark=batch_x_mark,
            expert_predictions=expert_hce,
            router_prior=prior_hce,
            router_weights=weights_hce,
            mixture_prediction=mixture_hc,
            capability_sketch=capability_sketch[0],
        )
        record.metadata["expert_update_count_at_prediction"] = int(
            self.expert_update_count
        )
        if checker is not None:
            checker.new_record(record, origin)
        return (
            prediction,
            true,
            expert_hce,
            prior_hce,
            weights_hce,
            mixture_hc,
            record,
        )

    def _update_router_from_partial_feedback(
        self, events: List[ProgressiveFeedbackEvent]
    ) -> dict[str, float]:
        """One batched Router-only update over positions maturing now."""

        if not events:
            return {}
        x = torch.cat([event.record.x for event in events], dim=0).float().to(
            self.device
        )
        x_mark = torch.cat(
            [event.record.x_mark for event in events], dim=0
        ).float().to(self.device)
        horizon_indices = torch.tensor(
            [event.horizon_index for event in events],
            dtype=torch.long,
            device=self.device,
        )
        batch_indices = torch.arange(len(events), device=self.device)
        expert_prediction = torch.stack(
            [
                event.record.expert_predictions[event.horizon_index]
                for event in events
            ],
            dim=0,
        ).to(self.device)
        target = torch.stack([event.target for event in events], dim=0).to(
            self.device
        )
        local_responsibility = torch.stack(
            [event.local_responsibility for event in events], dim=0
        ).to(self.device)
        local_confidence = torch.stack(
            [event.local_confidence for event in events], dim=0
        ).to(self.device)
        correction = (
            torch.stack(
                [
                    self.routing_correction.z[event.horizon_index]
                    for event in events
                ],
                dim=0,
            )
            if getattr(self, "online_correction_enabled", True)
            else torch.zeros_like(expert_prediction)
        )

        self.opt_router.zero_grad()
        self.opt_expert.zero_grad()
        with self._fsnet_state_updates(False):
            current_prior_all = self._horizon_prior(
                self.model._compute_prior(x, x_mark)
            )
            current_prior = current_prior_all[batch_indices, horizon_indices]
            loss_router, components = partial_router_objective(
                current_prior=current_prior,
                correction=correction,
                expert_prediction=expert_prediction,
                target=target,
                local_responsibility=local_responsibility,
                local_confidence=local_confidence,
                local_credit_weight=self.local_credit_weight,
                entropy_weight=self.router_entropy_weight,
                eps=self.min_credit_eps,
                top_k=getattr(self.model, "top_k", self.model.num_experts),
            )
        loss_router.backward()
        checker = getattr(self, "online_checker", None)
        if checker is not None:
            checker.gradients(
                self.router_params, self.progressive_origin, "Router"
            )
        router_grad_norm = nn.utils.clip_grad_norm_(
            self.router_params, self.router_grad_clip
        )
        if torch.isfinite(router_grad_norm):
            self.opt_router.step()
        else:
            print(
                "[ONLINE] skipped non-finite partial Router update at step",
                self.online_step,
            )
        self.opt_router.zero_grad()
        self.opt_expert.zero_grad()
        return {
            name: float(value.detach().item())
            for name, value in {
                "loss": loss_router,
                "grad_norm": router_grad_norm,
                **components,
            }.items()
        }

    def _evaluate_completed_record(
        self, record: ProgressiveForecastRecord, current_origin: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compare prediction-time ownership with the current Expert version."""

        x = record.x.float().to(self.device)
        x_mark = record.x_mark.float().to(self.device)
        target = record.matured_targets.float().to(self.device)
        sample_confidence = float(record.sample_confidence)
        if not math.isfinite(sample_confidence):
            raise FloatingPointError("sample confidence contains NaN or Inf")
        if not 0.0 <= sample_confidence <= 1.0:
            raise ValueError("sample confidence must be in [0,1]")
        prediction_expert = record.expert_predictions.float().to(self.device)
        prediction_mixture = record.mixture_prediction.float().to(self.device)
        oracle = compute_router_oracle_diagnostics(
            expert_predictions=prediction_expert,
            mixture_prediction=prediction_mixture,
            target=target,
            eps=self.min_credit_eps,
        )
        prediction_expert_mse = oracle["expert_mse"].to(self.device)
        prediction_mixture_mse = float(oracle["router_mse"])
        best_prediction_expert_mse = float(oracle["oracle_hard_mse"])
        router_gap = float(oracle["gap_to_hard_oracle"])
        with torch.no_grad(), self._fsnet_state_updates(False):
            current_outputs, current_representations = self.model.forward_experts(
                x, x_mark, return_repr=True
            )
            sketch_now = self.model.compute_capability_sketch(
                current_representations
            )[0]
        current_prediction = current_outputs.reshape(
            1,
            self.model.num_experts,
            self.args.pred_len,
            self.args.c_out,
        ).permute(0, 2, 3, 1)[0]
        current_squared_error = (
            current_prediction - target.unsqueeze(-1)
        ).pow(2)
        current_credit = compute_sample_credit(
            accumulated_loss=current_squared_error.sum(dim=(0, 1)),
            num_matured_values=self.args.pred_len * self.args.c_out,
            matured_horizons=self.args.pred_len,
            total_horizons=self.args.pred_len,
            temperature=self.sample_credit_temperature,
            eps=self.min_credit_eps,
        )
        if record.capability_sketch is None:
            raise RuntimeError(
                "completed progressive record has no capability sketch"
            )
        sketch_pred = record.capability_sketch.float().to(self.device)
        cosine = (sketch_pred * sketch_now).sum(dim=-1).clamp(-1.0, 1.0)
        alignment = ((cosine + 1.0) / 2.0).clamp(0.0, 1.0)
        checker = getattr(self, "online_checker", None)
        if checker is not None:
            checker.alignment(alignment, current_origin)
        sketch_l2 = torch.linalg.vector_norm(
            sketch_pred - sketch_now, ord=2, dim=-1
        )
        prediction_responsibility = record.sample_responsibility.float().to(
            self.device
        )
        js_divergence = jensen_shannon_divergence(
            prediction_responsibility,
            current_credit.responsibility,
            eps=self.min_credit_eps,
        )
        ranking_reversal = bool(
            prediction_responsibility.argmax().item()
            != current_credit.responsibility.argmax().item()
        )
        prediction_update_count = int(
            record.metadata.get(
                "expert_update_count_at_prediction", self.expert_update_count
            )
        )
        expert_update_delta = self.expert_update_count - prediction_update_count
        if expert_update_delta < 0:
            raise RuntimeError("Expert update counter moved backwards")
        current_expert_mse = current_squared_error.mean(dim=(0, 1))
        decisions = [
            classify_capability_evolution(
                alignment=float(alignment[expert_id].item()),
                prediction_loss=float(
                    prediction_expert_mse[expert_id].item()
                ),
                current_loss=float(current_expert_mse[expert_id].item()),
                alignment_threshold=float(
                    getattr(self, "alignment_threshold", 0.8)
                ),
                degradation_margin=float(
                    getattr(self, "recovery_degradation_margin", 0.0)
                ),
                eps=self.min_credit_eps,
            )
            for expert_id in range(self.model.num_experts)
        ]
        category_codes = {
            "retained": 0,
            "harmful_drift": 1,
            "beneficial_or_neutral_evolution": 2,
        }
        evolution_categories = [decision.category for decision in decisions]
        relative_degradation = [
            decision.relative_degradation for decision in decisions
        ]
        reference_capability_loss = (
            prediction_expert_mse.detach().cpu().tolist()
        )
        original_relative_degradation = list(relative_degradation)
        reference_relative_degradation = list(relative_degradation)
        degraded_flags = [decision.degraded for decision in decisions]
        harmful_flags = [
            decision.category == "harmful_drift" for decision in decisions
        ]
        beneficial_flags = [
            decision.category == "beneficial_or_neutral_evolution"
            for decision in decisions
        ]
        evolution_codes = [
            category_codes[decision.category] for decision in decisions
        ]
        alignment_values = alignment.detach().cpu().tolist()
        diagnostic = {
            "origin": record.origin,
            "prediction_responsibility": (
                prediction_responsibility.detach().cpu().tolist()
            ),
            "current_responsibility": (
                current_credit.responsibility.detach().cpu().tolist()
            ),
            "prediction_expert_mse": (
                prediction_expert_mse.detach().cpu().tolist()
            ),
            "current_expert_mse": current_expert_mse.detach().cpu().tolist(),
            "reference_capability_loss": reference_capability_loss,
            "relative_capability_degradation": relative_degradation,
            "original_relative_capability_degradation": (
                original_relative_degradation
            ),
            "reference_relative_capability_degradation": (
                reference_relative_degradation
            ),
            "capability_degraded": degraded_flags,
            "capability_evolution": evolution_categories,
            "capability_evolution_code": evolution_codes,
            "harmful_drift": harmful_flags,
            "beneficial_or_neutral_evolution": beneficial_flags,
            "js_divergence": float(js_divergence.item()),
            "ranking_reversal": ranking_reversal,
            "capability_alignment_existing": alignment_values,
            "capability_alignment": alignment_values,
            "capability_l2_distance": sketch_l2.detach().cpu().tolist(),
            "sample_confidence": sample_confidence,
            "horizon_delay": int(current_origin - record.origin),
            "pred_len": int(self.args.pred_len),
            "expert_update_delta": int(expert_update_delta),
            "global_expert_update_count": int(self.expert_update_count),
            "prediction_mixture_mse": prediction_mixture_mse,
            "best_prediction_expert_mse": best_prediction_expert_mse,
            "router_gap": router_gap,
            "oracle_hard_expert_id": int(oracle["oracle_hard_expert_id"]),
            "oracle_hard_mse": float(oracle["oracle_hard_mse"]),
            "oracle_top2_mse": float(oracle["oracle_top2_mse"]),
            "oracle_top2_pair": list(oracle["oracle_top2_pair"]),
            "oracle_top2_alpha": float(oracle["oracle_top2_alpha"]),
            "oracle_all_expert_mse": float(
                oracle["oracle_all_expert_mse"]
            ),
            "oracle_all_expert_weights": oracle[
                "oracle_all_expert_weights"
            ].tolist(),
            "router_mse": float(oracle["router_mse"]),
            "gap_to_hard_oracle": float(oracle["gap_to_hard_oracle"]),
            "gap_to_top2_oracle": float(oracle["gap_to_top2_oracle"]),
            "gap_to_all_expert_oracle": float(
                oracle["gap_to_all_expert_oracle"]
            ),
        }
        self.credit_diagnostics.append(diagnostic)
        self.credit_diagnostic_total_count += 1
        record.metadata["version_diagnostic"] = diagnostic
        record.metadata["current_capability_sketch"] = (
            sketch_now.detach().cpu().clone()
        )
        self.diagnostics.increment(
            harmful_drift_count=sum(harmful_flags),
            beneficial_evolution_count=sum(beneficial_flags),
        )
        sample_entropy = -(
            prediction_responsibility.clamp_min(self.min_credit_eps)
            * prediction_responsibility.clamp_min(self.min_credit_eps).log()
        ).sum()
        self.diagnostics.update(
            sample_responsibility_entropy=float(sample_entropy.item()),
            sample_confidence=sample_confidence,
            responsibility_js_divergence=float(js_divergence.item()),
            ranking_reversal=float(ranking_reversal),
            capability_alignment_existing=alignment.detach().cpu().numpy(),
            capability_alignment=alignment.detach().cpu().numpy(),
            relative_capability_degradation=np.asarray(
                relative_degradation, dtype=np.float64
            ),
            original_relative_capability_degradation=np.asarray(
                original_relative_degradation, dtype=np.float64
            ),
            reference_relative_capability_degradation=np.asarray(
                reference_relative_degradation, dtype=np.float64
            ),
            reference_capability_loss=np.asarray(
                reference_capability_loss, dtype=np.float64
            ),
            harmful_drift=np.asarray(harmful_flags, dtype=np.float64),
            beneficial_or_neutral_evolution=np.asarray(
                beneficial_flags, dtype=np.float64
            ),
            router_gap=router_gap,
        )
        return (
            alignment.detach(),
            sketch_l2.detach(),
            current_credit.responsibility.detach(),
            prediction_expert_mse.detach(),
        )

    def _build_memory_candidates(
        self,
        record: ProgressiveForecastRecord,
        alignment: torch.Tensor,
        prediction_loss: torch.Tensor,
        timestamp: int,
        current_loss: Optional[torch.Tensor] = None,
        current_sketch: Optional[torch.Tensor] = None,
    ) -> List[VersionedMemoryItem]:
        """Classify a completed record without mutating memory yet."""

        responsibility = record.sample_responsibility.float()
        prediction_loss = prediction_loss.float()
        current_loss = (
            prediction_loss if current_loss is None else current_loss.float()
        )
        if prediction_loss.shape != alignment.shape:
            raise ValueError(
                "prediction loss and alignment must have matching Expert shapes"
            )
        if current_loss.shape != alignment.shape:
            raise ValueError(
                "current loss and alignment must have matching Expert shapes"
            )
        if record.capability_sketch is None:
            raise RuntimeError("completed progressive record has no capability sketch")
        if current_sketch is None:
            current_sketch = getattr(record, "metadata", {}).get(
                "current_capability_sketch"
            )
        if current_sketch is not None:
            if not isinstance(current_sketch, torch.Tensor):
                raise TypeError("current capability sketch must be a tensor")
            if current_sketch.shape != record.capability_sketch.shape:
                raise ValueError(
                    "current and prediction capability sketches must have "
                    "matching shapes"
                )
            current_sketch = current_sketch.detach()

        sample_confidence = float(record.sample_confidence)
        if not math.isfinite(sample_confidence):
            raise FloatingPointError("sample confidence contains NaN or Inf")
        if not 0.0 <= sample_confidence <= 1.0:
            raise ValueError("sample confidence must be in [0,1]")
        top_k = torch.topk(
            responsibility, k=self.credit_top_k
        ).indices.tolist()
        candidates: List[VersionedMemoryItem] = []
        direction_awareness_enabled = bool(
            getattr(
                self,
                "direction_awareness_enabled",
                getattr(self, "directional_recovery_enabled", False),
            )
            and getattr(self, "version_awareness_enabled", True)
        )
        for expert_id in top_k:
            expert_responsibility = float(responsibility[expert_id].item())
            if expert_responsibility < self.responsibility_threshold:
                continue
            observed_alignment = float(alignment[expert_id].item())
            original_loss = float(prediction_loss[expert_id].item())
            current_expert_loss = float(current_loss[expert_id].item())
            decision = classify_capability_evolution(
                alignment=observed_alignment,
                prediction_loss=original_loss,
                current_loss=current_expert_loss,
                alignment_threshold=self.alignment_threshold,
                degradation_margin=float(
                    getattr(self, "recovery_degradation_margin", 0.0)
                ),
                eps=float(getattr(self, "min_credit_eps", 1e-8)),
            )
            effective_alignment = (
                observed_alignment
                if self.version_awareness_enabled
                else 1.0
            )
            low_alignment = (
                self.version_awareness_enabled
                and effective_alignment < self.alignment_threshold
            )
            recovery_eligible = bool(
                self.recovery_enabled
                and (
                    not direction_awareness_enabled
                    or (
                        low_alignment
                        and decision.category == "harmful_drift"
                    )
                )
            )
            beneficial_rebase = bool(
                low_alignment
                and direction_awareness_enabled
                and decision.category
                == "beneficial_or_neutral_evolution"
            )
            if (
                low_alignment
                and not self.recovery_enabled
                and not beneficial_rebase
            ):
                continue
            if beneficial_rebase:
                diagnostics = getattr(self, "diagnostics", None)
                if diagnostics is not None:
                    diagnostics.increment(
                        recovery_skipped_non_degraded_count=1
                    )
                if current_sketch is None:
                    continue
                reference_sketch = current_sketch[expert_id]
                reference_loss = current_expert_loss
                admission_alignment = 1.0
            else:
                reference_sketch = record.capability_sketch[expert_id]
                reference_loss = original_loss
                admission_alignment = effective_alignment

            candidates.append(
                VersionedMemoryItem(
                    sample_id=record.origin,
                    origin=record.origin,
                    expert_id=expert_id,
                    x=record.x,
                    x_mark=record.x_mark,
                    target=record.matured_targets.unsqueeze(0),
                    prediction_capability_sketch=reference_sketch,
                    normalized_sketch=reference_sketch,
                    sample_responsibility=expert_responsibility,
                    sample_confidence=sample_confidence,
                    last_alignment=admission_alignment,
                    stable_credit=(
                        sample_confidence
                        * expert_responsibility
                        * admission_alignment
                    ),
                    recovery_credit=(
                        sample_confidence
                        * expert_responsibility
                        * (1.0 - admission_alignment)
                    ),
                    timestamp=timestamp,
                    prediction_time_loss=original_loss,
                    reference_capability_loss=reference_loss,
                    recent_prediction_loss=current_expert_loss,
                    last_observed_alignment=observed_alignment,
                    recovery_eligible=recovery_eligible,
                    capability_evolution=decision.category,
                )
            )
        return candidates

    def _commit_memory_candidates(
        self, candidates: List[VersionedMemoryItem]
    ) -> None:
        recovery_capacity_evictions = 0
        directional_admissions = 0
        rebased_stable_admissions = 0
        for item in candidates:
            kind, result = self.memory_manager.add_candidate_with_result(item)
            if (
                kind == "recovery"
                and getattr(
                    self.memory_manager,
                    "directional_recovery_enabled",
                    False,
                )
                and item.capability_evolution == "harmful_drift"
            ):
                directional_admissions += 1
            if (
                kind == "stable"
                and getattr(
                    self.memory_manager,
                    "directional_recovery_enabled",
                    False,
                )
                and item.capability_evolution
                == "beneficial_or_neutral_evolution"
            ):
                rebased_stable_admissions += 1
            if kind == "recovery" and result.reason == "replaced":
                recovery_capacity_evictions += 1
        if directional_admissions:
            self.diagnostics.increment(
                directional_recovery_admission_count=directional_admissions
            )
        if rebased_stable_admissions:
            self.diagnostics.increment(
                capability_rebase_count=rebased_stable_admissions
            )
        if recovery_capacity_evictions:
            self.diagnostics.increment(
                recovery_evicted=recovery_capacity_evictions,
                drop_count=recovery_capacity_evictions,
            )
        self._update_memory_diagnostics()

    def _transport_credit_to_memory(
        self,
        record: ProgressiveForecastRecord,
        alignment: torch.Tensor,
        prediction_loss: torch.Tensor,
        timestamp: int,
    ) -> None:
        """Compatibility wrapper for callers that do not need deferred commit."""

        candidates = self._build_memory_candidates(
            record, alignment, prediction_loss, timestamp
        )
        self._commit_memory_candidates(candidates)

    def _refresh_expert_memory(self, timestamp: int) -> dict[str, int]:
        unique_items = {}
        for item in self.memory_manager.all_items():
            unique_items.setdefault(item.sample_id, item)
        if not unique_items:
            return {
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

        sample_ids = list(unique_items)
        x = torch.cat([unique_items[key].x for key in sample_ids], dim=0)
        x_mark = torch.cat(
            [unique_items[key].x_mark for key in sample_ids], dim=0
        )
        with torch.no_grad(), self._fsnet_state_updates(False):
            outputs, representations = self.model.forward_experts(
                x.float().to(self.device),
                x_mark.float().to(self.device),
                return_repr=True,
            )
            sketches = self.model.compute_capability_sketch(representations)
        predictions = outputs.reshape(
            len(sample_ids),
            self.model.num_experts,
            self.args.pred_len,
            self.args.c_out,
        )
        sample_index = {
            sample_id: index for index, sample_id in enumerate(sample_ids)
        }

        def evaluator(
            expert_id: int, item: VersionedMemoryItem
        ) -> tuple[float, float]:
            index = sample_index[item.sample_id]
            current_sketch = sketches[index, expert_id]
            old_sketch = item.normalized_sketch.float().to(self.device)
            alignment = (
                (
                    (
                        torch.dot(old_sketch, current_sketch)
                        .clamp(-1.0, 1.0)
                        + 1.0
                    )
                    / 2.0
                ).clamp(0.0, 1.0)
            )
            target = item.target.float().to(self.device).reshape(
                self.args.pred_len, self.args.c_out
            )
            prediction = predictions[index, expert_id]
            loss = (prediction - target).pow(2).mean()
            return (
                float(alignment.item()),
                float(loss.item()),
                current_sketch.detach(),
            )

        return self.memory_manager.refresh(
            evaluator, timestamp=timestamp, count_recovery_attempts=False
        )

    def _refresh_subspaces(self, step: int) -> None:
        """Refresh per-Expert bases from current read-only Stable features."""

        for expert_id, buffer in enumerate(self.memory_manager.stable_buffers):
            items = buffer.items
            if not items:
                self.subspace_protector.deactivate_protection(
                    expert_id, step=step
                )
                continue
            x = torch.cat([item.x for item in items], dim=0).float().to(self.device)
            x_mark = (
                torch.cat([item.x_mark for item in items], dim=0)
                .float()
                .to(self.device)
            )
            expert = self.model.experts[expert_id]
            with torch.no_grad(), self._fsnet_state_updates(False):
                head_features = expert.extract_head_features(x.clone(), x_mark)
            if self.credit_weighted_subspace:
                sample_weights = torch.tensor(
                    [
                        item.stable_credit
                        for item in items
                    ],
                    dtype=head_features.dtype,
                    device=head_features.device,
                )
            else:
                sample_weights = torch.ones(
                    len(items),
                    dtype=head_features.dtype,
                    device=head_features.device,
                )
            if head_features.ndim == 2:
                features = head_features
                observation_weights = sample_weights
            elif head_features.ndim == 3:
                channels = head_features.shape[1]
                features = head_features.reshape(-1, head_features.shape[-1])
                observation_weights = (
                    sample_weights.unsqueeze(1)
                    .expand(-1, channels)
                    .reshape(-1)
                    / float(channels)
                )
            else:
                raise ValueError("head features must be [B,320] or [B,C,320]")
            stable_evidence_mass = sum(item.stable_credit for item in items)
            current_lr = float(
                self.opt_expert.param_groups[0]["lr"]
                if hasattr(self, "opt_expert")
                and self.opt_expert.param_groups
                else getattr(self, "base_learning_rate_expert", 0.0)
            )
            self.subspace_protector.refresh_expert(
                expert_id=expert_id,
                features=features,
                observation_weights=observation_weights,
                sample_count=len(items),
                step=step,
                stable_evidence_mass=stable_evidence_mass,
                current_lr=current_lr,
            )
        self.diagnostics.update(**self.subspace_protector.metrics())

    def _update_memory_diagnostics(self) -> None:
        stable_sizes, recovery_sizes = self.memory_manager.buffer_sizes()
        self.diagnostics.update(
            stable_buffer_size=stable_sizes,
            recovery_buffer_size=recovery_sizes,
        )

    def _periodic_memory_and_subspace_refresh(self, timestamp: int) -> None:
        memory_due = (
            self.memory_refresh_interval > 0
            and self.completed_record_count % self.memory_refresh_interval == 0
        )
        subspace_due = (
            self.subspace_refresh_interval > 0
            and self.completed_record_count % self.subspace_refresh_interval == 0
        )
        if memory_due or subspace_due:
            stats = self._refresh_expert_memory(timestamp)
            protection_deactivated = False
            for expert_id, stable_buffer in enumerate(
                self.memory_manager.stable_buffers
            ):
                if len(stable_buffer) == 0:
                    self.subspace_protector.deactivate_protection(
                        expert_id, step=timestamp
                    )
                    protection_deactivated = True
            if protection_deactivated:
                self.diagnostics.update(**self.subspace_protector.metrics())
            self.diagnostics.increment(
                demotion_count=stats["stable_to_recovery"],
                promotion_count=stats["recovery_to_stable"],
                drop_count=stats["evicted"],
                recovery_to_stable=stats.get("recovery_to_stable", 0),
                recovery_dropped_after_success=stats.get(
                    "recovery_dropped_after_success", 0
                ),
                recovery_failed=stats.get("recovery_failed", 0),
                recovery_evicted=stats.get("recovery_evicted", 0),
                recovery_attempt_exhausted=stats.get(
                    "recovery_attempt_exhausted", 0
                ),
                directional_recovery_admission_count=(
                    stats.get("stable_to_recovery", 0)
                    if getattr(
                        self, "directional_recovery_enabled", False
                    )
                    else 0
                ),
                harmful_drift_count=stats.get("harmful_drift", 0),
                beneficial_evolution_count=stats.get(
                    "beneficial_evolution", 0
                ),
                recovery_skipped_non_degraded_count=stats.get(
                    "recovery_skipped_non_degraded", 0
                ),
                performance_recovery_count=stats.get(
                    "performance_recovery", 0
                ),
                capability_rebase_count=stats.get(
                    "capability_rebase", 0
                ),
            )
        if subspace_due:
            self._refresh_subspaces(step=timestamp)
        self._update_memory_diagnostics()
        checker = getattr(self, "online_checker", None)
        if checker is not None:
            checker.memory(self.memory_manager, timestamp)
            checker.subspaces(self.subspace_protector, timestamp)

    def _router_only_completed_update(
        self,
        completed: ProgressiveForecastRecord,
    ) -> None:
        """Use complete feedback for the Router while freezing all Experts."""

        x = completed.x.float().to(self.device)
        x_mark = completed.x_mark.float().to(self.device)
        target = completed.matured_targets.unsqueeze(0).float().to(self.device)
        true = rearrange(target, "b t d -> b (t d)")
        self._set_optimizer_lr(
            self.opt_router, self.base_learning_rate_router
        )
        self._run_full_router_update(x, x_mark, true)
        self.online_step += 1

    def _learn_then_commit_completed_record(
        self,
        dataset,
        completed: ProgressiveForecastRecord,
        origin: int,
    ) -> None:
        """Learn with the old basis, then make the sample future evidence."""

        checker = getattr(self, "online_checker", None)
        if checker is not None:
            checker.begin_completed(completed.origin, origin)
        alignment, _, _, prediction_loss = self._evaluate_completed_record(
            completed, origin
        )
        version_diagnostic = getattr(completed, "metadata", {}).get(
            "version_diagnostic"
        )
        if version_diagnostic is None:
            candidates = self._build_memory_candidates(
                completed, alignment, prediction_loss, timestamp=origin
            )
        else:
            current_loss = torch.tensor(
                version_diagnostic["current_expert_mse"],
                dtype=prediction_loss.dtype,
                device=prediction_loss.device,
            )
            candidates = self._build_memory_candidates(
                completed,
                alignment,
                prediction_loss,
                timestamp=origin,
                current_loss=current_loss,
                current_sketch=completed.metadata.get(
                    "current_capability_sketch"
                ),
            )
        reconstructed_target = completed.matured_targets.unsqueeze(0)
        # The current sample learns through the old basis first.  Only after
        # the optimizer step may it enter memory and influence a future basis.
        if getattr(self, "expert_online_update_enabled", True):
            self._ol_one_batch(
                dataset,
                completed.x,
                reconstructed_target,
                completed.x_mark,
            )
        else:
            self._router_only_completed_update(completed)
        if checker is not None:
            checker.expert_updated(completed.origin, origin)
            checker.before_memory_commit(completed.origin, origin)
        self._commit_memory_candidates(candidates)
        self.completed_record_count += 1
        self._periodic_memory_and_subspace_refresh(timestamp=origin)

    def _update_feedback_diagnostics(
        self, events: List[ProgressiveFeedbackEvent]
    ) -> None:
        if not events:
            return
        responsibilities = torch.stack(
            [event.local_responsibility for event in events], dim=0
        ).float()
        entropy = -(
            responsibilities.clamp_min(self.min_credit_eps)
            * responsibilities.clamp_min(self.min_credit_eps).log()
        ).sum(dim=-1).mean()
        self.diagnostics.update(
            local_responsibility_entropy=float(entropy.item()),
            z_norm=float(torch.linalg.vector_norm(self.routing_correction.z).item()),
        )

    def _update_prediction_diagnostics(
        self,
        true: torch.Tensor,
        expert_hce: torch.Tensor,
        prior_hce: torch.Tensor,
        weights_hce: torch.Tensor,
        mixture_hc: torch.Tensor,
    ) -> None:
        """Record evaluation-only metrics; ``true`` never affects an update."""

        target = true.reshape(self.args.pred_len, self.args.c_out).to(expert_hce)
        specialization = getattr(self, "specialization_diagnostics", None)
        if specialization is not None:
            specialization.update(
                router_weights=weights_hce,
                expert_predictions=expert_hce,
                target=target,
            )
        comparator = getattr(self, "comparator_diagnostics", None)
        if comparator is not None:
            comparator.update(
                expert_predictions=expert_hce,
                router_prediction=mixture_hc,
                target=target,
            )
        dynamic_comparator = getattr(
            self, "dynamic_comparator_diagnostics", None
        )
        if dynamic_comparator is not None:
            dynamic_comparator.update(
                expert_predictions=expert_hce,
                router_prediction=mixture_hc,
                target=target,
            )
        expert_mse = (expert_hce - target.unsqueeze(-1)).pow(2).mean(dim=(0, 1))
        mixture_mse = (mixture_hc - target).pow(2).mean()
        prior_entropy = -(
            prior_hce.clamp_min(self.min_credit_eps)
            * prior_hce.clamp_min(self.min_credit_eps).log()
        ).sum(dim=-1).mean()
        effective_entropy = -(
            weights_hce.clamp_min(self.min_credit_eps)
            * weights_hce.clamp_min(self.min_credit_eps).log()
        ).sum(dim=-1).mean()
        self.diagnostics.update(
            online_mse=float(mixture_mse.item()),
            raw_prior_entropy=float(prior_entropy.item()),
            effective_routing_entropy=float(effective_entropy.item()),
            z_norm=float(torch.linalg.vector_norm(self.routing_correction.z).item()),
            expert_weights=weights_hce.mean(dim=(0, 1)).detach().cpu().numpy(),
            expert_independent_mse=expert_mse.detach().cpu().numpy(),
            router_gap=float((mixture_mse - expert_mse.min()).item()),
        )

    def _progressive_online_batch(
        self,
        dataset,
        feedback_manager: ProgressiveFeedbackManager,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance rolling origins without exposing an unmatured target."""

        del batch_y_mark
        batch_preds: List[torch.Tensor] = []
        batch_trues: List[torch.Tensor] = []
        for i in range(batch_x.shape[0]):
            origin = self.progressive_origin
            x_t = batch_x[i : i + 1]
            x_mark_t = batch_x_mark[i : i + 1]

            if self.online_correction_enabled:
                self.routing_correction.begin_origin(origin)
            observation = self._current_observation(x_t)
            events, completed_records = feedback_manager.release(
                origin, observation
            )
            if self.online_correction_enabled:
                for event in events:
                    self.routing_correction.update(
                        horizon_index=event.horizon_index,
                        expert_prediction=event.record.expert_predictions[
                            event.horizon_index
                        ],
                        mixture_prediction=event.record.mixture_prediction[
                            event.horizon_index
                        ],
                        target=event.target,
                    )
            checker = getattr(self, "online_checker", None)
            if checker is not None:
                checker.feedback(events, origin)
            self._update_feedback_diagnostics(events)
            self._update_router_from_partial_feedback(events)

            for completed in completed_records:
                self._learn_then_commit_completed_record(
                    dataset, completed, origin
                )

            (
                pred_t,
                true_t,
                expert_hce,
                prior_hce,
                weights_hce,
                mixture_hc,
                record,
            ) = self._predict_and_create_record(
                origin,
                x_t,
                batch_y[i : i + 1],
                x_mark_t,
            )
            self._update_prediction_diagnostics(
                true_t[0], expert_hce, prior_hce, weights_hce, mixture_hc
            )
            feedback_manager.add_record(record)
            batch_preds.append(pred_t)
            batch_trues.append(true_t)
            self.diagnostics.maybe_record(origin)
            self.progressive_origin += 1

        return torch.cat(batch_preds, dim=0), torch.cat(batch_trues, dim=0)

    def _reset_progressive_online_state(
        self, feedback_manager: ProgressiveFeedbackManager | None
    ) -> None:
        """Reset mutable stream state without changing weights or optimizer state."""

        self.online_buffer.clear()
        self.online_step = 0
        self.prev_online_mse = None
        self.online_mse_ema = None
        self.fallback_count = 0
        self.fallback_channel_count = 0
        if feedback_manager is not None:
            feedback_manager.reset()
        self.routing_correction.reset()
        self.memory_manager.clear()
        self.subspace_protector.reset()
        self.diagnostics.reset()
        tsb_diagnostics = getattr(self, "tsb_gradient_diagnostics", None)
        if tsb_diagnostics is not None:
            tsb_diagnostics.reset()
        specialization = getattr(self, "specialization_diagnostics", None)
        if specialization is not None:
            specialization.reset()
        comparator = getattr(self, "comparator_diagnostics", None)
        if comparator is not None:
            comparator.reset()
        dynamic_comparator = getattr(
            self, "dynamic_comparator_diagnostics", None
        )
        if dynamic_comparator is not None:
            dynamic_comparator.reset()
        self.progressive_origin = 0
        self.credit_diagnostics.clear()
        self.credit_diagnostic_total_count = 0
        self.expert_update_count = 0
        self.completed_record_count = 0
        self.processed_online_origins = 0
        self.online_test_early_ended = False
        self.last_expert_update_diagnostics = {}
        checker = getattr(self, "online_checker", None)
        if checker is not None:
            checker.reset()
        self.opt_expert.zero_grad(set_to_none=True)
        self.opt_router.zero_grad(set_to_none=True)
        self.diagnostics.update(**self.subspace_protector.metrics())
        self._update_memory_diagnostics()

    def test(self, setting):
        self._prepare_test_start_state()
        test_data, test_loader = self._get_data(flag="test")

        self.model.eval()
        self.model.set_router_mode(True)
        if self.online == "regressor":
            for expert in self.model.experts:
                for name, parameter in expert.named_parameters():
                    if "regressor" not in name:
                        parameter.requires_grad = False

        preds = []
        trues = []
        start = time.time()
        maes, mses, rmses, mapes, mspes = [], [], [], [], []
        progressive_manager = None
        if self.progressive_fb and self.online != "none":
            progressive_manager = ProgressiveFeedbackManager(
                pred_len=self.args.pred_len,
                c_out=self.args.c_out,
                local_credit_temperature=self.local_credit_temperature,
                sample_credit_temperature=self.sample_credit_temperature,
                min_credit_eps=self.min_credit_eps,
            )
            print(
                "[PROGRESSIVE_FB] rolling origins; one newly observed timestamp "
                "is released before each prediction"
            )
        self._reset_progressive_online_state(progressive_manager)
        feedback_queue = (
            deque()
            if progressive_manager is None
            and self.args.delay_fb
            and self.online != "none"
            else None
        )
        if feedback_queue is not None:
            print("[DELAY_FB] rolling origins; feedback delay={} steps".format(self.args.pred_len))
        processed_origins = 0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(test_loader):
            if self.max_online_steps > 0:
                remaining = self.max_online_steps - processed_origins
                if remaining <= 0:
                    break
                if batch_x.shape[0] > remaining:
                    batch_x = batch_x[:remaining]
                    batch_y = batch_y[:remaining]
                    batch_x_mark = batch_x_mark[:remaining]
                    batch_y_mark = batch_y_mark[:remaining]
            if progressive_manager is not None:
                pred, true = self._progressive_online_batch(
                    test_data,
                    progressive_manager,
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                )
            elif feedback_queue is not None:
                pred, true = self._delayed_online_batch(
                    test_data, feedback_queue, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
            else:
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="test"
                )
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            processed_origins += int(pred.shape[0])
            mae, mse, rmse, mape, mspe = metric(
                pred.detach().cpu().numpy(), true.detach().cpu().numpy()
            )
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)

        self.processed_online_origins = processed_origins
        self.online_test_early_ended = processed_origins < len(test_data)
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print("test shape:", preds.shape, trues.shape)

        MAE, MSE, RMSE, MAPE, MSPE = (
            cumavg(maes),
            cumavg(mses),
            cumavg(rmses),
            cumavg(mapes),
            cumavg(mspes),
        )
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        exp_time = time.time() - start
        if progressive_manager is not None:
            final_step = max(0, self.progressive_origin - 1)
            self.diagnostics.maybe_record(final_step, force=True)
        print("mse:{}, mae:{}, time:{}".format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def save_online_diagnostics(self, result_directory: str) -> tuple[str, str]:
        """Persist interval and bounded record-level credit diagnostics."""

        npz_path, json_path = self.diagnostics.save(result_directory)
        records = list(self.credit_diagnostics)
        num_experts = (
            len(records[0]["capability_alignment"])
            if records
            else int(getattr(getattr(self, "model", None), "num_experts", 0))
        )

        def expert_field(
            name: str, fallback: str | None = None
        ) -> np.ndarray:
            if not records:
                return np.empty((0, num_experts), dtype=np.float64)
            def value(record: dict[str, Any]):
                if name in record:
                    return record[name]
                if fallback is not None:
                    return record[fallback]
                raise KeyError(name)

            values = np.asarray(
                [value(record) for record in records],
                dtype=np.float64,
            )
            expected = (len(records), num_experts)
            if values.shape != expected:
                raise ValueError(
                    f"credit diagnostic {name} must have shape {expected}, "
                    f"got {values.shape}"
                )
            return values

        def direction_field(name: str) -> np.ndarray:
            """Load direction diagnostics or derive them for old records."""

            if not records:
                return np.empty((0, num_experts), dtype=np.float64)
            values = []
            margin = float(
                getattr(self, "recovery_degradation_margin", 0.0)
            )
            eps = float(getattr(self, "min_credit_eps", 1e-8))
            threshold = float(getattr(self, "alignment_threshold", 0.8))
            for record in records:
                if name in record:
                    values.append(record[name])
                    continue
                prediction = np.asarray(
                    record["prediction_expert_mse"], dtype=np.float64
                )
                current = np.asarray(
                    record["current_expert_mse"], dtype=np.float64
                )
                reference = np.asarray(
                    record.get(
                        "reference_capability_loss", prediction
                    ),
                    dtype=np.float64,
                )
                alignment = np.asarray(
                    record.get(
                        "capability_alignment_existing",
                        record["capability_alignment"],
                    ),
                    dtype=np.float64,
                )
                original_relative = (current - prediction) / (prediction + eps)
                reference_relative = (current - reference) / (reference + eps)
                degraded = current > reference * (1.0 + margin) + eps
                harmful = (alignment < threshold) & degraded
                beneficial = (alignment < threshold) & ~degraded
                derived = {
                    "relative_capability_degradation": original_relative,
                    "original_relative_capability_degradation": (
                        original_relative
                    ),
                    "reference_relative_capability_degradation": (
                        reference_relative
                    ),
                    "capability_degraded": degraded,
                    "harmful_drift": harmful,
                    "beneficial_or_neutral_evolution": beneficial,
                    "capability_evolution_code": np.where(
                        harmful, 1, np.where(beneficial, 2, 0)
                    ),
                }
                values.append(derived[name])
            result = np.asarray(values, dtype=np.float64)
            expected = (len(records), num_experts)
            if result.shape != expected:
                raise ValueError(
                    f"credit diagnostic {name} must have shape {expected}, "
                    f"got {result.shape}"
                )
            return result

        fields = {
            "origin": np.asarray([r["origin"] for r in records], dtype=np.int64),
            "prediction_responsibility": expert_field(
                "prediction_responsibility"
            ),
            "current_responsibility": expert_field("current_responsibility"),
            "prediction_expert_mse": expert_field("prediction_expert_mse"),
            "current_expert_mse": expert_field("current_expert_mse"),
            "reference_capability_loss": expert_field(
                "reference_capability_loss", "prediction_expert_mse"
            ),
            "relative_capability_degradation": direction_field(
                "relative_capability_degradation"
            ),
            "original_relative_capability_degradation": direction_field(
                "original_relative_capability_degradation"
            ),
            "reference_relative_capability_degradation": direction_field(
                "reference_relative_capability_degradation"
            ),
            "capability_evolution_code": direction_field(
                "capability_evolution_code"
            ).astype(np.int64),
            "capability_degraded": direction_field(
                "capability_degraded"
            ).astype(np.bool_),
            "harmful_drift": direction_field("harmful_drift").astype(
                np.bool_
            ),
            "beneficial_or_neutral_evolution": direction_field(
                "beneficial_or_neutral_evolution"
            ).astype(np.bool_),
            "capability_alignment_existing": expert_field(
                "capability_alignment_existing", "capability_alignment"
            ),
            "capability_alignment": expert_field("capability_alignment"),
            "capability_l2_distance": expert_field(
                "capability_l2_distance"
            ),
            "horizon_delay": np.asarray(
                [r["horizon_delay"] for r in records], dtype=np.int64
            ),
            "pred_len": np.asarray(
                [r.get("pred_len", r["horizon_delay"]) for r in records],
                dtype=np.int64,
            ),
            "js_divergence": np.asarray(
                [r["js_divergence"] for r in records], dtype=np.float64
            ),
            "ranking_reversal": np.asarray(
                [r["ranking_reversal"] for r in records], dtype=np.bool_
            ),
            "mean_alignment": np.asarray(
                [np.mean(r["capability_alignment"]) for r in records],
                dtype=np.float64,
            ),
            "min_alignment": np.asarray(
                [np.min(r["capability_alignment"]) for r in records],
                dtype=np.float64,
            ),
            "sample_confidence": np.asarray(
                [r["sample_confidence"] for r in records], dtype=np.float64
            ),
            "expert_update_delta": np.asarray(
                [r["expert_update_delta"] for r in records], dtype=np.int64
            ),
            "global_expert_update_count": np.asarray(
                [r["global_expert_update_count"] for r in records],
                dtype=np.int64,
            ),
            "prediction_mixture_mse": np.asarray(
                [r["prediction_mixture_mse"] for r in records],
                dtype=np.float64,
            ),
            "best_prediction_expert_mse": np.asarray(
                [r["best_prediction_expert_mse"] for r in records],
                dtype=np.float64,
            ),
            "router_gap": np.asarray(
                [r["router_gap"] for r in records], dtype=np.float64
            ),
            "oracle_hard_expert_id": np.asarray(
                [r["oracle_hard_expert_id"] for r in records],
                dtype=np.int64,
            ),
            "oracle_hard_mse": np.asarray(
                [r["oracle_hard_mse"] for r in records], dtype=np.float64
            ),
            "oracle_top2_mse": np.asarray(
                [r["oracle_top2_mse"] for r in records], dtype=np.float64
            ),
            "oracle_top2_pair": np.asarray(
                [r["oracle_top2_pair"] for r in records], dtype=np.int64
            ).reshape(-1, 2),
            "oracle_top2_alpha": np.asarray(
                [r["oracle_top2_alpha"] for r in records], dtype=np.float64
            ),
            "oracle_all_expert_mse": np.asarray(
                [r["oracle_all_expert_mse"] for r in records],
                dtype=np.float64,
            ),
            "oracle_all_expert_weights": expert_field(
                "oracle_all_expert_weights"
            ),
            "router_mse": np.asarray(
                [r["router_mse"] for r in records], dtype=np.float64
            ),
            "gap_to_hard_oracle": np.asarray(
                [r["gap_to_hard_oracle"] for r in records],
                dtype=np.float64,
            ),
            "gap_to_top2_oracle": np.asarray(
                [r["gap_to_top2_oracle"] for r in records],
                dtype=np.float64,
            ),
            "gap_to_all_expert_oracle": np.asarray(
                [r["gap_to_all_expert_oracle"] for r in records],
                dtype=np.float64,
            ),
        }
        credit_path = os.path.join(result_directory, "credit_diagnostics.npz")
        np.savez_compressed(credit_path, **fields)
        with open(json_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        retained = len(records)
        summary["credit_diagnostics"] = {
            "total_records": int(self.credit_diagnostic_total_count),
            "retained_records": retained,
            "estimated_overwritten": max(
                0, int(self.credit_diagnostic_total_count) - retained
            ),
            "mean_js_divergence": (
                float(fields["js_divergence"].mean()) if retained else None
            ),
            "ranking_reversal_rate": (
                float(fields["ranking_reversal"].mean()) if retained else None
            ),
            "mean_alignment": (
                float(fields["mean_alignment"].mean()) if retained else None
            ),
            "mean_relative_capability_degradation": (
                float(
                    fields["relative_capability_degradation"].mean()
                )
                if retained
                else None
            ),
            "mean_original_relative_capability_degradation": (
                float(
                    fields[
                        "original_relative_capability_degradation"
                    ].mean()
                )
                if retained
                else None
            ),
            "mean_reference_relative_capability_degradation": (
                float(
                    fields[
                        "reference_relative_capability_degradation"
                    ].mean()
                )
                if retained
                else None
            ),
            "harmful_drift_rate": (
                float(fields["harmful_drift"].mean())
                if retained
                else None
            ),
            "beneficial_evolution_rate": (
                float(
                    fields[
                        "beneficial_or_neutral_evolution"
                    ].mean()
                )
                if retained
                else None
            ),
            "recovery_skipped_non_degraded_count": float(
                self.diagnostics.counters.get(
                    "recovery_skipped_non_degraded_count", 0.0
                )
            ),
            "performance_recovery_count": float(
                self.diagnostics.counters.get(
                    "performance_recovery_count", 0.0
                )
            ),
            "min_alignment": (
                float(fields["min_alignment"].min()) if retained else None
            ),
            "mean_router_gap": (
                float(fields["router_gap"].mean()) if retained else None
            ),
            "mean_gap_to_top2_oracle": (
                float(fields["gap_to_top2_oracle"].mean())
                if retained
                else None
            ),
            "mean_gap_to_all_expert_oracle": (
                float(fields["gap_to_all_expert_oracle"].mean())
                if retained
                else None
            ),
            "file": os.path.basename(credit_path),
        }
        tsb_diagnostics = getattr(self, "tsb_gradient_diagnostics", None)
        if tsb_diagnostics is not None:
            summary["tsb_gradient_diagnostics"] = tsb_diagnostics.metrics()
        specialization = getattr(self, "specialization_diagnostics", None)
        if specialization is not None:
            specialization_path = os.path.join(
                result_directory, "specialization_diagnostics.npz"
            )
            specialization.save(specialization_path)
            summary["specialization_diagnostics"] = {
                "num_updates": int(specialization.num_updates),
                "file": os.path.basename(specialization_path),
            }
        comparator = getattr(self, "comparator_diagnostics", None)
        if comparator is not None:
            comparator_result = evaluate_static_comparator(comparator)
            hc_statistics = comparator.horizon_channel_statistics()
            hc_result = (
                evaluate_horizon_channel_static_comparator(comparator)
                if hc_statistics is not None
                else None
            )
            dynamic_accumulator = getattr(
                self, "dynamic_comparator_diagnostics", None
            )
            dynamic_result = (
                evaluate_k_switch_dynamic_comparator(
                    dynamic_accumulator,
                    max_switches=int(
                        getattr(self, "dynamic_comparator_max_switches", 1)
                    ),
                )
                if dynamic_accumulator is not None
                else None
            )
            comparator_npz, comparator_json = save_comparator_diagnostics(
                comparator_result,
                result_directory,
                hc_result=hc_result,
                dynamic_result=dynamic_result,
            )
            comparator_summary = {
                **comparator_result.as_dict(),
                "npz_file": os.path.basename(comparator_npz),
                "json_file": os.path.basename(comparator_json),
            }
            if hc_result is not None:
                comparator_summary.update(hc_result.as_dict())
            if dynamic_result is not None:
                comparator_summary.update(dynamic_result.as_dict())
            summary["comparator_diagnostics"] = comparator_summary
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return npz_path, json_path

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="train"):
        if mode == "test":
            if self.online == "none":
                return self._predict_without_update(batch_x, batch_y, batch_x_mark)
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float()

        gates = self.model._compute_gates(x, batch_x_mark)
        expert_outputs, expert_reps = self.model.forward_experts(x, batch_x_mark, return_repr=True)
        routed_pred = self.model.aggregate_with_gates(gates, expert_outputs.detach())
        f_dim = -1 if self.args.features == "MS" else 0
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
        true = rearrange(batch_y, "b t d -> b (t d)")
        return expert_outputs, routed_pred, expert_reps, gates, true

    def _predict_without_update(self, batch_x, batch_y, batch_x_mark):
        x = batch_x.float().to(self.device)
        x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        f_dim = -1 if self.args.features == "MS" else 0
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
        true = rearrange(batch_y, "b t d -> b (t d)")
        with torch.no_grad(), self._fsnet_state_updates(False):
            gates = self.model._compute_gates(x, x_mark)
            outputs = self.model.forward_experts(x, x_mark, return_repr=False)
            prediction, _ = self._safe_prediction(gates, outputs)
        return prediction, true

    def _compute_tsb_reference_gradient(
        self,
    ) -> List[torch.Tensor] | None:
        if not self.tsb_reference_enabled or not self.online_buffer:
            return None
        x_buffer = torch.cat([item[0] for item in self.online_buffer], dim=0)
        x_mark_buffer = torch.cat(
            [item[1] for item in self.online_buffer], dim=0
        )
        y_buffer = torch.cat([item[2] for item in self.online_buffer], dim=0)
        self.opt_expert.zero_grad()
        with self._fsnet_state_updates(False):
            outputs = self.model.forward_experts(
                x_buffer, x_mark_buffer, return_repr=False
            )
            loss = self._independent_expert_mse(outputs, y_buffer)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("TSB reference loss is not finite")
        loss.backward()
        return [
            parameter.grad.detach().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter)
            for parameter in self.expert_params
        ]

    def _run_expert_optimizer_step(
        self,
        x_t: torch.Tensor,
        x_mark_t: torch.Tensor,
        y_t: torch.Tensor,
        expert_lr: float,
        tsb_alpha: float,
        recovery_batches: List[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run supervised + Recovery backward, strategy filtering and step."""

        self.opt_expert.zero_grad()
        self.opt_router.zero_grad()
        with torch.no_grad():
            gates = self.model._compute_gates(x_t, x_mark_t)
        outputs = self.model.forward_experts(x_t, x_mark_t, return_repr=False)
        prediction, unsafe_channels = self._safe_prediction(
            gates, outputs.detach()
        )
        raw_prediction = self.model.aggregate_with_gates(gates, outputs.detach())
        supervised_loss = self._independent_expert_mse(outputs, y_t)
        recovery_loss, recovery_metrics = self._compute_recovery_replay_loss(
            recovery_batches
        )
        total_loss = supervised_loss + recovery_loss
        if not bool(torch.isfinite(total_loss).item()):
            raise FloatingPointError("Expert online loss is not finite")
        total_loss.backward()
        current_gradients = [
            parameter.grad.detach().clone()
            if parameter.grad is not None
            else None
            for parameter in self.expert_params
        ]
        reference_gradients = self._compute_tsb_reference_gradient()
        self.opt_expert.zero_grad()
        strategy_metrics = self._apply_expert_update_strategy(
            current=current_gradients,
            reference=reference_gradients,
            alpha=tsb_alpha,
            current_lr=expert_lr,
        )
        tsb_aggregate_metrics = {}
        tsb_aggregator = getattr(self, "tsb_gradient_diagnostics", None)
        if (
            tsb_aggregator is not None
            and self.expert_update_strategy in {"tsb", "hybrid"}
        ):
            tsb_aggregator.update(
                grad_cosine=strategy_metrics["tsb_grad_cosine"],
                conflict=strategy_metrics["tsb_gradient_conflict"],
                modification_ratio=strategy_metrics[
                    "tsb_modification_ratio"
                ],
                projection_removal_ratio=strategy_metrics[
                    "tsb_projection_removal_ratio"
                ],
            )
            tsb_aggregate_metrics = {
                key: value
                for key, value in tsb_aggregator.metrics().items()
                if value is not None
            }

        checker = getattr(self, "online_checker", None)
        if checker is not None:
            checker.gradients(
                self.expert_params, self.progressive_origin, "Expert"
            )
        grad_norm = nn.utils.clip_grad_norm_(
            self.expert_params, self.expert_grad_clip
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise FloatingPointError("Expert gradient norm is not finite")
        self.opt_expert.step()
        self.model.store_grad()
        self.expert_update_count += 1
        self._finalize_recovery_replay(recovery_batches)
        if checker is not None:
            checker.memory(self.memory_manager, self.progressive_origin)
        self.opt_expert.zero_grad()
        self.opt_router.zero_grad()
        diagnostics = {
            **tsb_aggregate_metrics,
            **strategy_metrics,
            **recovery_metrics,
            "expert_gradient_norm": float(grad_norm.item()),
        }
        self.last_expert_update_diagnostics = diagnostics
        self.diagnostics.update(**diagnostics)
        return {
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "unsafe_channels": unsafe_channels,
            "expert_outputs": outputs.detach(),
            "expert_gradient_norm": float(grad_norm.item()),
        }

    def _run_full_router_update(
        self,
        x_t: torch.Tensor,
        x_mark_t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> dict[str, Any]:
        """Perform the legal full-feedback slow Router update."""

        self.opt_router.zero_grad()
        self.opt_expert.zero_grad()
        with torch.no_grad(), self._fsnet_state_updates(False):
            outputs = self.model.forward_experts(
                x_t, x_mark_t, return_repr=False
            ).detach()
        dense_prior = self.model._compute_prior(x_t, x_mark_t)
        effective_weights = self.model._sparsify_prior(dense_prior)
        expert_hce = outputs.reshape(
            outputs.shape[0],
            self.model.num_experts,
            self.args.pred_len,
            self.args.c_out,
        ).permute(0, 2, 3, 1)
        target_hc = y_t.reshape(
            y_t.shape[0], self.args.pred_len, self.args.c_out
        )
        loss, components = full_router_objective(
            dense_prior=dense_prior,
            effective_weights=effective_weights,
            expert_prediction=expert_hce,
            target=target_hc,
            entropy_weight=self.router_entropy_weight,
            eps=self.min_credit_eps,
        )
        prediction = components["prediction"].reshape(y_t.shape)
        entropy = components["entropy"]
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("full-feedback Router loss is not finite")
        loss.backward()
        checker = getattr(self, "online_checker", None)
        if checker is not None:
            checker.gradients(
                self.router_params, self.progressive_origin, "Router"
            )
        grad_norm = nn.utils.clip_grad_norm_(
            self.router_params, self.router_grad_clip
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise FloatingPointError("Router gradient norm is not finite")
        self.opt_router.step()
        self.opt_router.zero_grad()
        self.opt_expert.zero_grad()
        return {
            "prediction": prediction.detach(),
            "entropy": float(entropy.item()),
            "gradient_norm": float(grad_norm.item()),
        }

    def _log_compact_online_update(
        self,
        expert_result: dict[str, Any],
        router_result: dict[str, Any],
        y_t: torch.Tensor,
        expert_lr: float,
        router_lr: float,
    ) -> None:
        if self.online_log_interval <= 0:
            return
        if self.online_step % self.online_log_interval != 0:
            return
        mse = self._select_criterion()(
            expert_result["prediction"], y_t
        ).item()
        print(
            "[ONLINE] step={} mse={:.6f} lr_e={:.2e} lr_r={:.2e} "
            "strategy={} grad_e={:.3f} grad_r={:.3f} z_norm={:.3f}".format(
                self.online_step,
                mse,
                expert_lr,
                router_lr,
                self.expert_update_strategy,
                expert_result["expert_gradient_norm"],
                router_result["gradient_norm"],
                float(torch.linalg.vector_norm(self.routing_correction.z).item()),
            )
        )


    def _ol_one_batch(
        self,
        dataset_object,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark=None,
    ):
        """Update fully matured samples without accessing any future target."""

        del dataset_object, batch_y_mark
        x = batch_x.float().to(self.device)
        x_mark = batch_x_mark.float().to(self.device)
        target = batch_y.float().to(self.device)
        f_dim = -1 if self.args.features == "MS" else 0
        target = target[:, -self.args.pred_len :, f_dim:]
        expected = (x.shape[0], self.args.pred_len, self.args.c_out)
        if tuple(target.shape) != expected:
            raise ValueError(
                f"online target must have shape {expected}, got {tuple(target.shape)}"
            )
        true = rearrange(target, "b t d -> b (t d)")

        predictions = []
        truths = []
        for index in range(x.shape[0]):
            x_t = x[index : index + 1]
            x_mark_t = x_mark[index : index + 1]
            y_t = true[index : index + 1]
            recovery_batches = self._sample_recovery_batches()
            prediction = None
            for inner_index in range(self.n_inner):
                expert_lr, router_lr, dynamic_alpha = (
                    self._adaptive_online_hparams()
                )
                self._set_optimizer_lr(self.opt_expert, expert_lr)
                self._set_optimizer_lr(self.opt_router, router_lr)
                active_recovery = recovery_batches if inner_index == 0 else []
                expert_result = self._run_expert_optimizer_step(
                    x_t,
                    x_mark_t,
                    y_t,
                    expert_lr,
                    dynamic_alpha,
                    active_recovery,
                )
                router_result = self._run_full_router_update(
                    x_t, x_mark_t, y_t
                )
                if prediction is None:
                    prediction = expert_result["prediction"]
                if self.adaptive_controller == "dynamic":
                    self._update_online_mse_state(
                        self._select_criterion()(
                            router_result["prediction"], y_t
                        ).item()
                    )
                self._log_compact_online_update(
                    expert_result,
                    router_result,
                    y_t,
                    expert_lr,
                    router_lr,
                )
            if prediction is None:
                raise RuntimeError("n_inner must be at least one")
            if self.tsb_reference_enabled:
                self.online_buffer.append(
                    (x_t.detach().clone(), x_mark_t.detach().clone(), y_t.detach().clone())
                )
            predictions.append(prediction.detach())
            truths.append(y_t.detach())
            self.online_step += 1
        return torch.cat(predictions, dim=0), torch.cat(truths, dim=0)
