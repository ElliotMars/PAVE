"""DynaME adapted from https://github.com/shhong97/DynaME.

The dynamic period experts and EWMA controller follow the upstream algorithm;
the forecasting backbone is FSNet-Time so the baseline uses this repository's
existing encoder and data protocol.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ts2vec.fsnet import TSEncoder


class FSNetTimeBackbone(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.encoder = TSEncoder(
            input_dims=args.seq_len, output_dims=320, hidden_dims=64,
            depth=10, device=device,
        )
        self.head = nn.Linear(320, args.pred_len)

    def forward(self, x, return_emb=False):
        with torch.backends.cudnn.flags(enabled=False):
            rep = self.encoder.forward_time(x, mask="all_true")  # [B, C, H]
        pred = self.head(rep).transpose(1, 2)  # [B, L, C]
        if return_emb:
            return pred, rep.permute(0, 2, 1)  # [B, H, C]
        return pred

    def store_grad(self):
        for module in self.encoder.modules():
            if "PadConv" in type(module).__name__:
                module.store_grad()


class ExpertGate(nn.Module):
    def __init__(self, feature_dim, num_experts, temperature):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_experts),
        )
        self.temperature = temperature

    def forward(self, rep):
        return F.softmax(self.net(rep) / self.temperature, dim=-1)


class DynaME(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.backbone = FSNetTimeBackbone(args, device)
        self.period_num = args.dyname_period_num
        self.expert_gate = ExpertGate(320, self.period_num + 1, args.dyname_temperature)
        self.ewma = None
        self.signal = 0.0

    def backbone_parameters(self):
        return self.backbone.parameters()

    def gate_parameters(self):
        return self.expert_gate.parameters()

    def freeze_backbone(self, frozen=True):
        self.backbone.requires_grad_(not frozen)

    def _periods(self, x):
        spectrum = torch.fft.rfft(x.permute(0, 2, 1), dim=-1).abs().mean(dim=1)
        available = max(spectrum.shape[-1] - 1, 1)
        k = min(self.period_num, available)
        indices = torch.topk(spectrum[:, 1:], k=k, dim=-1).indices + 1
        periods = (x.shape[1] // indices).squeeze(0).tolist()
        if not isinstance(periods, list):
            periods = [periods]
        return sorted([max(int(p), 1) for p in periods], reverse=True)

    def _krr(self, train_rep, train_y, test_rep):
        # Inputs: [N, H, C], [N, L, C], [1, H, C].
        x = train_rep.permute(2, 0, 1)
        y = train_y.permute(2, 0, 1)
        q = test_rep.permute(2, 0, 1)
        kernel = x @ x.transpose(-1, -2)
        query = q @ x.transpose(-1, -2)
        trace = torch.diagonal(kernel, dim1=-2, dim2=-1).sum(-1)
        ridge = self.args.dyname_krr_lambda * trace / x.shape[1]
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        alpha = torch.linalg.solve(kernel + ridge[:, None, None] * eye + 1e-8 * eye, y)
        return (query @ alpha).permute(1, 2, 0)

    def _period_expert(self, period, past, current_rep, fallback):
        last_start = past.shape[1] - self.args.seq_len - self.args.pred_len
        if last_start < 0:
            return fallback
        starts = torch.arange(last_start, -1, -period, device=past.device)
        starts = starts[:self.args.dyname_krr_train_num]
        if starts.numel() == 0:
            return fallback
        xs, ys = [], []
        for start in starts.tolist():
            xs.append(past[:, start:start + self.args.seq_len])
            ys.append(past[:, start + self.args.seq_len:start + self.args.seq_len + self.args.pred_len])
        train_x = torch.cat(xs, dim=0)
        train_y = torch.cat(ys, dim=0)
        _, train_rep = self.backbone(train_x, return_emb=True)
        return self._krr(train_rep, train_y, current_rep)

    def combine_cached(self, rep, stacked, blend=None):
        weights = self.expert_gate(rep.permute(0, 2, 1))  # [B, C, E]
        base = torch.zeros_like(weights)
        base[..., 0] = 1.0
        if blend is None:
            blend = self.args.dyname_beta + self.signal * (1.0 - self.args.dyname_beta)
        weights = (1.0 - blend) * weights + blend * base
        pred = (stacked * weights.permute(0, 2, 1).unsqueeze(2)).sum(dim=1)
        return pred, weights, blend

    def forward(self, x, past=None, backbone_only=False, return_details=False):
        backbone_pred, rep = self.backbone(x, return_emb=True)
        if backbone_only or past is None:
            return backbone_pred
        experts = [backbone_pred]
        for period in self._periods(x):
            experts.append(self._period_expert(period, past, rep, backbone_pred))
        while len(experts) < self.period_num + 1:
            experts.append(backbone_pred)
        stacked = torch.stack(experts[:self.period_num + 1], dim=1)  # [B,E,L,C]
        pred, weights, blend = self.combine_cached(rep, stacked)
        if return_details:
            return pred, backbone_pred, weights, blend, rep, stacked
        return pred

    def update_signal(self, mse):
        mse = float(mse)
        if self.ewma is None:
            self.ewma = mse
            self.signal = 0.0
        else:
            self.ewma = 0.95 * self.ewma + 0.05 * mse
            diff = mse - self.ewma
            new_signal = 1.0 - np.exp(-self.args.dyname_delta * diff * diff)
            self.signal = max(0.8 * self.signal, float(new_signal))
        return self.signal

