"""Concept-conditioned parameter adaptation from SJTU-DMTai/OnlineTSF."""

import torch
import torch.nn as nn

from models.patchtst_dgrad import PatchTSTDGrad


class ProceedForecaster(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = PatchTSTDGrad(args)
        dim = args.proceed_concept_dim
        self.current_encoder = nn.Sequential(
            nn.Linear(args.seq_len, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.recent_encoder = nn.Sequential(
            nn.Linear(args.seq_len + args.pred_len, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.generator = nn.Sequential(
            nn.Linear(dim, args.proceed_bottleneck_dim), nn.Sigmoid(),
            nn.Linear(args.proceed_bottleneck_dim, 2 * args.c_out),
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)
        self.register_buffer("recent", torch.zeros(1, args.seq_len + args.pred_len, args.c_out))

    def forward(self, x, x_mark=None, backbone_only=False):
        base = self.backbone(x)
        if backbone_only:
            return base
        current = self.current_encoder(x.transpose(1, 2)).mean(1)
        recent = self.recent_encoder(self.recent.transpose(1, 2)).mean(1)
        scale, shift = self.generator(current - recent).chunk(2, dim=-1)
        return base * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def observe(self, x, y):
        self.recent = torch.cat([x.detach(), y.detach()], dim=1).mean(0, keepdim=True)

    def freeze_backbone(self, frozen=True):
        self.backbone.requires_grad_(not frozen)

    def adapter_parameters(self):
        return (list(self.current_encoder.parameters()) + list(self.recent_encoder.parameters())
                + list(self.generator.parameters()))

