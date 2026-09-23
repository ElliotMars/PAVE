"""OneNet dynamic ensemble adapted from yfzhang114/OneNet."""

import torch
import torch.nn as nn

from models.fsnet import FSNetForecaster, FSNetTimeForecaster


class OneNetForecaster(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.variate = FSNetForecaster(args, device)
        self.temporal = FSNetTimeForecaster(args, device)
        self.gate = nn.Sequential(nn.Linear(args.seq_len, 32), nn.Tanh(), nn.Linear(32, 1))
        self.base_logits = nn.Parameter(torch.zeros(1, 1, args.c_out))

    def forward(self, x, x_mark=None, return_experts=False):
        y_var = self.variate(x, x_mark)
        y_time = self.temporal(x, x_mark)
        bias = self.gate(x.transpose(1, 2)).transpose(1, 2)
        weight = torch.sigmoid(self.base_logits + bias)
        pred = weight * y_time + (1.0 - weight) * y_var
        return (pred, y_time, y_var, weight) if return_experts else pred

    def store_grad(self):
        self.variate.store_grad()
        self.temporal.store_grad()

