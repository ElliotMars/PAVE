"""FSNet forecasting models adapted from salesforce/fsnet."""

import torch
import torch.nn as nn

from models.ts2vec.fsnet import TSEncoder


class FSNetForecaster(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.pred_len, self.channels = args.pred_len, args.c_out
        self.encoder = TSEncoder(args.enc_in + 7, 320, 64, 10, device=device)
        self.head = nn.Linear(320, args.pred_len * args.c_out)

    def forward(self, x, x_mark=None):
        z = torch.cat([x.clone(), x_mark], dim=-1)
        with torch.backends.cudnn.flags(enabled=False):
            z = self.encoder(z, mask="all_true")[:, -1]
        return self.head(z).view(x.shape[0], self.pred_len, self.channels)

    def store_grad(self):
        for module in self.encoder.modules():
            if "PadConv" in type(module).__name__:
                module.store_grad()


class FSNetTimeForecaster(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.encoder = TSEncoder(args.seq_len, 320, 64, 10, device=device)
        self.head = nn.Linear(320, args.pred_len)

    def forward(self, x, x_mark=None):
        with torch.backends.cudnn.flags(enabled=False):
            z = self.encoder.forward_time(x.clone(), mask="all_true")
        return self.head(z).transpose(1, 2)

    def store_grad(self):
        for module in self.encoder.modules():
            if "PadConv" in type(module).__name__:
                module.store_grad()

