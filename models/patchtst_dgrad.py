"""Channel-independent PatchTST used by the DGrad online baseline.

Adapted from yfzhang114/OneNet's PatchTST implementation.  It retains the
patching, shared Transformer encoder, optional RevIN, and per-channel heads,
while removing unrelated TCN/MoE variants from the upstream backbone file.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, channels, affine=False, subtract_last=False):
        super().__init__()
        self.subtract_last = subtract_last
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, channels))

    def normalize(self, x):
        self.center = x[:, -1:, :].detach() if self.subtract_last else x.mean(1, keepdim=True).detach()
        self.scale = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - self.center) / self.scale
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denormalize(self, x):
        if self.affine:
            x = (x - self.bias) / (self.weight + 1e-8)
        return x * self.scale + self.center


class PatchTSTDGrad(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.channels = args.enc_in
        self.pred_len = args.pred_len
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.padding_patch = args.padding_patch == "end"
        self.revin = RevIN(args.enc_in, bool(args.affine), bool(args.subtract_last)) if args.revin else None

        patch_num = (args.seq_len - args.patch_len) // args.stride + 1
        if self.padding_patch:
            patch_num += 1
            self.pad = nn.ReplicationPad1d((0, args.stride))
        else:
            self.pad = nn.Identity()

        self.patch_projection = nn.Linear(args.patch_len, args.d_model)
        self.position = nn.Parameter(torch.zeros(1, patch_num, args.d_model))
        nn.init.uniform_(self.position, -0.02, 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=args.d_model, nhead=args.n_heads, dim_feedforward=args.d_ff,
            dropout=args.dropout, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=args.e_layers)
        head_in = patch_num * args.d_model
        self.individual = bool(args.individual)
        if self.individual:
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Flatten(), nn.Dropout(args.head_dropout), nn.Linear(head_in, args.pred_len))
                for _ in range(args.enc_in)
            ])
        else:
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(args.head_dropout), nn.Linear(head_in, args.pred_len))

    def forward(self, x):
        if self.revin is not None:
            x = self.revin.normalize(x)
        z = self.pad(x.transpose(1, 2))
        z = z.unfold(-1, self.patch_len, self.stride)  # [B,C,P,patch_len]
        batch, channels, patches, _ = z.shape
        z = self.patch_projection(z).reshape(batch * channels, patches, -1)
        # PyTorch 1.8 TransformerEncoder uses [sequence, batch, feature].
        z = (z + self.position[:, :patches]).transpose(0, 1)
        z = self.encoder(z).transpose(0, 1)
        z = z.reshape(batch, channels, patches, -1)
        if self.individual:
            pred = torch.stack([self.heads[c](z[:, c]) for c in range(channels)], dim=-1)
        else:
            pred = self.head(z.reshape(batch * channels, patches, -1)).reshape(batch, channels, -1).transpose(1, 2)
        if self.revin is not None:
            pred = self.revin.denormalize(pred)
        return pred
