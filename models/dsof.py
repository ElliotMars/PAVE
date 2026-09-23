"""Fast MLP stream used with a PatchTST slow stream in DSOF."""

import torch.nn as nn


class DSOFStudent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.seq_len, args.dsof_student_width),
            nn.Identity(),
            nn.Linear(args.dsof_student_width, args.pred_len),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)

