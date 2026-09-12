# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class XLSRSLS(nn.Module):
    def __init__(self, n_layers=25, dim=1024, hidden=256, dropout=0.3):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        w = torch.sigmoid(self.layer_w).view(1, -1, 1)
        feat = (x * w).sum(dim=1) / (w.sum() + 1e-8)
        return self.mlp(self.norm(feat)).squeeze(-1)

    @torch.no_grad()
    def prob(self, x):
        return torch.sigmoid(self.forward(x))
