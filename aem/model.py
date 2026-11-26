"""Score network s_theta(x, t) for t in [0, 1]. The output is used directly as the score (no sigma scaling)."""
import math

import torch
from torch import nn


class ScoreNet(nn.Module):
    """MLP on [x, sinusoidal(t)] with SiLU activations.

    n_layers counts hidden layers of width `hidden`; the output layer is linear and has dim outputs.
    """

    def __init__(self, dim, hidden=256, n_layers=4, time_emb=64):
        super().__init__()
        assert time_emb % 2 == 0, 'time_emb must be even'
        half = time_emb // 2
        self.register_buffer('freqs', 2 * math.pi * torch.exp(torch.linspace(0.0, math.log(100.0), half)))
        layers, d_in = [], dim + time_emb
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.SiLU()]
            d_in = hidden
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def time_embedding(self, t):
        ang = t.reshape(-1, 1).to(self.freqs.dtype) * self.freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, x, t):
        """x: [B, dim]; t: [B] or [B, 1]. Returns the raw network output [B, dim]."""
        return self.net(torch.cat([x, self.time_embedding(t)], dim=-1))

    def score(self, x, t):
        """The score used by the sampler (identical to forward)."""
        return self.forward(x, t)
