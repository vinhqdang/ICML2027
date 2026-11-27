"""Score network s_theta(x, t) for t in [0, 1]. With sigma_min/max given, score = output / sigma(t)."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class ScoreNet(nn.Module):
    """MLP on [x, sinusoidal(t)] with SiLU activations.

    n_layers counts hidden layers of width `hidden`; the output layer is linear and has dim outputs.
    """

    def __init__(self, dim, hidden=256, n_layers=4, time_emb=64, sigma_min=None, sigma_max=None):
        super().__init__()
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
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
        """Score used for training and sampling. If sigma_min/sigma_max are set, the network predicts
        sigma(t) * score (the usual VE preconditioning), so score = forward / sigma(t)."""
        out = self.forward(x, t)
        if self.sigma_min is None:
            return out
        sig = self.sigma_min * (self.sigma_max / self.sigma_min) ** t.reshape(-1, 1).to(out.dtype)
        return out / sig


class _EGNNLayer(nn.Module):
    """One EGNN layer: edge messages m_ij, a bounded coordinate weight phi_x(m_ij) and a node update.

    The first edge layer acts on the concatenation [h_i, h_j, ||x_i - x_j||^2, t_emb]. It is stored as one Linear and
    evaluated by splitting its weight, so the per-node and per-time parts are computed once and only the
    distance term and the sum are formed on the n x n pairs.
    """

    def __init__(self, hidden, time_emb):
        super().__init__()
        self.edge_in = nn.Linear(2 * hidden + 1 + time_emb, hidden)
        self.edge_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.coord = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1), nn.Tanh())
        self.node = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))


class EGNNScore(nn.Module):
    """E(n)-equivariant score network (Satorras et al. 2021) for n_particles points in spatial_dim dimensions.

    Input x: [B, n * s] (flat particle coordinates), t: [B] or [B, 1]. The output is the predicted sigma(t) * score,
    [B, n * s], projected to zero centre of mass. Node features start from the time embedding, so all particles
    start identical (they are indistinguishable). Messages use only invariant quantities (node features, squared
    distances, time), and coordinates move along relative vectors, so the output is translation invariant and
    rotation/reflection equivariant. The whole particle dimension is vectorised as [B, n, n, .] tensors; the batch
    is processed in chunks of max_batch rows to bound memory.
    """

    def __init__(self, n_particles, spatial_dim, hidden=128, n_layers=4, time_emb=32, sigma_min=None, sigma_max=None,
                 max_batch=1024):
        super().__init__()
        self.n_particles, self.spatial_dim = n_particles, spatial_dim
        self.dim = n_particles * spatial_dim
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.max_batch = max_batch
        assert time_emb % 2 == 0, 'time_emb must be even'
        half = time_emb // 2
        self.register_buffer('freqs', 2 * math.pi * torch.exp(torch.linspace(0.0, math.log(100.0), half)))
        self.embed = nn.Linear(time_emb, hidden)
        self.layers = nn.ModuleList([_EGNNLayer(hidden, time_emb) for _ in range(n_layers)])

    def time_embedding(self, t):
        ang = t.reshape(-1, 1).to(self.freqs.dtype) * self.freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def _project(self, v):
        """Zero centre of mass of displacements v [B, n, s]."""
        return v - v.mean(dim=1, keepdim=True)

    def _forward_chunk(self, x, temb):
        B, n, s = x.shape[0], self.n_particles, self.spatial_dim
        pos0 = x.reshape(B, n, s)
        pos = pos0
        h = self.embed(temb)[:, None, :].expand(B, n, -1)
        eye_mask = (1.0 - torch.eye(n, dtype=x.dtype, device=x.device))[None, :, :, None]  # no self-edges
        H = h.shape[-1]
        for layer in self.layers:
            rel = pos[:, :, None, :] - pos[:, None, :, :]  # [B, n, n, s], x_i - x_j
            d2 = (rel**2).sum(-1, keepdim=True)  # [B, n, n, 1]
            W, bias = layer.edge_in.weight, layer.edge_in.bias  # columns: [h_i | h_j | d2 | t_emb]
            pre = (F.linear(h, W[:, :H])[:, :, None, :] + F.linear(h, W[:, H:2 * H])[:, None, :, :]
                   + d2 * W[:, 2 * H] + F.linear(temb, W[:, 2 * H + 1:], bias)[:, None, None, :])
            m = layer.edge_out(pre) * eye_mask  # [B, n, n, H]
            w = layer.coord(m) * eye_mask  # [B, n, n, 1], bounded by tanh
            pos = pos + (rel * w).sum(2) / (n - 1)
            agg = m.sum(2) / (n - 1)  # [B, n, H]
            h = h + layer.node(torch.cat([h, agg], dim=-1))
        return self._project(pos - pos0).reshape(B, n * s)

    def forward(self, x, t):
        """x: [B, dim]; t: [B] or [B, 1]. Returns the network output sigma(t) * score, [B, dim]."""
        temb = self.time_embedding(t)
        if temb.shape[0] == 1 and x.shape[0] > 1:  # one time for the whole batch
            temb = temb.expand(x.shape[0], -1)
        outs = [self._forward_chunk(x[s:s + self.max_batch], temb[s:s + self.max_batch])
                for s in range(0, x.shape[0], self.max_batch)]
        return torch.cat(outs) if outs else x.new_zeros(0, self.dim)

    def score(self, x, t):
        """Score = forward / sigma(t) when sigma_min/sigma_max are set (as ScoreNet.score)."""
        out = self.forward(x, t)
        if self.sigma_min is None:
            return out
        sig = self.sigma_min * (self.sigma_max / self.sigma_min) ** t.reshape(-1, 1).to(out.dtype)
        return out / sig
