"""Exact population MCMC on the replay buffer: noise-then-anchored-resampling (i-SIR Gibbs on the augmented space).

For chain x (a buffer point) and a noise level sigma:
  1. x_t = x + sigma * eps                       (exact draw from N(x_t | x))
  2. draw K proposals from q(y | x_t, R) = alpha N(x_t, sigma^2) + (1 - alpha) sum_j r_j N(R_j, tau^2), with R a reference
     set of OTHER chains (so q does not depend on x)
  3. resample x' among {x} ∪ proposals with weights exp(-E(y)) N(x_t; y, sigma^2) / q(y)
This is one Gibbs sweep on pi(x) N(x_t | x): step 1 samples x_t | x exactly, steps 2-3 are an i-SIR kernel that leaves
pi(x | x_t) invariant. Hence pi^{|S|} is invariant for every block S of chains updated given the reference set R.
"""
import math

import torch

from .estimators import _energy
from .replay import _log_normal
from .targets import effective_dim


@torch.no_grad()
def refine_block(target, x, ref, sigma, K=64, alpha=0.5, tau_max=1.0, generator=None):
    """One i-SIR Gibbs update of the chains x [B,d] using reference chains ref [R,d] (disjoint from x).
    sigma: [B] noise levels. Returns (new x [B,d], fraction of chains that moved)."""
    B, d = x.shape
    d_eff = effective_dim(target)
    sig = sigma.reshape(B, 1).to(x.dtype)
    s2 = sig[:, 0] ** 2
    tau = torch.clamp(sig[:, 0], max=tau_max)
    t2 = tau ** 2
    x_t = x + sig * target.project(torch.randn(B, d, generator=generator, dtype=x.dtype))
    logr = -0.5 * torch.cdist(x_t, ref) ** 2 / (s2 + t2)[:, None]
    logr = logr - torch.logsumexp(logr, 1, keepdim=True)
    comp_g = torch.rand(B, K, generator=generator, dtype=x.dtype) < alpha
    j = torch.multinomial(logr.exp(), K, replacement=True, generator=generator)
    eps = target.project(torch.randn(B, K, d, generator=generator, dtype=x.dtype))
    props = torch.where(comp_g[..., None], x_t[:, None, :] + sig[:, :, None] * eps, ref[j] + tau[:, None, None] * eps)
    parts = torch.cat([x[:, None, :], props], 1)
    log_qg = math.log(alpha) + _log_normal(parts, x_t[:, None, :], s2[:, None], d_eff) if alpha > 0 else None
    dy = torch.cdist(parts, ref.unsqueeze(0).expand(B, -1, -1)) ** 2
    log_k = logr[:, None, :] - 0.5 * dy / t2[:, None, None] - 0.5 * d_eff * torch.log(2 * math.pi * t2)[:, None, None]
    log_qr = math.log(1 - alpha) + torch.logsumexp(log_k, -1) if alpha < 1 else None
    log_q = log_qr if log_qg is None else (log_qg if log_qr is None else torch.logaddexp(log_qg, log_qr))
    E, _ = _energy(target, parts, need_grad=False)
    logw = -E + _log_normal(parts, x_t[:, None, :], s2[:, None], d_eff) - log_q
    logw = torch.where(torch.isfinite(logw), logw, torch.full_like(logw, -float('inf')))
    w = torch.softmax(logw, 1)
    w = torch.nan_to_num(w, nan=0.0)
    bad = w.sum(1) <= 0
    w[bad, 0] = 1.0
    J = torch.multinomial(w, 1, generator=generator)[:, 0]
    return parts[torch.arange(B), J], float((J > 0).float().mean())


@torch.no_grad()
def refine_buffer(target, buf, n_sweeps=1, block=256, K=64, alpha=0.5, tau_max=1.0, sigma_min=0.01, sigma_max=1.0,
                  generator=None):
    """Apply n_sweeps sweeps of block updates to the whole buffer. Each block's reference set is a random subset of
    the remaining chains. Noise levels are log-uniform in [sigma_min, sigma_max]. Returns (buffer, mean move rate)."""
    buf = buf.clone()
    N = buf.shape[0]
    rates = []
    for _ in range(n_sweeps):
        perm = torch.randperm(N, generator=generator)
        for s in range(0, N, block):
            idx = perm[s:s + block]
            rest = torch.cat([perm[:s], perm[s + block:]])
            if rest.numel() == 0:  # buffer smaller than one block: skip (no reference chains)
                continue
            ref = buf[rest[torch.randperm(rest.numel(), generator=generator)[:512]]]
            u = torch.rand(idx.numel(), generator=generator, dtype=buf.dtype)
            sig = sigma_min * (sigma_max / sigma_min) ** u
            buf[idx], r = refine_block(target, buf[idx], ref, sig, K=K, alpha=alpha, tau_max=tau_max,
                                       generator=generator)
            rates.append(r)
    return buf, (sum(rates) / len(rates) if rates else 0.0)
