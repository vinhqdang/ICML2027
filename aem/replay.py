"""Anchored SNIS with replay (buffer-informed) proposals.

For row i with noised point x_t[i] (noise sigma[i]) and anchor x0[i], the proposal is the mixture
    q_i(y) = alpha * N(y; x_t[i], sigma_i^2 I) + (1 - alpha) * sum_{j != i} r_ij N(y; x0[j], tau_i^2 I),
    r_ij ∝ N(x_t[i]; x0[j], (sigma_i^2 + tau_i^2) I),    tau_i = min(sigma_i, tau_max).
The kernel-density part is built from the OTHER rows of the minibatch (leave-one-out), so q_i does not depend on
the anchor x0[i] and the conditional-importance-sampling argument (unbiasedness for exact anchors, contraction for
inexact ones) applies unchanged. Importance weights are exact: w(y) = exp(-E(y)) N(x_t; y, sigma^2) / q_i(y).
"""
import math

import torch

from .estimators import _energy
from .targets import effective_dim


def _log_normal(y, mean, var, d=None):
    """log N(y; mean, var I) on a d-dimensional space (d defaults to the last axis of y).
    For projected (zero-centre-of-mass) Gaussians pass d = effective_dim(target). y: [B,P,d], var: [B,1]."""
    d = y.shape[-1] if d is None else d
    return -0.5 * ((y - mean) ** 2).sum(-1) / var - 0.5 * d * torch.log(2 * math.pi * var)


def _block(target, x_t, sig, anchor, ref, ref_mask, K, kind, alpha, tau_max, clip, gen, return_diag):
    B, d = x_t.shape
    d_eff = effective_dim(target)
    s2 = sig[:, 0] ** 2  # [B]
    tau = torch.clamp(sig[:, 0], max=tau_max)
    t2 = tau**2
    # reference responsibilities r_ij (leave-one-out through ref_mask)
    d2 = torch.cdist(x_t, ref) ** 2  # [B, R]
    logr = -0.5 * d2 / (s2 + t2)[:, None]
    logr = logr.masked_fill(~ref_mask, -float('inf'))
    logr = logr - torch.logsumexp(logr, dim=1, keepdim=True)
    # draw K proposals per row from the mixture
    comp_gauss = torch.rand(B, K, generator=gen, dtype=x_t.dtype) < alpha
    j = torch.multinomial(logr.exp(), K, replacement=True, generator=gen)  # [B, K]
    eps = target.project(torch.randn(B, K, d, generator=gen, dtype=x_t.dtype))
    y_g = x_t[:, None, :] + sig[:, :, None] * eps
    y_r = ref[j] + tau[:, None, None] * eps
    props = torch.where(comp_gauss[..., None], y_g, y_r)
    parts = props if anchor is None else torch.cat([anchor[:, None, :], props], dim=1)  # [B, P, d]
    # log q_i(parts)
    log_qg = math.log(alpha) + _log_normal(parts, x_t[:, None, :], s2[:, None], d_eff) if alpha > 0 else None
    # kde term: logsumexp_j [log r_ij + log N(y; ref_j, tau_i^2)]
    P = parts.shape[1]
    flat = parts.reshape(B, P, d)
    dy = torch.cdist(flat, ref.unsqueeze(0).expand(B, -1, -1)) ** 2  # [B, P, R]
    log_k = logr[:, None, :] - 0.5 * dy / t2[:, None, None] - 0.5 * d_eff * torch.log(2 * math.pi * t2)[:, None, None]
    log_qr = math.log(1 - alpha) + torch.logsumexp(log_k, dim=-1) if alpha < 1 else None
    if log_qg is None:
        log_q = log_qr
    elif log_qr is None:
        log_q = log_qg
    else:
        log_q = torch.logaddexp(log_qg, log_qr)
    E, neg_grad = _energy(target, parts, need_grad=kind in ('tsi', 'mix', 'avg'))
    loglik = _log_normal(parts, x_t[:, None, :], s2[:, None], d_eff)  # log N(x_t; y, sigma^2) (symmetric)
    logw = -E + loglik - log_q
    logw = torch.where(torch.isfinite(logw), logw, torch.full_like(logw, -float('inf')))
    w = torch.nan_to_num(torch.softmax(logw, dim=1), nan=0.0)
    dead = w.sum(1) <= 0
    w = w.masked_fill(dead[:, None], 0.0)
    w[:, 0] = torch.where(dead, torch.ones_like(w[:, 0]), w[:, 0])
    est_tsi = est_dsi = None
    if kind in ('tsi', 'mix', 'avg'):
        g = neg_grad
        if clip is not None:
            nrm = g.norm(dim=-1, keepdim=True)
            g = g * torch.clamp(clip / nrm.clamp_min(1e-30), max=1.0)
        est_tsi = (w[..., None] * g).sum(1)
    if kind in ('dsi', 'mix', 'avg'):
        est_dsi = (w[..., None] * (parts - x_t[:, None, :])).sum(1) / sig**2
    if kind == 'tsi':
        est = est_tsi
    elif kind == 'dsi':
        est = est_dsi
    elif kind == 'avg':
        est = 0.5 * (est_tsi + est_dsi)
    else:
        a = 1.0 / (1.0 + sig**2)
        est = a * est_tsi + (1 - a) * est_dsi
    diag = None
    if return_diag:
        diag = {'anchor_weight': w[:, 0] if anchor is not None else torch.zeros(B, dtype=x_t.dtype),
                'ess': 1.0 / (w**2).sum(1).clamp_min(1e-300)}
    return est, diag


def score_estimate_replay(x_t, sigma, target, K, ref, anchor=None, kind='mix', alpha=0.5, tau_max=1.0, clip=None,
                          generator=None, loo=True, return_diag=False, max_rows=64):
    """Anchored (or plain) SNIS score estimate with replay proposals. ref: [R, d] reference buffer points.
    If loo, row i excludes ref[i] (requires R == B and ref[i] = the anchor of row i)."""
    with torch.no_grad():
        x_t = x_t.detach()
        B, d = x_t.shape
        sig = sigma.reshape(-1, 1).to(x_t.dtype) if torch.is_tensor(sigma) else torch.full((B, 1), float(sigma))
        if sig.shape[0] == 1 and B > 1:
            sig = sig.expand(B, 1)
        ref = ref.detach().to(x_t.dtype)
        R = ref.shape[0]
        mask_full = torch.ones(B, R, dtype=torch.bool)
        if loo:
            assert R == B, 'leave-one-out requires ref to be the minibatch of anchors'
            mask_full &= ~torch.eye(B, dtype=torch.bool)
        ests, diags = [], []
        for s in range(0, B, max_rows):
            e = slice(s, s + max_rows)
            est, diag = _block(target, x_t[e], sig[e], None if anchor is None else anchor[e].detach().to(x_t.dtype),
                               ref, mask_full[e], K, kind, alpha, tau_max, clip, generator, return_diag)
            ests.append(est)
            diags.append(diag)
        est = torch.cat(ests).detach()
        if return_diag:
            return est, {k: torch.cat([dd[k] for dd in diags]) for k in diags[0]}
        return est
