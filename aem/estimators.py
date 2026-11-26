"""Self-normalised importance sampling estimators of the noised target score, with an optional anchor particle.

For x_t = x0 + sigma * eps the noised score is grad log p_t(x_t) = E_{pi(x0|x_t)}[-grad E(x0)] (TSI)
= E_{pi(x0|x_t)}[(x0 - x_t) / sigma^2] (DSI). Proposals x0^k = x_t + sigma * eps_k weighted by exp(-E).
If an anchor x0* (a buffer sample that generated x_t) is given, it is an extra particle in the SNIS sum.
"""
import torch

from .targets import as_col

# Upper bound on particles (rows * (K + anchor)) evaluated at once, to bound memory.
_MAX_PARTICLES = 2**18


def _energy(target, parts, need_grad):
    """Energies [B, P] and, if need_grad, -grad E [B, P, d]."""
    B, P, d = parts.shape
    flat = parts.reshape(-1, d)
    if need_grad:
        flat = flat.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            E = target.energy(flat)
            (grad,) = torch.autograd.grad(E.sum(), flat)
        return E.detach().reshape(B, P), (-grad).reshape(B, P, d)
    return target.energy(flat).reshape(B, P), None


def _estimate_block(target, x_t, sig, anchor, K, kind, n_isir, clip, generator, return_diag):
    B, d = x_t.shape
    cur_anchor = anchor
    for rep in range(n_isir):
        last = rep == n_isir - 1
        eps = torch.randn(B, K, d, generator=generator, dtype=x_t.dtype)
        props = x_t[:, None, :] + sig[:, :, None] * eps
        parts = props if cur_anchor is None else torch.cat([cur_anchor[:, None, :], props], dim=1)

        E, neg_grad = _energy(target, parts, need_grad=last and kind in ('tsi', 'mix', 'avg'))
        logw = torch.where(torch.isfinite(-E), -E, torch.full_like(E, -float('inf')))
        w = torch.nan_to_num(torch.softmax(logw, dim=1), nan=0.0)
        dead = w.sum(1) <= 0
        w = w.masked_fill(dead[:, None], 0.0)
        w[:, 0] = torch.where(dead, torch.ones_like(w[:, 0]), w[:, 0])

        if not last:
            J = torch.multinomial(w, 1, generator=generator)[:, 0]
            cur_anchor = parts[torch.arange(B), J]
            continue

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
        else:  # 'mix': precision-motivated schedule, TSI at low noise and DSI at high noise
            alpha = 1.0 / (1.0 + sig**2)
            est = alpha * est_tsi + (1.0 - alpha) * est_dsi

    if not return_diag:
        return est, None
    anchor_weight = w[:, 0] if anchor is not None else torch.zeros(B, dtype=x_t.dtype)
    ess = 1.0 / (w**2).sum(1).clamp_min(1e-300)
    return est, {'anchor_weight': anchor_weight, 'ess': ess}


def score_estimate(x_t, sigma, target, K, anchor=None, kind='tsi', n_isir=1, clip=None, generator=None,
                   return_diag=False):
    """Estimate the noised score at x_t from K proposals, optionally with an anchor particle.

    Args:
        x_t: [B, d] noised points.
        sigma: scalar, [B] or [B, 1] noise level.
        target: object with energy(x) -> [N].
        K: number of proposals per row (K = 0 requires an anchor).
        anchor: [B, d] clean sample that generated x_t, or None for the iDEM baseline.
        kind: 'tsi' (-grad E), 'dsi' ((x0 - x_t)/sigma^2) or 'mix' (average of both on the same particles).
        n_isir: number of i-SIR repetitions (anchored only). Repetition r > 1 resamples the anchor from the weights
            of repetition r-1 and redraws proposals around the same x_t. The returned estimate is the last repetition.
        clip: if not None, clip the norm of each per-particle TSI gradient to this value.
        generator: torch.Generator for proposals and resampling (None uses the global RNG).
        return_diag: also return {'anchor_weight': [B], 'ess': [B]} for the last repetition.

    Returns:
        est [B, d] (detached), and the diagnostics dict if return_diag.
    """
    if kind not in ('tsi', 'dsi', 'mix', 'avg'):
        raise ValueError(f'unknown kind {kind!r}')
    if K < 0 or (K == 0 and anchor is None):
        raise ValueError('K = 0 requires an anchor')
    if n_isir > 1 and anchor is None:
        raise ValueError('n_isir > 1 requires an anchor')
    if n_isir < 1:
        raise ValueError('n_isir must be >= 1')

    with torch.no_grad():
        x_t = x_t.detach()
        B = x_t.shape[0]
        sig = as_col(sigma, B).to(x_t.dtype)
        if anchor is not None:
            anchor = anchor.detach().to(x_t.dtype)

        rows = max(1, _MAX_PARTICLES // (K + (anchor is not None)))
        if B > rows:
            parts = []
            for s in range(0, B, rows):
                e = slice(s, s + rows)
                parts.append(score_estimate(
                    x_t[e], sig[e], target, K,
                    anchor=None if anchor is None else anchor[e], kind=kind, n_isir=n_isir,
                    clip=clip, generator=generator, return_diag=True))
            est = torch.cat([p[0] for p in parts])
            if not return_diag:
                return est
            diag = {k: torch.cat([p[1][k] for p in parts]) for k in ('anchor_weight', 'ess')}
            return est, diag

        est, diag = _estimate_block(target, x_t, sig, anchor, K, kind, n_isir, clip, generator, return_diag)
        est = est.detach()
        if return_diag:
            return est, {k: v.detach() for k, v in diag.items()}
        return est
