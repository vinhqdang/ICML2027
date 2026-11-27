"""Decoupled exploration for energy-based diffusion samplers: exact subspace i-SIR moves with external references.

Two ingredients, both usable on any free (non-projected) target:

1. Subspace i-SIR Gibbs refinement (`refine_subspace_block`). For chain x, a coordinate block S (random per chain)
   and a noise level sigma:
     x_t,S = x_S + sigma * eps_S                                  (exact draw of the auxiliary variable)
     proposals y_S ~ q_S(. | x_t,S, R) = alpha N(x_t,S, sigma^2 I_S) + (1-alpha) sum_j r_j N(R_j,S, tau^2 I_S),
                y_{-S} = x_{-S},     r_j ∝ N(x_t,S; R_j,S, (sigma^2 + tau^2) I_S)
     resample among {x} ∪ proposals with weights exp(-E(y)) N(x_t,S; y_S, sigma^2) / q_S(y_S).
   This is a Gibbs step on the augmented density pi(x) N(x_t,S | x_S): step 1 samples x_t,S | x exactly, the i-SIR
   step leaves pi(x_S | x_{-S}, x_t,S) invariant. Hence pi is invariant for every S and every reference set R that
   does not depend on x. S = all coordinates recovers aem.refine.refine_block. With R taken from other chains,
   subspace KDE proposals recombine coordinate blocks of different chains ("crossover"), exactly.

2. Exploration population (`annealed_walkers`). Walkers start from the VE prior N(0, sigma_max^2 I) and follow the
   geometric path gamma_b(x) = N(x; 0, s0^2 I)^{1-b} exp(-b E(x)) with one MALA step per level. They use only E and
   grad E and never see the buffer or the network, so they are independent of the chains. Their endpoints serve as
   (part of) the reference set R. Their law need not be pi: the i-SIR weights correct the mode weights; the walkers
   only have to *cover* the modes. AIS log-weights are returned for diagnostics.
"""
import math

import torch

from .estimators import _energy


def _logn_masked(y, mean, var, m, nS):
    """log N(y_S; mean_S, var I_S). y, mean broadcastable [B,P,d]; var [B,1]; m [B,1,d] float mask; nS [B,1]."""
    return -0.5 * (((y - mean) ** 2) * m).sum(-1) / var - 0.5 * nS * torch.log(2 * math.pi * var)


def random_masks(B, d, b_max=None, p_full=0.25, generator=None, dtype=torch.float32):
    """Per-row coordinate masks [B,d]. With prob p_full the full vector; otherwise a uniformly random subset whose
    size is log-uniform on [1, b_max] (rounded)."""
    b_max = d if b_max is None or b_max <= 0 else min(b_max, d)
    u = torch.rand(B, generator=generator)
    size = torch.floor(torch.exp(u * math.log(b_max + 1))).clamp(1, b_max).long()
    full = torch.rand(B, generator=generator) < p_full
    size = torch.where(full, torch.full_like(size, d), size)
    ranks = torch.argsort(torch.rand(B, d, generator=generator), dim=1).argsort(dim=1)  # random permutation ranks
    return (ranks < size[:, None]).to(dtype)


@torch.no_grad()
def refine_subspace_block(target, x, ref, sigma, mask, K=32, alpha=0.5, tau_max=1.0, generator=None):
    """One subspace i-SIR Gibbs update of the chains x [B,d] given reference points ref [R,d] (independent of x).
    sigma [B]; mask [B,d] in {0,1} (at least one 1 per row). Returns (new x, fraction moved)."""
    B, d = x.shape
    dt = x.dtype
    m = mask.to(dt)
    nS = m.sum(1, keepdim=True)  # [B,1]
    sig = sigma.reshape(B, 1).to(dt)
    s2 = sig ** 2
    tau = torch.clamp(sig, max=tau_max)
    t2 = tau ** 2
    x_t = x + sig * m * torch.randn(B, d, generator=generator, dtype=dt)
    # masked squared distances |m_i (x_t,i - R_j)|^2 = sum m x^2 - 2 (m x) R^T + m (R^2)^T
    R2 = ref ** 2
    d2 = ((x_t ** 2) * m).sum(1, keepdim=True) - 2.0 * (x_t * m) @ ref.T + m @ R2.T
    logr = -0.5 * d2.clamp_min(0) / (s2 + t2)
    logr = logr - torch.logsumexp(logr, 1, keepdim=True)
    comp_g = torch.rand(B, K, generator=generator, dtype=dt) < alpha
    j = torch.multinomial(logr.exp(), K, replacement=True, generator=generator)
    eps = torch.randn(B, K, d, generator=generator, dtype=dt)
    y_g = x_t[:, None, :] + sig[:, :, None] * eps
    y_r = ref[j] + tau[:, :, None] * eps
    y_new = torch.where(comp_g[..., None], y_g, y_r)
    m3 = m[:, None, :]
    props = m3 * y_new + (1 - m3) * x[:, None, :]
    parts = torch.cat([x[:, None, :], props], 1)  # [B,P,d]
    # log q_S
    log_qg = math.log(alpha) + _logn_masked(parts, x_t[:, None, :], s2, m3, nS) if alpha > 0 else None
    if alpha < 1:
        A = ((parts ** 2) * m3).sum(-1)  # [B,P]
        C = torch.einsum('bpk,jk->bpj', parts * m3, ref)
        D = m @ R2.T  # [B,R]
        dy = (A[..., None] - 2 * C + D[:, None, :]).clamp_min(0)
        log_k = logr[:, None, :] - 0.5 * dy / t2[:, :, None] - 0.5 * nS[:, :, None] * torch.log(2 * math.pi * t2)[:, :, None]
        log_qr = math.log(1 - alpha) + torch.logsumexp(log_k, -1)
    else:
        log_qr = None
    log_q = log_qr if log_qg is None else (log_qg if log_qr is None else torch.logaddexp(log_qg, log_qr))
    E, _ = _energy(target, parts, need_grad=False)
    logw = -E + _logn_masked(parts, x_t[:, None, :], s2, m3, nS) - log_q
    logw = torch.where(torch.isfinite(logw), logw, torch.full_like(logw, -float('inf')))
    w = torch.nan_to_num(torch.softmax(logw, 1), nan=0.0)
    bad = w.sum(1) <= 0
    w[bad] = 0.0
    w[bad, 0] = 1.0
    J = torch.multinomial(w, 1, generator=generator)[:, 0]
    return parts[torch.arange(B), J], float((J > 0).float().mean())


@torch.no_grad()
def refine_buffer_explore(target, buf, explore_ref=None, n_sweeps=1, block=256, K=32, alpha=0.5, tau_max=1.0,
                          sigma_min=0.01, sigma_max=1.0, b_max=None, p_full=0.25, explore_frac=0.5, n_ref=512,
                          generator=None):
    """n_sweeps sweeps of subspace i-SIR updates over the whole buffer. For each block of chains the reference set is
    a random subset of the OTHER chains, mixed with points of `explore_ref` (an independent exploration population)
    in proportion explore_frac. Noise levels log-uniform in [sigma_min, sigma_max]. Returns (buffer, move rate,
    number of energy evaluations)."""
    buf = buf.clone()
    N, d = buf.shape
    rates, n_e = [], 0
    n_x = 0 if explore_ref is None or explore_ref.shape[0] == 0 else int(round(n_ref * explore_frac))
    for _ in range(n_sweeps):
        perm = torch.randperm(N, generator=generator)
        for s in range(0, N, block):
            idx = perm[s:s + block]
            rest = torch.cat([perm[:s], perm[s + block:]])
            refs = []
            if rest.numel() > 0 and n_ref - n_x > 0:
                refs.append(buf[rest[torch.randperm(rest.numel(), generator=generator)[:n_ref - n_x]]])
            if n_x > 0:
                refs.append(explore_ref[torch.randint(explore_ref.shape[0], (n_x,), generator=generator)])
            if not refs:
                continue
            ref = torch.cat(refs).to(buf.dtype)
            u = torch.rand(idx.numel(), generator=generator, dtype=buf.dtype)
            sig = sigma_min * (sigma_max / sigma_min) ** u
            mask = random_masks(idx.numel(), d, b_max=b_max, p_full=p_full, generator=generator, dtype=buf.dtype)
            buf[idx], r = refine_subspace_block(target, buf[idx], ref, sig, mask, K=K, alpha=alpha, tau_max=tau_max,
                                                generator=generator)
            rates.append(r)
            n_e += idx.numel() * (K + 1)
    return buf, (sum(rates) / len(rates) if rates else 0.0), n_e


def _e_and_grad(target, x):
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        E = target.energy(x)
        (g,) = torch.autograd.grad(E.sum(), x)
    E = E.detach()
    g = g.detach()
    bad = ~torch.isfinite(E) | ~torch.isfinite(g).all(1)
    E = torch.where(bad, torch.full_like(E, float('inf')), E)
    g = torch.where(bad[:, None], torch.zeros_like(g), g)
    return E, g


@torch.no_grad()
def annealed_walkers(target, n, s0, n_levels=200, step0=None, accept_target=0.55, generator=None,
                     dtype=torch.float32, return_logw=False):
    """Annealed MALA walkers on gamma_b ∝ N(0, s0^2 I)^{1-b} exp(-b E), b from 0 to 1 (geometric spacing in b at the
    start, linear later), one MALA step per level, global step size adapted towards accept_target.
    Returns endpoints [n,d] (and AIS log-weights) and the number of energy+gradient evaluations."""
    d = target.dim
    x = s0 * torch.randn(n, d, generator=generator, dtype=dtype)
    # schedule: half geometric from 1e-4, then linear to 1
    h = n_levels // 2
    betas = torch.cat([torch.logspace(-4, -1, h, dtype=torch.float64),
                       torch.linspace(0.1, 1.0, n_levels - h + 1, dtype=torch.float64)[1:]]).tolist()
    E, gE = _e_and_grad(target, x)
    n_eval = n
    logw = torch.zeros(n, dtype=torch.float64)
    step = (0.5 * s0) ** 2 if step0 is None else step0
    b_prev = 0.0

    def logg(xx, EE, b):
        return -(1 - b) * 0.5 * (xx ** 2).sum(1) / s0 ** 2 - b * EE

    def grad_logg(xx, gg, b):
        return -(1 - b) * xx / s0 ** 2 - b * gg

    for b in betas:
        # AIS increment: log gamma_b(x) - log gamma_{b_prev}(x)
        inc = (b - b_prev) * (0.5 * (x ** 2).sum(1) / s0 ** 2 - E)
        logw += torch.nan_to_num(inc.double(), nan=-float('inf'), neginf=-float('inf'))
        b_prev = b
        # step size cap from the curvature of the prior part (keeps early levels stable)
        st = step
        gl = grad_logg(x, gE, b)
        xp = x + st * gl + math.sqrt(2 * st) * torch.randn(n, d, generator=generator, dtype=dtype)
        Ep, gEp = _e_and_grad(target, xp)
        n_eval += n
        glp = grad_logg(xp, gEp, b)
        lq_f = -((xp - x - st * gl) ** 2).sum(1) / (4 * st)
        lq_b = -((x - xp - st * glp) ** 2).sum(1) / (4 * st)
        la = logg(xp, Ep, b) - logg(x, E, b) + lq_b - lq_f
        la = torch.nan_to_num(la, nan=-float('inf'))
        acc = torch.log(torch.rand(n, generator=generator, dtype=dtype)) < la
        x = torch.where(acc[:, None], xp, x)
        E = torch.where(acc, Ep, E)
        gE = torch.where(acc[:, None], gEp, gE)
        a = float(acc.float().mean())
        step = step * math.exp(2.0 * (a - accept_target))
    if return_logw:
        return x, logw, n_eval
    return x, n_eval
