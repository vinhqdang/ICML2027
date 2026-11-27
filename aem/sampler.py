"""Reverse-time VE SDE sampler (Euler-Maruyama) for sigma(t) = sigma_min * (sigma_max / sigma_min)**t."""
import math

import torch


@torch.no_grad()
def reverse_sde_sample(score_fn, n, dim, sigma_min, sigma_max, n_steps=1000, generator=None,
                       batch_size=10000, score_clip=None, dtype=torch.float32, project=None):
    """Integrate the reverse VE SDE from t = 1 to t = 0.

    score_fn(x, t) takes x [b, dim] and t [b] and returns the score [b, dim].
    Start: x ~ N(0, sigma_max^2 I). Update backward in time:
        x <- x + g^2(t) s(x, t) dt + g(t) sqrt(dt) z,   g^2(t) = 2 sigma(t)^2 log(sigma_max / sigma_min),
    with no noise on the final step. score_clip, if given, caps the norm of sigma(t) * s per sample.
    project, if given, maps the initial noise, the per-step noise and the score onto the sampling space (e.g. the
    zero-centre-of-mass subspace). None is the identity and draws exactly the same random numbers.
    """
    proj = project if project is not None else (lambda z: z)
    log_ratio = math.log(sigma_max / sigma_min)
    dt = 1.0 / n_steps
    outs = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        x = proj(sigma_max * torch.randn(b, dim, generator=generator, dtype=dtype))
        for k in range(n_steps):
            t = 1.0 - k * dt
            s = proj(score_fn(x, torch.full((b,), t, dtype=dtype)))
            sig = sigma_min * (sigma_max / sigma_min) ** t
            if score_clip is not None:  # clip the preconditioned score sigma * s (scale-free)
                nrm = sig * s.norm(dim=-1, keepdim=True)
                s = s * torch.clamp(score_clip / nrm.clamp_min(1e-30), max=1.0)
            g2 = 2.0 * sig**2 * log_ratio
            x = x + g2 * s * dt
            if k < n_steps - 1:
                x = x + math.sqrt(g2 * dt) * proj(torch.randn(b, dim, generator=generator, dtype=dtype))
        outs.append(x)
    return torch.cat(outs) if outs else torch.empty(0, dim, dtype=dtype)


@torch.no_grad()
def sample_model(net, n, dim, sigma_min, sigma_max, n_steps=1000, generator=None, batch_size=10000,
                 score_clip=None, project=None):
    """Samples from a ScoreNet under the reverse VE SDE."""
    was_training = net.training
    net.eval()
    try:
        return reverse_sde_sample(net.score, n, dim, sigma_min, sigma_max, n_steps=n_steps,
                                  generator=generator, batch_size=batch_size, score_clip=score_clip,
                                  project=project)
    finally:
        net.train(was_training)
