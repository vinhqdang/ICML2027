"""iDEM / AEM training of a score network on a Boltzmann target.

Run: python -m aem.train --target gmm40 --method aem --K 100
"""
import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .estimators import score_estimate
from .replay import score_estimate_replay
from .refine import refine_buffer
from .metrics import compute_all
from .model import EGNNScore, ScoreNet
from .sampler import sample_model
from .targets import DEFAULTS, get_target

REF_SEED = 12345  # fixed seed for the exact reference samples used in every evaluation


@dataclass
class TrainConfig:
    target: str = 'gmm40'
    dim: Optional[int] = None  # only for targets with a free dimension (mw32, funnel10, gmm_d)
    method: str = 'aem'  # 'idem' or 'aem'
    kind: str = 'tsi'  # 'tsi', 'dsi' or 'mix'
    K: int = 100
    n_isir: int = 1
    clip: Optional[float] = None  # None: default for the target; 0: no clipping
    refine: int = 0  # number of exact i-SIR Gibbs sweeps applied to the buffer after each outer iteration
    refine_K: int = 64
    refine_start: int = -1  # first outer iteration with refinement; -1: same as anchor_warmup + 1
    refine_sigma_max: Optional[float] = None  # None: sigma_max of the target
    refine_mode: str = 'full'  # 'full': aem.refine; 'subspace': aem.explore (whole-vector sweeps with walker
    #                            references, then coordinate-block sweeps)
    sub_blocks: int = 0  # subspace mode: block updates per chain in the local sweep (0: no local sweep)
    sub_bmax: int = 1  # subspace mode: max block size of the local sweep
    sub_K: int = 4  # subspace mode: proposals per local block update
    explore_walkers: int = 0  # >0: annealed-MALA exploration walkers per outer iteration, used as i-SIR references
    explore_levels: int = 200  # annealing levels (one MALA step each) per walker
    explore_frac: float = 0.5  # fraction of the reference set taken from the walkers
    explore_every: int = 1  # refresh the walker population every this many outer iterations
    no_replace: int = 0  # 1: sample minibatch rows without replacement
    oracle: int = 0  # 1: fixed buffer of exact target samples, no sampler loop (tests the fixed point directly)
    replay: int = 0  # 1: add replay (buffer-KDE) proposals, leave-one-out over the minibatch
    alpha: float = 0.5  # probability of a Gaussian (iDEM) proposal in the replay mixture
    tau_max: float = 1.0  # max KDE bandwidth; tau = min(sigma, tau_max)
    anchor_warmup: int = 0  # outer iterations with plain (unanchored) SNIS targets before anchoring
    anchor_ramp: int = 0  # then linearly ramp the anchor-use probability from 0 to 1 over this many iterations
    device: str = 'cpu'
    out_clip: Optional[float] = None  # cap on ||sigma*s|| for targets and sampling; None: 10*sqrt(d); 0: off
    sigma_min: Optional[float] = None  # None: default for the target
    sigma_max: Optional[float] = None
    seed: int = 0
    buffer_size: int = 10000
    n_outer: int = 100
    n_inner: int = 200
    batch_size: int = 256
    n_new: int = 1000
    lr: float = 5e-4
    sde_steps: int = 500
    sde_steps_eval: int = 1000
    n_eval: int = 2000
    eval_every: int = 10
    hidden: int = 256
    n_layers: int = 4
    time_emb: int = 64
    arch: str = 'mlp'  # 'mlp': ScoreNet on the flat vector; 'egnn': EGNNScore (molecular targets)
    out_dir: str = 'results/train'


_INT_FIELDS = {'refine', 'sub_blocks', 'sub_K', 'sub_bmax', 'explore_walkers', 'explore_levels', 'explore_every', 'refine_K', 'refine_start', 'no_replace', 'oracle', 'replay', 'anchor_warmup', 'anchor_ramp', 'dim', 'K', 'n_isir', 'seed', 'buffer_size', 'n_outer', 'n_inner', 'batch_size', 'n_new',
               'sde_steps', 'sde_steps_eval', 'n_eval', 'eval_every', 'hidden', 'n_layers', 'time_emb'}


def _json_safe(d):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}


def _sde(net, n, dim, smin, smax, steps, gen, batch_size=10000, score_clip=None, project=None):
    x = sample_model(net, n, dim, smin, smax, n_steps=steps, generator=gen, batch_size=batch_size,
                     score_clip=score_clip, project=project)
    return x[torch.isfinite(x).all(dim=1)]


def train(cfg=None, **overrides):
    """Train a ScoreNet with iDEM (method='idem') or AEM (method='aem') targets. Returns a summary dict."""
    if cfg is None:
        cfg = TrainConfig(**overrides)
    elif isinstance(cfg, dict):
        cfg = TrainConfig(**{**cfg, **overrides})
    elif overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    if cfg.method not in ('idem', 'aem'):
        raise ValueError(f'method must be idem or aem, got {cfg.method!r}')
    if cfg.arch not in ('mlp', 'egnn'):
        raise ValueError(f'arch must be mlp or egnn, got {cfg.arch!r}')
    anchored = cfg.method == 'aem'

    d_smin, d_smax, d_clip = DEFAULTS[cfg.target]
    smin = cfg.sigma_min if cfg.sigma_min is not None else d_smin
    smax = cfg.sigma_max if cfg.sigma_max is not None else d_smax
    clip = d_clip if cfg.clip is None else (cfg.clip or None)
    log_ratio = math.log(smax / smin)

    torch.set_default_device(cfg.device)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=torch.get_default_device())
    gen.manual_seed(cfg.seed)
    target = get_target(cfg.target, **({} if cfg.dim is None else {'dim': cfg.dim}))
    d = target.dim
    oc = 10.0 * math.sqrt(d) if cfg.out_clip is None else (cfg.out_clip or None)

    project = target.project  # identity for free targets; zero-centre-of-mass for molecular targets
    if cfg.arch == 'egnn':
        if not hasattr(target, 'n_particles'):
            raise ValueError(f'arch egnn needs a molecular target, got {cfg.target!r}')
        net = EGNNScore(target.n_particles, target.spatial_dim, cfg.hidden, cfg.n_layers, cfg.time_emb,
                        sigma_min=smin, sigma_max=smax)
    else:
        net = ScoreNet(d, cfg.hidden, cfg.n_layers, cfg.time_emb, sigma_min=smin, sigma_max=smax)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    ref = target.sample(cfg.n_eval, seed=REF_SEED).cpu().numpy()

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, 'config.json'), 'w') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    n_evals = 0
    t0 = time.time()
    if cfg.oracle:
        buf = target.sample(cfg.buffer_size, seed=cfg.seed + 777).to(torch.float32)
    else:
        buf = _sde(net, cfg.n_new, d, smin, smax, cfg.sde_steps, gen, score_clip=oc, project=project)  # buffer from the untrained network
    last_samples = None
    walkers = None
    summary = {}
    log_path = os.path.join(cfg.out_dir, 'log.jsonl')
    with open(log_path, 'w') as log:
        for it in range(1, cfg.n_outer + 1):
            losses = []
            net.train()
            for _ in range(cfg.n_inner):
                B = cfg.batch_size
                if cfg.no_replace:  # without replacement: exact leave-one-out for replay proposals
                    idx = torch.randperm(buf.shape[0], generator=gen)[:B]
                else:
                    idx = torch.randint(buf.shape[0], (B,), generator=gen)
                x0 = buf[idx]
                t = torch.rand(B, generator=gen, dtype=x0.dtype)
                sig = smin * torch.exp(t * log_ratio)
                eps = project(torch.randn(B, d, generator=gen, dtype=x0.dtype))
                x_t = x0 + sig[:, None] * eps
                p_anchor = 0.0
                if anchored:
                    p_anchor = 1.0 if cfg.anchor_ramp == 0 else (it - cfg.anchor_warmup) / cfg.anchor_ramp
                    p_anchor = float(min(1.0, max(0.0, p_anchor))) if it > cfg.anchor_warmup else 0.0
                if cfg.replay and p_anchor > 0.0:  # replay proposals only once anchoring is active
                    if p_anchor >= 1.0:
                        anc = x0
                    elif p_anchor <= 0.0:
                        anc = None
                    else:
                        anc = None if float(torch.rand(1, generator=gen)) > p_anchor else x0
                    tgt = score_estimate_replay(x_t, sig, target, cfg.K, ref=x0, anchor=anc, kind=cfg.kind,
                                                alpha=cfg.alpha, tau_max=cfg.tau_max, clip=clip, generator=gen)
                elif p_anchor >= 1.0:
                    tgt = score_estimate(x_t, sig, target, cfg.K, anchor=x0, kind=cfg.kind, n_isir=cfg.n_isir,
                                         clip=clip, generator=gen)
                elif p_anchor <= 0.0:
                    tgt = score_estimate(x_t, sig, target, cfg.K, anchor=None, kind=cfg.kind, clip=clip,
                                         generator=gen)
                else:  # per-sample anchor use with probability p_anchor
                    use = torch.rand(B, generator=gen) < p_anchor
                    tgt = torch.empty_like(x_t)
                    if use.any():
                        tgt[use] = score_estimate(x_t[use], sig[use], target, cfg.K, anchor=x0[use], kind=cfg.kind,
                                                  n_isir=cfg.n_isir, clip=clip, generator=gen)
                    if (~use).any():
                        tgt[~use] = score_estimate(x_t[~use], sig[~use], target, cfg.K, anchor=None, kind=cfg.kind,
                                                   clip=clip, generator=gen)
                if oc is not None:  # cap ||sigma * target|| (same rule for every method)
                    nrm = sig[:, None] * tgt.norm(dim=-1, keepdim=True)
                    tgt = tgt * torch.clamp(oc / nrm.clamp_min(1e-30), max=1.0)
                s = net.score(x_t, t)
                loss = ((sig[:, None] ** 2) * (s - tgt) ** 2).sum(1).mean() / d
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
                n_evals += B * (cfg.K + p_anchor) * (cfg.n_isir if p_anchor > 0 else 1)

            if not cfg.oracle:
                new = project(_sde(net, cfg.n_new, d, smin, smax, cfg.sde_steps, gen, score_clip=oc, project=project))
                buf = torch.cat([buf, new])[-cfg.buffer_size:]
            r_start = cfg.refine_start if cfg.refine_start >= 0 else cfg.anchor_warmup + 1
            if cfg.refine > 0 and it >= r_start and cfg.refine_mode == 'subspace':
                from .explore import annealed_walkers, refine_buffer_explore
                if cfg.explore_walkers > 0 and (walkers is None or (it - r_start) % cfg.explore_every == 0):
                    walkers, n_w = annealed_walkers(target, cfg.explore_walkers, smax, n_levels=cfg.explore_levels,
                                                    generator=gen, dtype=buf.dtype)
                    n_evals += n_w
                kw = dict(alpha=cfg.alpha, tau_max=cfg.tau_max, sigma_min=smin, sigma_max=cfg.refine_sigma_max or smax,
                          explore_frac=cfg.explore_frac, generator=gen)
                buf, _, n_r = refine_buffer_explore(target, buf, walkers, n_sweeps=cfg.refine, K=cfg.refine_K,
                                                    p_full=1.0, **kw)
                n_evals += n_r
                if cfg.sub_blocks > 0:
                    buf, _, n_r = refine_buffer_explore(target, buf, walkers, n_sweeps=1, K=cfg.sub_K, p_full=0.0,
                                                        b_max=cfg.sub_bmax, n_blocks=cfg.sub_blocks, **kw)
                    n_evals += n_r
            elif cfg.refine > 0 and it >= r_start:
                buf, _ = refine_buffer(target, buf, n_sweeps=cfg.refine, K=cfg.refine_K, alpha=cfg.alpha,
                                       tau_max=cfg.tau_max, sigma_min=smin,
                                       sigma_max=cfg.refine_sigma_max or smax, generator=gen)
                n_evals += cfg.refine * buf.shape[0] * (cfg.refine_K + 1)

            if it % cfg.eval_every == 0 or it == cfg.n_outer:
                net.eval()
                eval_gen = torch.Generator(device=torch.get_default_device())
                eval_gen.manual_seed(cfg.seed + 1)
                last_samples = _sde(net, cfg.n_eval, d, smin, smax, cfg.sde_steps_eval, eval_gen, score_clip=oc,
                                    project=project).cpu().numpy()
                metrics = compute_all(target, last_samples, ref)
                rec = {'iter': it, 'wall_time': time.time() - t0, 'n_energy_evals': n_evals,
                       'loss': float(np.mean(losses)), 'buffer_size': int(buf.shape[0]), **metrics}
                log.write(json.dumps(_json_safe(rec)) + '\n')
                log.flush()
                summary = rec
                print(json.dumps(_json_safe(rec)), flush=True)

    np.save(os.path.join(cfg.out_dir, 'samples.npy'), last_samples)
    torch.save(net.state_dict(), os.path.join(cfg.out_dir, 'model.pt'))
    return {'final': summary, 'log': log_path, 'out_dir': cfg.out_dir}


def _build_parser():
    p = argparse.ArgumentParser(description='Train iDEM/AEM score networks.')
    for f in dataclasses.fields(TrainConfig):
        default = f.default
        if f.name in _INT_FIELDS:
            typ = int
        elif isinstance(default, str):
            typ = str
        elif isinstance(default, bool):
            typ = bool
        else:
            typ = float
        if f.name == 'K':
            typ = int
        p.add_argument(f'--{f.name}', type=typ, default=default)
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == '__main__':
    main()
