"""Estimator-level study: bias, variance and MSE of the noised-score estimators against the exact score.

For each noise level, x_t comes from the anchor distribution rho (exact target for 'exact', corrupted for the
others). The truth is always the exact target score at x_t. Each repeat draws fresh proposals and, for AEM, a fresh
anchor from rho(x0 | x_t), so the repeat average is an average over the anchor law as well.

Run: python -m aem.estimator_study --target gmm40 --out results/
"""
import argparse
import csv
import os

import numpy as np
import torch

from .estimators import score_estimate
from .targets import DEFAULTS, get_target

IDEM_CONFIGS = [('idem', K, kind) for K in (10, 100, 500) for kind in ('tsi', 'dsi')]
AEM_CONFIGS = [('aem', K, kind) for K in (0, 10, 100, 500) for kind in ('tsi', 'dsi')]
COLUMNS = ['target', 'estimator', 'K', 'kind', 'anchor_type', 'sigma', 'bias2', 'var', 'mse',
           'rel_bias2', 'rel_var', 'rel_mse']


def corrupted_targets(target):
    """Anchor distributions rho: 'exact' is the target itself, then the corrupted ones the target supports."""
    rhos = {'exact': target}
    if hasattr(target, 'tempered'):
        rhos['tempered'] = target.tempered()
    if hasattr(target, 'mode_dropped'):
        rhos['dropped'] = target.mode_dropped()
    return rhos


def run_study(target_name, out_dir='results', n=256, R=32, n_sigma=12, seed=0, dim=None, verbose=True):
    """Run the study and write out_dir/estimator_<target>.csv. Returns the list of row dicts."""
    kw = {} if dim is None else {'dim': dim}
    pi = get_target(target_name, **kw)
    if not (hasattr(pi, 'noised_score') and hasattr(pi, 'posterior_sample')):
        raise NotImplementedError(f'{target_name} has no exact noised score or posterior sampler')
    sigma_min, sigma_max, clip = DEFAULTS[target_name]
    sigmas = np.geomspace(sigma_min, sigma_max, n_sigma)
    gen = torch.Generator()
    gen.manual_seed(seed)
    dt = torch.float64

    rows = []
    for anchor_type, rho in corrupted_targets(pi).items():
        for sigma in sigmas:
            sig = torch.full((n,), float(sigma), dtype=dt)
            x0 = rho.sample(n, generator=gen).to(dt)
            x_t = x0 + float(sigma) * torch.randn(n, pi.dim, generator=gen, dtype=dt)
            truth = pi.noised_score(x_t, sig)
            norm2 = float((truth**2).sum(-1).mean())
            anchors = torch.stack([rho.posterior_sample(x_t, sig, generator=gen) for _ in range(R)])

            for estimator, K, kind in IDEM_CONFIGS + AEM_CONFIGS:
                ests = torch.stack([
                    score_estimate(x_t, sig, pi, K, anchor=anchors[r] if estimator == 'aem' else None,
                                   kind=kind, clip=clip, generator=gen)
                    for r in range(R)])
                mean = ests.mean(0)
                bias2 = float(((mean - truth) ** 2).sum(-1).mean())
                var = float(((ests - mean) ** 2).sum(-1).mean())
                mse = bias2 + var
                rows.append({
                    'target': target_name, 'estimator': estimator, 'K': K, 'kind': kind,
                    'anchor_type': anchor_type, 'sigma': float(sigma), 'bias2': bias2, 'var': var, 'mse': mse,
                    'rel_bias2': bias2 / norm2, 'rel_var': var / norm2, 'rel_mse': mse / norm2,
                })
            if verbose:
                best = {(r['estimator'], r['K'], r['kind']): r['rel_mse'] for r in rows[-len(IDEM_CONFIGS + AEM_CONFIGS):]}
                print(f'{anchor_type:9s} sigma={sigma:8.3f} idem10/tsi rel_mse={best[("idem", 10, "tsi")]:.3e}  '
                      f'aem100/tsi rel_mse={best[("aem", 100, "tsi")]:.3e}', flush=True)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'estimator_{target_name}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6e}' if isinstance(v, float) else v) for k, v in r.items()})
    if verbose:
        print(f'wrote {path}')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description='Estimator-level bias/variance study.')
    p.add_argument('--target', default='gmm40')
    p.add_argument('--out', default='results/')
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--R', type=int, default=32)
    p.add_argument('--n_sigma', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dim', type=int, default=None)
    a = p.parse_args(argv)
    run_study(a.target, out_dir=a.out, n=a.n, R=a.R, n_sigma=a.n_sigma, seed=a.seed, dim=a.dim)


if __name__ == '__main__':
    main()
