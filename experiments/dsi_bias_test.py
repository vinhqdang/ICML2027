"""Population unbiasedness test of the iDEM / AEM training targets.

For x0 ~ pi, eps ~ N(0, I), x_t = x0 + sigma * P(eps) (P = target.project), r = (x0 - x_t) / sigma^2.
The true noised score s satisfies E[s(x_t) h(x_t)] = E[r h(x_t)] for any h, so an estimator T is conditionally
unbiased only if D_h = E[(T - r) . h(x_t)] = 0. Per pair z_i = (T_i - r_i) . h_i is averaged; the SE is the sample
SE of z, zscore = mean / SE, normalised = mean / mean_i |r_i . h_i|.

Usage: python experiments/dsi_bias_test.py --target dw4 --sigmas 0.03 0.1 0.3 1 2 3 --out results/dsi_bias/dsi_bias.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aem.estimators import score_estimate  # noqa: E402
from aem.targets import get_target  # noqa: E402

COLUMNS = ['target', 'sigma', 'estimator', 'K', 'h', 'N', 'mean', 'se', 'zscore', 'normalised']
CLIP_H = 1000.0


def _grad_neg_energy(target, x):
    """-grad E(x) at x (float64), [N, d]."""
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        (g,) = torch.autograd.grad(target.energy(x).sum(), x)
    return -g.detach()


def _clip_rows(v, c):
    nrm = v.norm(dim=-1, keepdim=True)
    return v * torch.clamp(c / nrm.clamp_min(1e-300), max=1.0)


def run_target(name, sigmas, N, out_path, seed_x0=12345, seed_eps=2024, seed_est=7):
    target = get_target('gmm_d', dim=16) if name == 'gmm16' else get_target(name)
    tname = name
    clip = 1000.0 if name == 'lj13' else None  # training clip for LJ-13, none for DW-4 and gmm16
    x0 = target.sample(N, seed=seed_x0).to(torch.float64)
    ests = [('idem', None, 100), ('aem', 'x0', 100)]
    if name == 'dw4':
        ests.append(('idem', None, 1000))
    has_exact = hasattr(target, 'noised_score')
    if has_exact:
        ests.append(('exact', 'exact', 0))
    rows = []
    for sig_v in sigmas:
        t0 = time.time()
        sig = torch.full((N,), float(sig_v), dtype=torch.float64)
        g = torch.Generator().manual_seed(seed_eps)
        eps = target.project(torch.randn(N, target.dim, generator=g, dtype=torch.float64))
        x_t = x0 + sig[:, None] * eps
        r = (x0 - x_t) / sig[:, None] ** 2
        hs = {'pos': target.project(x_t), 'grad': _clip_rows(target.project(_grad_neg_energy(target, x_t)), CLIP_H)}
        for est_name, anc_flag, K in ests:
            if anc_flag == 'exact':
                T = target.noised_score(x_t, sig).to(torch.float64)
            else:
                gen = torch.Generator().manual_seed(seed_est + K + (1 if anc_flag else 0))
                anchor = x0 if anc_flag == 'x0' else None
                kind = 'mix'
                T = score_estimate(x_t, sig, target, K, anchor=anchor, kind=kind, clip=clip, generator=gen)
                T = T.to(torch.float64)
            D = T - r
            for h_name, h in hs.items():
                z = (D * h).sum(-1).numpy()
                ref = ((r * h).sum(-1)).abs().numpy()
                mean = float(z.mean())
                se = float(z.std(ddof=1) / np.sqrt(N))
                rows.append({'target': tname, 'sigma': sig_v, 'estimator': est_name, 'K': K if K else '',
                             'h': h_name, 'N': N, 'mean': mean, 'se': se,
                             'zscore': mean / se if se > 0 else float('nan'),
                             'normalised': mean / float(ref.mean())})
                print(f'{tname} sigma={sig_v} {est_name} K={K or "-"} h={h_name}: mean={mean:.4e} '
                      f'se={se:.2e} z={rows[-1]["zscore"]:.2f} norm={rows[-1]["normalised"]:.3e}', flush=True)
        print(f'[{tname} sigma={sig_v}] {time.time() - t0:.1f}s', flush=True)
    new = not os.path.exists(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--target', required=True, choices=['gmm16', 'dw4', 'lj13'])
    p.add_argument('--sigmas', type=float, nargs='+', required=True)
    p.add_argument('--N', type=int, default=20000)
    p.add_argument('--out', default=os.path.join(ROOT, 'results', 'dsi_bias', 'dsi_bias.csv'))
    a = p.parse_args()
    torch.set_default_dtype(torch.float32)
    run_target(a.target, a.sigmas, a.N, a.out)
