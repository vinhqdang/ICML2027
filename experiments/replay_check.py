"""Estimator check for replay proposals: bias^2 / variance / MSE of iDEM, AEM (Gaussian proposals),
replay-SNIS (no anchor) and AEM+replay against the exact noised score, with exact or corrupted buffers.

Usage: python experiments/replay_check.py --dim 16 --out results/replay_check_d16.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aem.estimators import score_estimate  # noqa: E402
from aem.replay import score_estimate_replay  # noqa: E402
from aem.targets import GMM, get_target  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='gmm_d')
    p.add_argument('--dim', type=int, default=16)
    p.add_argument('--K', type=int, default=100)
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--R', type=int, default=16)
    p.add_argument('--tau_max', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--out', default='results/replay_check.csv')
    a = p.parse_args()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    kw = {} if a.target == 'gmm40' else {'dim': a.dim}
    pi = get_target(a.target, **kw)
    half = pi.n_comp // 2
    rhos = {'exact': pi,
            'skewed': GMM(torch.cat([pi.means[:half].repeat(4, 1), pi.means[half:]]), std=pi.std),
            'dropped': pi.mode_dropped()}
    sigmas = np.geomspace(0.1, 20.0 if a.target != 'gmm40' else 70.0, 8)
    rows = []
    for name, rho in rhos.items():
        for s in sigmas:
            sig = torch.full((a.n,), float(s), dtype=torch.float64)
            x0 = rho.sample(a.n, generator=gen)
            xt = x0 + s * torch.randn(a.n, pi.dim, generator=gen, dtype=torch.float64)
            truth = pi.noised_score(xt, sig)
            nrm = float((truth**2).sum(-1).mean())
            res = {k: [] for k in ('idem', 'aem', 'replay', 'aem_replay')}
            for r in range(a.R):
                anc = rho.posterior_sample(xt, sig, generator=gen)
                res['idem'].append(score_estimate(xt, sig, pi, a.K, anchor=None, kind='mix', generator=gen))
                res['aem'].append(score_estimate(xt, sig, pi, a.K, anchor=anc, kind='mix', generator=gen))
                res['replay'].append(score_estimate_replay(xt, sig, pi, a.K, ref=anc, anchor=None, kind='mix',
                                                           alpha=a.alpha, tau_max=a.tau_max, generator=gen))
                res['aem_replay'].append(score_estimate_replay(xt, sig, pi, a.K, ref=anc, anchor=anc, kind='mix',
                                                               alpha=a.alpha, tau_max=a.tau_max, generator=gen))
            line = f'{name:8s} sigma={s:7.3f}'
            for k, v in res.items():
                e = torch.stack(v)
                m = e.mean(0)
                var = float(((e - m) ** 2).sum(-1).mean())
                b2 = max(0.0, float(((m - truth) ** 2).sum(-1).mean()) - var / (a.R - 1))
                rows.append({'buffer': name, 'sigma': s, 'estimator': k, 'rel_bias2': b2 / nrm,
                             'rel_var': var / nrm, 'rel_mse': (b2 + var) / nrm})
                line += f'  {k}: mse={(b2 + var) / nrm:.2e} b2={b2 / nrm:.2e}'
            print(line, flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
