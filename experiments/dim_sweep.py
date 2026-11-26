"""Estimator-level dimension sweep with the training test function ('mix'): bias^2 / MSE of iDEM and AEM (exact
anchors) on GMM-10 in d dimensions, plus the mean anchor weight and ESS of AEM.

Usage: python experiments/dim_sweep.py --out results/dim_sweep.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aem.estimators import score_estimate  # noqa: E402
from aem.targets import GMM  # noqa: E402


def make_gmm(d, n_comp=10, scale=10.0, std=1.5, seed=0):
    """Mixture with NON-unit component std (1.5), so the 'mix' test function is not exact within a component."""
    g = torch.Generator().manual_seed(seed)
    means = (torch.rand(n_comp, d, generator=g, dtype=torch.float64) * 2 - 1) * scale
    return GMM(means, std=std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dims', default='2,8,16,32')
    p.add_argument('--K', type=int, default=100)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--R', type=int, default=16)
    p.add_argument('--out', default='results/dim_sweep.csv')
    a = p.parse_args()
    gen = torch.Generator().manual_seed(0)
    rows = []
    for d in [int(x) for x in a.dims.split(',')]:
        pi = make_gmm(d)
        for s in [0.3, 1.0, 3.0, 10.0]:
            sig = torch.full((a.n,), s, dtype=torch.float64)
            x0 = pi.sample(a.n, generator=gen)
            xt = x0 + s * torch.randn(a.n, d, generator=gen, dtype=torch.float64)
            truth = pi.noised_score(xt, sig)
            nrm = float((truth ** 2).sum(-1).mean())
            res = {'idem': [], 'aem': []}
            aw, ess = [], []
            for _ in range(a.R):
                anc = pi.posterior_sample(xt, sig, generator=gen)
                res['idem'].append(score_estimate(xt, sig, pi, a.K, kind='mix', generator=gen))
                e, dg = score_estimate(xt, sig, pi, a.K, anchor=anc, kind='mix', generator=gen, return_diag=True)
                res['aem'].append(e)
                aw.append(float(dg['anchor_weight'].mean()))
                ess.append(float(dg['ess'].mean()))
            line = f'd={d:3d} sigma={s:5.1f}'
            for k, v in res.items():
                E = torch.stack(v)
                m = E.mean(0)
                var = float(((E - m) ** 2).sum(-1).mean())
                b2 = max(0.0, float(((m - truth) ** 2).sum(-1).mean()) - var / (a.R - 1))
                rows.append({'d': d, 'sigma': s, 'estimator': k, 'rel_bias2': b2 / nrm, 'rel_mse': (b2 + var) / nrm,
                             'anchor_weight': np.mean(aw) if k == 'aem' else np.nan,
                             'ess': np.mean(ess) if k == 'aem' else np.nan})
                line += f'  {k}: mse={(b2 + var) / nrm:.2e} b2={b2 / nrm:.2e}'
            line += f'  anchor_w={np.mean(aw):.3f} ess={np.mean(ess):.1f}'
            print(line, flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
