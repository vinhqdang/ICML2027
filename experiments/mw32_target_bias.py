"""Bias and variance of the iDEM training target on MW-32 against the exact noised score (quadrature).
For each sigma: N points x_t = x0 + sigma eps with x0 exact; R independent iDEM estimates per point (K=100, mix).
Reports relative squared bias (debiased for finite R) and relative variance, with and without the
10 sqrt(d) cap on ||sigma * target|| used in training. Usage: python experiments/mw32_target_bias.py"""
import os, sys, math
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from aem.targets import get_target
from aem.estimators import score_estimate
torch.set_num_threads(4)
t = get_target('mw32'); d = t.dim
N, R, K = 400, 64, int(sys.argv[1]) if len(sys.argv) > 1 else 100
g = torch.Generator().manual_seed(0)
x0 = t.sample(N, generator=g).double()
cap = 10 * math.sqrt(d)
print(f'K={K} N={N} R={R}')
for s in [0.1, 0.3, 1.0, 2.0]:
    xt = x0 + s * torch.randn(x0.shape, generator=g, dtype=torch.float64)
    truth = t.noised_score(xt, torch.full((N,), s, dtype=torch.float64))
    den = (truth ** 2).sum(-1).mean()
    for capped in (False, True):
        ests = []
        for r in range(R):
            e = score_estimate(xt, s, t, K, anchor=None, kind='mix', generator=torch.Generator().manual_seed(1000 * r + 1))
            if capped:
                n = (s * e).norm(dim=-1, keepdim=True)
                e = e * torch.clamp(cap / n.clamp_min(1e-30), max=1.0)
            ests.append(e)
        E = torch.stack(ests)  # R,N,d
        m = E.mean(0); var = E.var(0, unbiased=True).sum(-1).mean()
        bias2 = (((m - truth) ** 2).sum(-1).mean() - var / R) / den
        print(f'sigma={s} capped={capped}: rel_bias2={bias2:.4g} rel_var={var/den:.4g} rel_mse={bias2 + var/den:.4g}', flush=True)
