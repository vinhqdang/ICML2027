"""Relative error of the learned score against the exact noised score, for oracle-buffer models.
Usage: python experiments/oracle_score_error.py mw32 results/oracle/mw32/*_K*"""
import json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aem.model import ScoreNet
from aem.targets import get_target

name = sys.argv[1]
kw = {'dim': 16} if name == 'gmm16' else {}
t = get_target('gmm_d' if name == 'gmm16' else name, **kw)
g = torch.Generator().manual_seed(0)
x0 = t.sample(2000, generator=g)
rows = []
for run in sys.argv[2:]:
    cfg = json.load(open(os.path.join(run, 'config.json')))
    from aem.targets import DEFAULTS
    smin, smax, _ = DEFAULTS[cfg['target']]
    net = ScoreNet(t.dim, cfg['hidden'], cfg['n_layers'], cfg['time_emb'], sigma_min=smin, sigma_max=smax)
    net.load_state_dict(torch.load(os.path.join(run, 'model.pt')))
    net.eval()
    out = [os.path.basename(run)]
    for s in [0.03, 0.1, 0.3, 1.0, 2.0]:
        if s > smax: continue
        sig = torch.full((x0.shape[0],), s, dtype=torch.float64)
        xt = x0 + s * torch.randn(x0.shape, generator=g, dtype=torch.float64)
        tt = torch.log(sig / smin) / np.log(smax / smin)
        with torch.no_grad():
            pred = net.score(xt.float(), tt.float()).double()
        truth = t.noised_score(xt, sig)
        err = float(((pred - truth) ** 2).sum(-1).mean() / (truth ** 2).sum(-1).mean())
        out.append(f'sigma={s}: {err:.6e}')
        rows.append((os.path.basename(run), s, err))
    print('  '.join(out))
