"""Held-out denoising score matching (DSM) loss of the oracle-buffer models on DW-4 and LJ-13.

For a model s_theta and a noise level sigma, with x0 from a held-out set, eps ~ P(N(0, I)), x_t = x0 + sigma eps:
    L(sigma) = mean_i || s_theta(x_t_i, sigma) - (x0_i - x_t_i) / sigma^2 ||^2
    Ltilde(sigma) = sigma^2 L(sigma) / d_eff,   d_eff = effective_dim(target)
Every model sees the same x0 and the same eps (fixed seeds per sigma), so differences between models are paired.

Held-out set. The oracle buffer of seed S is target.sample(10000, seed=S + 777), which is ref[idx_S] with
idx_S = randperm(M, generator=seeded(S + 777))[:10000] over the M cached reference rows (aem/targets.py,
_Molecular.sample). The script recomputes idx_S, asserts that ref[idx_S] equals target.sample(...) exactly, forms the
union U over the seeds used (0, 1, 2), and draws the held-out indices uniformly from the complement of U.

Usage: python experiments/oracle_dsm_eval.py
Writes results/oracle/oracle_dsm.csv (target,method,seed,sigma,Ltilde) and results/oracle/oracle_samples_last.csv.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aem.model import EGNNScore, ScoreNet  # noqa: E402
from aem.targets import DEFAULTS, _ref_path, effective_dim, get_target  # noqa: E402

SEEDS = [0, 1, 2]
METHODS = ['idem', 'aem']
GRIDS = {'dw4': [0.03, 0.1, 0.3, 1.0, 2.0, 3.0], 'lj13': [0.01, 0.03, 0.1, 0.3, 1.0, 2.0]}
BUFFER_SIZE = 10000
N_HELD = 5000
HELDOUT_SEED = 4242
OUT_DIR = os.path.join(ROOT, 'results', 'oracle')


def buffer_indices(M, seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.randperm(M, generator=g)[:BUFFER_SIZE]


def heldout_indices(target, ref):
    M = ref.shape[0]
    used = []
    for S in SEEDS:
        idx = buffer_indices(M, S + 777)
        assert torch.equal(ref[idx], target.sample(BUFFER_SIZE, seed=S + 777).to(ref.dtype)), 'index replay mismatch'
        used.append(idx)
    U = torch.unique(torch.cat(used))
    mask = torch.ones(M, dtype=torch.bool)
    mask[U] = False
    comp = torch.nonzero(mask).flatten()
    g = torch.Generator()
    g.manual_seed(HELDOUT_SEED)
    held = comp[torch.randperm(comp.shape[0], generator=g)[:N_HELD]]
    assert not np.isin(held.numpy(), U.numpy()).any()
    return held, U


def load_model(run_dir, target, cfg):
    smin, smax, _ = DEFAULTS[cfg['target']]
    smin = cfg['sigma_min'] if cfg['sigma_min'] is not None else smin
    smax = cfg['sigma_max'] if cfg['sigma_max'] is not None else smax
    if cfg['arch'] == 'egnn':
        net = EGNNScore(target.n_particles, target.spatial_dim, cfg['hidden'], cfg['n_layers'], cfg['time_emb'],
                        sigma_min=smin, sigma_max=smax)
    else:
        net = ScoreNet(target.dim, cfg['hidden'], cfg['n_layers'], cfg['time_emb'], sigma_min=smin, sigma_max=smax)
    net.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu'))
    net.eval()
    return net, smin, smax


def dsm_loss(net, x0, sigmas, smin, smax, target, seed_base):
    """Ltilde per sigma for one model on the fixed held-out set."""
    out = {}
    d_eff = effective_dim(target)
    for k, s in enumerate(sigmas):
        g = torch.Generator()
        g.manual_seed(seed_base + k)
        eps = target.project(torch.randn(x0.shape, generator=g, dtype=torch.float64))
        sig = torch.full((x0.shape[0],), float(s), dtype=torch.float64)
        x_t = x0 + sig[:, None] * eps
        r = (x0 - x_t) / sig[:, None] ** 2
        t = torch.log(sig / smin) / np.log(smax / smin)
        with torch.no_grad():
            pred = net.score(x_t.float(), t.float()).double()
        L = float(((pred - r) ** 2).sum(-1).mean())
        out[s] = float(s * s * L / d_eff)
    return out


def main():
    torch.set_default_dtype(torch.float32)
    rows, samp = [], []
    for tname in ['dw4', 'lj13']:
        target = get_target(tname)
        ref = torch.load(_ref_path(tname), map_location='cpu', weights_only=True)['samples'].to(torch.float64)
        held, U = heldout_indices(target, ref)
        x0 = ref[held].clone()
        print(f'{tname}: cache rows {ref.shape[0]}, union of buffer indices {U.numel()}, '
              f'held-out {held.numel()} (disjoint: True), d_eff {effective_dim(target)}', flush=True)
        for method in METHODS:
            for S in SEEDS:
                run = os.path.join(OUT_DIR, tname, f'{method}_K100_s{S}')
                if not os.path.isfile(os.path.join(run, 'model.pt')):
                    print(f'  missing {run}, skipped', flush=True)
                    continue
                cfg = json.load(open(os.path.join(run, 'config.json')))
                net, smin, smax = load_model(run, target, cfg)
                res = dsm_loss(net, x0, GRIDS[tname], smin, smax, target, seed_base=31337)
                for s in GRIDS[tname]:
                    rows.append({'target': tname, 'method': method, 'seed': S, 'sigma': s, 'Ltilde': res[s]})
                last = json.loads(open(os.path.join(run, 'log.jsonl')).read().strip().splitlines()[-1])
                samp.append({'target': tname, 'method': method, 'seed': S, 'iter': last['iter'],
                             'energy_w1': last['energy_w1'], 'dist_w1': last['dist_w1'], 'w2': last['w2'],
                             'mmd': last['mmd']})
                print(f'  {method} s{S}: ' + ' '.join(f'{s}:{res[s]:.4e}' for s in GRIDS[tname]), flush=True)
    with open(os.path.join(OUT_DIR, 'oracle_dsm.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['target', 'method', 'seed', 'sigma', 'Ltilde'])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, 'oracle_samples_last.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(samp[0].keys()))
        w.writeheader()
        w.writerows(samp)
    summarise(rows)


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    lines = []
    for tname in df.target.unique():
        d = df[df.target == tname]
        lines.append(f'== {tname}: Ltilde mean +- std (ddof=1) over seeds; paired diff = iDEM - AEM ==')
        lines.append(f'{"sigma":>6} {"iDEM":>22} {"AEM":>22} {"dL (mean+-std)":>22} {"seeds dL>0":>11}')
        for s in sorted(d.sigma.unique()):
            ds = d[d.sigma == s]
            i = ds[ds.method == 'idem'].sort_values('seed')
            a = ds[ds.method == 'aem'].sort_values('seed')
            n = min(len(i), len(a))
            diff = i.Ltilde.values[:n] - a.Ltilde.values[:n]
            sd = lambda v: v.std(ddof=1) if len(v) > 1 else float('nan')
            lines.append(f'{s:>6} {i.Ltilde.mean():>10.4e}+-{sd(i.Ltilde.values):<9.2e} '
                         f'{a.Ltilde.mean():>10.4e}+-{sd(a.Ltilde.values):<9.2e} '
                         f'{diff.mean():>10.4e}+-{sd(diff):<9.2e} {int((diff > 0).sum())}/{n}')
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(OUT_DIR, 'oracle_dsm_summary.txt'), 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
