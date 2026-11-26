"""Aggregate results/main into a LaTeX table (final iteration, mean ± std over seeds) and learning-curve plots.

Usage: python experiments/aggregate.py
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TGT_NAME = {'gmm40': 'GMM-40 ($d{=}2$)', 'gmm16': 'GMM-10 ($d{=}16$)', 'mw32': 'MW-32 ($d{=}32$)',
            'funnel10': 'Funnel ($d{=}10$)'}
METH_NAME = {'idem': 'iDEM', 'aem': 'AEM (cold)', 'aem_warm': 'AEM (explore$\\to$anchor)',
             'aem_warm_replay': 'AEM-R (explore$\\to$anchor)'}
METRICS = {'gmm40': ['w2', 'energy_w1', 'mmd', 'mode_tv', 'within_var'],
           'gmm16': ['w2', 'energy_w1', 'mmd', 'mode_tv', 'within_var'],
           'mw32': ['w2', 'energy_w1', 'mmd', 'pattern_tv', 'within_var'], 'funnel10': ['w2', 'energy_w1', 'mmd']}
MNAME = {'w2': '$\\mathcal{W}_2$', 'energy_w1': '$\\mathcal{E}$-$\\mathcal{W}_1$', 'mmd': 'MMD',
         'mode_tv': 'mode TV', 'pattern_tv': 'mode TV', 'mode_coverage': 'cov.', 'within_var': 'var. ratio'}


_TARGETS = {}


def within_var(tgt, path):
    """Within-mode variance ratio of the final samples (truth = 1): GMM: mean ||x - mu_nearest||^2 / (d std^2);
    MW: variance of the Gaussian coordinates."""
    import torch
    from aem.targets import get_target
    if tgt not in _TARGETS:
        _TARGETS[tgt] = {'gmm40': lambda: get_target('gmm40'), 'gmm16': lambda: get_target('gmm_d', dim=16),
                         'mw32': lambda: get_target('mw32'), 'funnel10': lambda: None}[tgt]()
    t = _TARGETS[tgt]
    f = os.path.join(os.path.dirname(path), 'samples.npy')
    if t is None or not os.path.exists(f):
        return float('nan')
    x = np.load(f)
    x = x[np.isfinite(x).all(1)]
    if tgt == 'mw32':
        return float(x[:, 1::2].var())
    xt = torch.tensor(x, dtype=torch.float64)
    mu = t.means[t.mode_assign(xt)].numpy()
    return float(((x - mu) ** 2).sum(1).mean() / (t.dim * t.std**2))


def load():
    import sys
    sys.path.insert(0, ROOT)
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, 'results/main/*/*/seed*/log.jsonl'))):
        tgt, meth = p.split(os.sep)[-4:-2]
        recs = [json.loads(l) for l in open(p) if l.strip()]
        if recs and recs[-1]['iter'] == 40:
            recs[-1]['within_var'] = within_var(tgt, p)
            recs[-1]['var_err'] = abs(np.log(recs[-1]['within_var']))
            runs[(tgt, meth)].append(recs)
    return runs


def table(runs):
    tgts = [t for t in ['gmm40', 'gmm16', 'mw32', 'funnel10'] if any(k[0] == t for k in runs)]
    meths = [m for m in METH_NAME if any(k[1] == m for k in runs)]
    lines = []
    for t in tgts:
        ms = METRICS[t]
        lines.append(f'\\multicolumn{{{len(ms) + 1}}}{{l}}{{\\textbf{{{TGT_NAME[t]}}}}}\\\\')
        best = {}
        for m in ms:
            key = 'var_err' if m == 'within_var' else m  # var. ratio: best = closest to 1
            vals = {me: np.mean([r[-1][key] for r in runs[(t, me)]]) for me in meths if (t, me) in runs}
            best[m] = min(vals, key=vals.get)
        for me in meths:
            if (t, me) not in runs:
                continue
            cells = []
            for m in ms:
                v = np.array([r[-1][m] for r in runs[(t, me)]])
                s = f'{v.mean():.3g}{{\\scriptsize$\\pm${v.std():.2g}}}'
                cells.append(f'\\textbf{{{s}}}' if best[m] == me else s)
            lines.append(f'{METH_NAME[me]} & ' + ' & '.join(cells) + f' \\\\  % n={len(runs[(t, me)])}')
        lines.append('\\midrule')
    out = os.path.join(ROOT, 'paper', 'tab_main_body.tex')
    open(out, 'w').write('\n'.join(lines[:-1] + ['\\bottomrule']) + '\n')
    print(open(out).read())


def curves(runs):
    tgts = [t for t in ['gmm40', 'gmm16', 'mw32', 'funnel10'] if any(k[0] == t for k in runs)]
    cols = {'idem': '#c44e52', 'aem': '#8c8c8c', 'aem_warm': '#4c72b0', 'aem_warm_replay': '#55a868'}
    fig, axes = plt.subplots(1, len(tgts), figsize=(3.3 * len(tgts), 2.6))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tgts):
        for me, c in cols.items():
            if (t, me) not in runs:
                continue
            it = [r['iter'] for r in runs[(t, me)][0]]
            v = np.array([[r['energy_w1'] for r in rr] for rr in runs[(t, me)]])
            ax.plot(it, v.mean(0), color=c, label=METH_NAME[me].replace('$\\to$', '→'))
            ax.fill_between(it, v.min(0), v.max(0), color=c, alpha=.15)
        ax.set_yscale('log'); ax.set_title(TGT_NAME[t].replace('$', '').replace('{=}', '='), fontsize=9)
        ax.set_xlabel('outer iteration'); ax.grid(alpha=.3)
        ax.axvline(20, color='k', lw=.6, ls=':')
    axes[0].set_ylabel(r'energy $W_1$')
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'paper', 'fig_curves.pdf'))
    fig.savefig(os.path.join(ROOT, 'paper', 'fig_curves.png'), dpi=120)


if __name__ == '__main__':
    r = load()
    table(r)
    curves(r)
