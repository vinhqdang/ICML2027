"""Aggregate results/main_gpu/<target>/<method>/seed<k>/ into LaTeX tables, a JSON summary and learning curves.

Final-iteration metrics: mean and sample standard deviation (ddof=1) over seeds. Extra statistics computed from
samples.npy: within-mode variance ratio (gmm16: mean ||x - mu_nearest||^2 / (d std^2); mw32: variance of the Gaussian
coordinates) and, for mw32, the fraction of double-well coordinates in the right (major) well.
Outputs: paper/tab_gpu_free.tex (gmm16, mw32), paper/tab_gpu_mol.tex (dw4, lj13), results/main_gpu/summary.json,
paper/fig_curves_gpu.pdf.   Usage: python experiments/aggregate_gpu.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASE = os.path.join(ROOT, 'results', 'main_gpu')
FINAL = {'gmm16': 80, 'mw32': 80, 'dw4': 40, 'lj13': 30, 'oracle_dw4': 40, 'oracle_lj13': 30}
TGT_NAME = {'gmm16': 'GMM-10 ($d{=}16$)', 'mw32': 'MW-32 ($d{=}32$)', 'dw4': 'DW-4 ($d{=}8$)', 'lj13': 'LJ-13 ($d{=}39$)'}
METH_NAME = {'idem': 'iDEM', 'aem_cold': 'AEM (cold)', 'aem_ea': 'AEM (explore$\\to$anchor)',
             'aem_ea_refine': 'AEM (explore$\\to$anchor) + refine', 'aem': 'AEM'}
METRICS = {'gmm16': ['w2', 'energy_w1', 'mmd', 'mode_tv', 'within_var'],
           'mw32': ['w2', 'energy_w1', 'mmd', 'pattern_tv', 'within_var'],
           'dw4': ['energy_w1', 'energy_w1_trim', 'dist_w1', 'bad_frac', 'w2', 'mmd'],
           'lj13': ['energy_w1', 'energy_w1_trim', 'dist_w1', 'bad_frac', 'w2', 'mmd']}
METRICS['oracle_dw4'] = METRICS['dw4']
METRICS['oracle_lj13'] = METRICS['lj13']
TGT_NAME['oracle_dw4'] = 'DW-4, oracle buffer'
TGT_NAME['oracle_lj13'] = 'LJ-13, oracle buffer'
TABLE_METRICS = {'oracle_dw4': ['energy_w1', 'energy_w1_trim', 'dist_w1', 'bad_frac'], 'oracle_lj13': ['energy_w1', 'energy_w1_trim', 'dist_w1', 'bad_frac'], 'dw4': ['energy_w1', 'energy_w1_trim', 'dist_w1', 'bad_frac'], 'lj13': ['energy_w1', 'energy_w1_trim', 'dist_w1', 'bad_frac']}
MNAME = {'w2': '$\\mathcal{W}_2$', 'energy_w1': '$\\mathcal{E}$-$\\mathcal{W}_1$', 'mmd': 'MMD', 'mode_tv': 'mode TV',
         'pattern_tv': 'mode TV', 'within_var': 'var.\\ ratio', 'major_well': 'major well', 'dist_w1': '$d$-$\\mathcal{W}_1$',
         'energy_w1_trim': '$\\mathcal{E}$-$\\mathcal{W}_1^{\\mathrm{tr}}$', 'bad_frac': 'frac.\\ high-$\\mathcal{E}$'}
TRUE_MAJOR_WELL = None  # computed from exact samples
_T = {}


def target(tgt):
    from aem.targets import get_target
    if tgt not in _T:
        _T[tgt] = get_target('gmm_d', dim=16) if tgt == 'gmm16' else get_target(tgt)
    return _T[tgt]


def extra_stats(tgt, run_dir):
    """within_var and major_well from the final samples (nan if not applicable)."""
    import torch
    out = {}
    f = os.path.join(run_dir, 'samples.npy')
    if not os.path.exists(f):
        return out
    tgt = tgt.replace('oracle_', '')
    if tgt in ('dw4', 'lj13'):  # robust energy statistics: samples with energy above the 99.9% quantile of the reference
        from scipy.stats import wasserstein_distance
        t = target(tgt)
        er = t.energy(t.sample(100000, seed=1)).numpy()
        thr = float(np.quantile(er, 0.999))
        x = np.load(f)
        x = x[np.isfinite(x).all(1)]
        em = t.energy(torch.tensor(x, dtype=torch.float64)).numpy()
        bad = em > thr
        out['bad_frac'] = float(bad.mean())
        out['energy_w1_trim'] = float(wasserstein_distance(em[~bad], er)) if (~bad).any() else float('nan')
        return out
    if tgt not in ('gmm16', 'mw32'):
        return out
    x = np.load(f)
    x = x[np.isfinite(x).all(1)]
    t = target(tgt)
    if tgt == 'mw32':
        out['within_var'] = float(x[:, 1::2].var())
        out['major_well'] = float((x[:, 0::2] >= 0).mean())
    else:
        xt = torch.tensor(x, dtype=torch.float64)
        mu = t.means[t.mode_assign(xt)].numpy()
        out['within_var'] = float(((x - mu) ** 2).sum(1).mean() / (t.dim * t.std ** 2))
    return out


def load():
    runs = defaultdict(dict)  # (tgt, meth) -> {seed: records}
    for p in sorted(glob.glob(os.path.join(BASE, '*', '*', 'seed*', 'log.jsonl'))):
        parts = p.split(os.sep)
        tgt, meth, seed = parts[-4], parts[-3], int(parts[-2][4:])
        if tgt not in FINAL:
            continue
        recs = [json.loads(l) for l in open(p) if l.strip()]
        if not recs or recs[-1]['iter'] != FINAL[tgt]:
            continue  # unfinished
        recs[-1].update(extra_stats(tgt, os.path.dirname(p)))
        runs[(tgt, meth)][seed] = recs
    return runs


def summarize(runs):
    s = {}
    for (t, m), d in runs.items():
        seeds = sorted(d)
        entry = {'seeds': seeds, 'n': len(seeds)}
        for k in METRICS[t] + ['wall_time', 'n_energy_evals'] + (['major_well'] if t == 'mw32' else []) + (['mode_coverage'] if t == 'gmm16' else []):
            v = np.array([d[sd][-1][k] for sd in seeds], dtype=float)
            entry[k] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else float('nan'),
                        'values': v.tolist()}
        s.setdefault(t, {})[m] = entry
    return s


def table(summary, targets, fname, header, colspec):
    lines = []
    for t in targets:
        if t not in summary:
            continue
        ms = TABLE_METRICS.get(t, METRICS[t])
        lines.append(f'\\multicolumn{{{len(ms) + 1}}}{{l}}{{\\textbf{{{TGT_NAME[t]}}}}}\\\\')
        best = {}
        for m in ms:
            if m == 'within_var':
                score = {me: abs(np.log(e[m]['mean'])) for me, e in summary[t].items()}
            else:
                score = {me: e[m]['mean'] for me, e in summary[t].items()}
            best[m] = min(score, key=score.get) if score else None
        for me in METH_NAME:
            if me not in summary[t]:
                continue
            e = summary[t][me]
            cells = []
            for m in ms:
                sd = e[m]['std']
                sds = '--' if not np.isfinite(sd) else ('0' if sd < 1e-9 else f'{sd:.2g}')
                c = f'{e[m]["mean"]:.3g}{{\\scriptsize$\\pm${sds}}}'
                cells.append(f'\\textbf{{{c}}}' if best[m] == me else c)
            lines.append(f'{METH_NAME[me]} & ' + ' & '.join(cells) + f' \\\\  % n={e["n"]}')
        lines.append('\\midrule')
    head = [f'\\begin{{tabular}}{{{colspec}}}', '\\toprule', header, '\\midrule']
    open(os.path.join(ROOT, 'paper', fname), 'w').write('\n'.join(head + lines[:-1] + ['\\bottomrule', '\\end{tabular}']) + '\n')


def curves(runs, targets):
    cols = {'idem': '#c44e52', 'aem_cold': '#8c8c8c', 'aem_ea': '#4c72b0', 'aem_ea_refine': '#55a868'}
    fig, axes = plt.subplots(1, len(targets), figsize=(3.3 * len(targets), 2.6))
    for ax, t in zip(np.atleast_1d(axes), targets):
        for me, c in cols.items():
            if (t, me) not in runs:
                continue
            d = runs[(t, me)]
            it = [r['iter'] for r in next(iter(d.values()))]
            v = np.array([[r['energy_w1'] for r in d[sd]] for sd in sorted(d)])
            ax.plot(it, v.mean(0), color=c, marker='o', ms=3, label=METH_NAME[me].replace('$\\to$', '->'))
            ax.fill_between(it, v.min(0), v.max(0), color=c, alpha=.15)
        ax.set_yscale('log')
        ax.set_title(TGT_NAME[t].replace('$', '').replace('{=}', '='), fontsize=9)
        ax.set_xlabel('outer iteration')
        ax.grid(alpha=.3)
        ax.axvline(FINAL[t] // 2, color='k', lw=.6, ls=':')
    np.atleast_1d(axes)[0].set_ylabel(r'energy $W_1$')
    np.atleast_1d(axes)[0].legend(fontsize=5.5)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'paper', 'fig_curves_gpu.pdf'))


if __name__ == '__main__':
    runs = load()
    summary = summarize(runs)
    if 'mw32' in summary:
        x = target('mw32').sample(20000, seed=0).numpy()
        summary['mw32_true_major_well'] = float((x[:, 0::2] >= 0).mean())
    json.dump(summary, open(os.path.join(BASE, 'summary.json'), 'w'), indent=1)
    table(summary, ['gmm16', 'mw32'], 'tab_gpu_free.tex', 'Method & $\\mathcal{W}_2$ & $\\mathcal{E}$-$\\mathcal{W}_1$ & MMD & mode TV & var.\\ ratio \\\\', 'lccccc')
    table(summary, ['dw4', 'lj13'], 'tab_gpu_mol.tex', 'Method & $\\mathcal{E}$-$\\mathcal{W}_1$ & $\\mathcal{E}$-$\\mathcal{W}_1^{\\mathrm{tr}}$ & $d$-$\\mathcal{W}_1$ & frac.\\ high-$\\mathcal{E}$ \\\\', 'lcccc')
    table(summary, ['oracle_dw4', 'oracle_lj13'], 'tab_gpu_oracle.tex', 'Method & $\\mathcal{E}$-$\\mathcal{W}_1$ & $\\mathcal{E}$-$\\mathcal{W}_1^{\\mathrm{tr}}$ & $d$-$\\mathcal{W}_1$ & frac.\\ high-$\\mathcal{E}$ \\\\', 'lcccc')
    curves(runs, [t for t in ['gmm16', 'mw32', 'dw4', 'lj13'] if any(k[0] == t for k in runs)])
    for t in summary:
        if isinstance(summary[t], dict) and 'seeds' not in summary[t]:
            for m, e in summary[t].items():
                print(t, m, 'n=', e['n'], {k: f'{e[k]["mean"]:.4g}+-{e[k]["std"]:.2g}' for k in METRICS[t]})
