"""Write paper/tab_ztest.tex and paper/tab_oracle3.tex from results/dsi_bias/dsi_bias.csv,
results/oracle/oracle_dsm.csv and results/oracle/{mw32,gmm16}_score_error_3seeds_summary.txt."""
import os
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(R, 'paper')

d = pd.read_csv(os.path.join(R, 'results/dsi_bias/dsi_bias.csv'))
d['K'] = d['K'].fillna(0).astype(int)
z = d.assign(az=d.zscore.abs()).groupby(['target', 'sigma', 'estimator', 'K']).az.max()
cols = [('dw4', 'idem', 100), ('dw4', 'idem', 1000), ('dw4', 'aem', 100), ('lj13', 'idem', 100), ('lj13', 'aem', 100)]
sig = {'dw4': [0.03, 0.1, 0.3, 1, 2, 3], 'lj13': [0.01, 0.03, 0.1, 0.3, 1, 2]}
lines = []
for i in range(6):
    row = []
    for tg, est, K in cols:
        s = sig[tg][i]
        v = z.get((tg, float(s), est, K), np.nan)
        cell = '--' if np.isnan(v) else (f'\\textbf{{{v:.1f}}}' if v > 3 else f'{v:.1f}')
        row.append(cell)
    lines.append(f"{sig['dw4'][i]:g} & " + ' & '.join(row[:3]) + f" & {sig['lj13'][i]:g} & " + ' & '.join(row[3:]) + r'\\')
open(os.path.join(P, 'tab_ztest.tex'), 'w').write('\n'.join(lines) + '\n')

o = pd.read_csv(os.path.join(R, 'results/oracle/oracle_dsm.csv'))
pv = o.pivot_table(index=['target', 'seed', 'sigma'], columns='method', values='Ltilde').reset_index()
pv['dL'] = pv['idem'] - pv['aem']
g = pv.groupby(['target', 'sigma']).dL.agg(['mean', 'std', lambda x: int((x > 0).sum())]).reset_index()
g.columns = ['target', 'sigma', 'mean', 'std', 'npos']
def fmt(m, s):
    e = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
    if -2 <= e <= 1:
        dig = max(0, 1 - int(np.floor(np.log10(abs(s))))) if s > 0 else 2
        dig = min(dig, 4)
        return f'{m:.{dig}f}\\pm{s:.{dig}f}'
    return f'({m/10**e:.1f}\\pm{s/10**e:.1f})\\!\\cdot\\!10^{{{e}}}'
def sci(m, s):
    if m == 0: return '0'
    e = int(np.floor(np.log10(abs(m))))
    if e >= -1: return f'{m:.2f}\\pm{s:.2f}' if e < 1 else f'{m:.1f}\\pm{s:.1f}'
    return f'({m/10**e:.2f}\\pm{s/10**e:.2f})\\cdot10^{{{e}}}'
dl = {(r.target, r.sigma): (r['mean'], r['std'], r['npos']) for _, r in g.iterrows()}
zl = open(os.path.join(P, 'tab_ztest.tex')).read().strip().split('\n')
out = []
for i in range(6):
    cells = zl[i].rstrip('\\').split(' & ')
    a = dl[('dw4', sig['dw4'][i])]; b = dl[('lj13', sig['lj13'][i])]
    out.append(' & '.join(cells[:4] + [f'${sci(a[0], a[1])}$ ({a[2]}/3)'] + cells[4:] + [f'${sci(b[0], b[1])}$ ({b[2]}/3)']) + r'\\')
open(os.path.join(P, 'tab_diag.tex'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))

# Appendix table: max |z| per estimator type (results/dsi_bias/kinds)
k = pd.read_csv(os.path.join(R, 'results/dsi_bias/kinds/dsi_bias_kinds.csv'))
k['az'] = k.zscore.abs()
mk = k.groupby(['target', 'sigma', 'estimator']).az.max().unstack('estimator')
order = ['idem_tsi', 'idem_dsi', 'idem_cv', 'aem_tsi', 'aem_dsi', 'aem_cv']
names = {'dw4': 'DW-4', 'lj13': 'LJ-13', 'gmm16': 'GMM-10'}
out = []
for tg in ['dw4', 'lj13', 'gmm16']:
    sub = mk.loc[tg]
    for i, (s, row) in enumerate(sub.iterrows()):
        cells = []
        for c in order:
            v = row[c]
            cells.append('--' if np.isnan(v) else (f'\\textbf{{{v:.1f}}}' if v > 3 else f'{v:.1f}'))
        out.append(f"{names[tg] if i == 0 else ''} & {s:g} & " + ' & '.join(cells) + r'\\')
    out.append(r'\midrule' if tg != 'gmm16' else '')
open(os.path.join(P, 'tab_kinds.tex'), 'w').write('\n'.join(out) + '\n')
