"""Aggregate the exploration runs. Usage: python experiments/explore_analyze.py RUNS_DIR
RUNS_DIR/<target>_<method>_s<seed>/{log.jsonl,samples.npy}. Metrics: the final log record (aem/metrics.compute_all
on the 2000 evaluation samples) plus, for MW-32, the major-well fraction (share of double-well coordinates > 0;
exact value 0.842) and the mean variance of the Gaussian coordinates (exact 1), computed from samples.npy.
Writes results/explore/runs.csv (one row per run) and results/explore/summary.csv (mean, std over seeds)."""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

root = sys.argv[1]
rows = []
for name in sorted(os.listdir(root)):
    m = re.match(r'(gmm16|mw32)_(.+)_s(\d+)$', name)
    lp = os.path.join(root, name, 'log.jsonl')
    if not m or not os.path.exists(lp):
        continue
    recs = [json.loads(l) for l in open(lp) if l.strip()]
    if not recs or recs[-1]['iter'] != 60:
        continue
    r = {'target': m.group(1), 'method': m.group(2), 'seed': int(m.group(3)), **recs[-1]}
    sp = os.path.join(root, name, 'samples.npy')
    if m.group(1) == 'mw32' and os.path.exists(sp):
        x = np.load(sp)
        r['major_frac'] = float((x[:, 0::2] >= 0).mean())
        r['gauss_var'] = float(x[:, 1::2].var(0).mean())
    rows.append(r)

keys = ['target', 'method', 'seed', 'n_energy_evals', 'wall_time', 'w2', 'energy_w1', 'mmd', 'mode_coverage', 'mode_tv',
        'pattern_tv', 'major_frac', 'gauss_var']
os.makedirs('results/explore', exist_ok=True)
with open('results/explore/runs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, keys, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
groups = defaultdict(list)
for r in rows:
    groups[(r['target'], r['method'])].append(r)
mkeys = keys[3:]
with open('results/explore/summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['target', 'method', 'n_seeds'] + [f'{k}_{s}' for k in mkeys for s in ('mean', 'std')])
    for (t, meth), rs in sorted(groups.items()):
        line = [t, meth, len(rs)]
        out = []
        for k in mkeys:
            v = np.array([r[k] for r in rs if k in r and r[k] is not None], dtype=float)
            line += [f'{v.mean():.4g}', f'{v.std():.2g}'] if len(v) else ['', '']
            if len(v):
                out.append(f'{k}={v.mean():.4g}±{v.std():.2g}')
        w.writerow(line)
        print(t, meth, f'n={len(rs)}', ' '.join(out))
