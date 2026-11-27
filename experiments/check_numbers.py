"""Independent consistency check: recompute the logged metrics straight from results/main_gpu/*/*/seed*/log.jsonl
(and samples.npy for within-mode variance and major-well fraction) and compare with the macros in paper/numbers.tex
that the text and tables use. Prints MISMATCH lines and a summary. Usage: python experiments/check_numbers.py"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = {'gmm16': 80, 'mw32': 80, 'dw4': 40, 'lj13': 30, 'oracle_dw4': 40, 'oracle_lj13': 30}
mac = {}
for l in open(os.path.join(ROOT, 'paper', 'numbers.tex')):
    m = re.match(r'\\expandafter\\def\\csname (.*?)\\endcsname\{(.*)\}', l)
    if m:
        mac[m.group(1)] = m.group(2)


def fmt(x, sig):
    if not np.isfinite(x):
        return '--'
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-3):
        mm, e = f'{x:.{sig - 1}e}'.split('e')
        return f'{mm}\\!\\cdot\\!10^{{{int(e)}}}'
    return f'{x:.{sig}g}'


runs = {}
for p in glob.glob(os.path.join(ROOT, 'results', 'main_gpu', '*', '*', 'seed*', 'log.jsonl')):
    t, m, s = p.split(os.sep)[-4:-1]
    if t not in FINAL:
        continue
    recs = [json.loads(x) for x in open(p) if x.strip()]
    if recs[-1]['iter'] != FINAL[t]:
        continue
    r = dict(recs[-1])
    xs = os.path.join(os.path.dirname(p), 'samples.npy')
    x = np.load(xs)
    x = x[np.isfinite(x).all(1)]
    if t == 'mw32':
        r['within_var'] = float(x[:, 1::2].var())
        r['major_well'] = float((x[:, 0::2] >= 0).mean())
    runs.setdefault((t, m), []).append(r)
bad = n = 0
for (t, m), rs in runs.items():
    for k in ('w2', 'energy_w1', 'mmd', 'mode_tv', 'pattern_tv', 'dist_w1', 'within_var', 'major_well', 'n_energy_evals', 'wall_time'):
        if k not in rs[0]:
            continue
        v = np.array([r[k] for r in rs], float)
        for suffix, val, sig in (('m', v.mean(), 3), ('s', v.std(ddof=1) if len(v) > 1 else float('nan'), 2)):
            key = f'v:{t}:{m}:{k}:{suffix}'
            n += 1
            if mac.get(key) != fmt(val, sig):
                bad += 1
                print('MISMATCH', key, mac.get(key), fmt(val, sig))
    if mac.get(f'n:{t}:{m}') != str(len(rs)):
        bad += 1
        print('MISMATCH count', t, m)
print(f'checked {n} macros against raw logs, {bad} mismatches, {len(runs)} (target, method) groups')
