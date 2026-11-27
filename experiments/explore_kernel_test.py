"""Buffer-level test of the exploration kernels (no network): does an exact subspace i-SIR refinement with an
independent annealed-walker reference set repair a buffer with wrong mode weights / missing modes, and does it keep
an exact buffer exact?  Usage: python experiments/explore_kernel_test.py [mw|gmm] [seed]"""
import json
import sys
import time

import numpy as np
import torch

from aem.explore import annealed_walkers, refine_buffer_explore
from aem.targets import get_target, DEFAULTS

torch.set_num_threads(4)
which = sys.argv[1] if len(sys.argv) > 1 else 'mw'
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
g = torch.Generator(); g.manual_seed(seed)
out = []


def mw_stats(t, x):
    s = t.mode_signs(x)
    return {'major': float((s > 0).float().mean()), 'gvar': float(x[:, 1::2].var(0).mean())}


def gmm_stats(t, x):
    a = t.mode_assign(x).numpy()
    c = np.bincount(a, minlength=t.n_comp) / len(a)
    return {'modes': int((c >= 0.01 / t.n_comp).sum()), 'tv': 0.5 * float(np.abs(c - 1 / t.n_comp).sum())}


if which == 'mw':
    t = get_target('mw32'); stats = mw_stats
    smin, smax, _ = DEFAULTS['mw32']
    exact = t.sample(10000, seed=seed + 1).float()
    bad = exact.clone()  # biased buffer: move minor-well coordinates to the major well with prob 0.85
    xe = bad[:, 0::2]
    flip = (xe < 0) & (torch.rand(xe.shape, generator=g) < 0.85)
    xe[flip] = -xe[flip]
    bad[:, 0::2] = xe
    configs = {'full_bufref': dict(p_full=1.0), 'sub_bufref': dict(p_full=0.0, b_max=4),
               'sub_walk': dict(p_full=0.0, b_max=4, walk=True), 'sub_walk_exactstart': dict(p_full=0.0, b_max=4, walk=True, start='exact')}
else:
    t = get_target('gmm_d', dim=16); stats = gmm_stats
    smin, smax, _ = DEFAULTS['gmm_d']
    exact = t.sample(10000, seed=seed + 1).float()
    keep = t.mode_assign(exact) < 4  # buffer with only 4 of 10 modes
    bad = exact[keep][:10000]
    bad = bad[torch.randint(bad.shape[0], (10000,), generator=g)]
    configs = {'full_bufref': dict(p_full=1.0), 'full_walk': dict(p_full=1.0, walk=True),
               'mix_walk': dict(p_full=0.5, b_max=16, walk=True), 'mix_walk_exactstart': dict(p_full=0.5, b_max=16, walk=True, start='exact')}

walk, logw, n_walk = annealed_walkers(t, 2000, smax, n_levels=200, generator=g, return_logw=True)
print('walkers', stats(t, walk), 'evals', n_walk, flush=True)
for name, c in configs.items():
    buf = (exact if c.get('start') == 'exact' else bad).clone()
    rec = {'config': name, 'sweep': 0, **stats(t, buf)}
    print(rec, flush=True); out.append(rec)
    t0 = time.time()
    for sw in range(1, 21):
        buf, rate, ne = refine_buffer_explore(t, buf, walk if c.get('walk') else None, n_sweeps=1, K=32,
                                              tau_max=0.5 if which == 'mw' else 1.0, sigma_min=smin, sigma_max=smax,
                                              b_max=c.get('b_max'), p_full=c['p_full'], generator=g)
        if sw in (1, 2, 5, 10, 20):
            rec = {'config': name, 'sweep': sw, 'rate': rate, 'evals_per_sweep': ne, 'time': time.time() - t0,
                   **stats(t, buf)}
            print(rec, flush=True); out.append(rec)
json.dump({'walkers': stats(t, walk), 'walker_evals': n_walk, 'records': out},
          open(f'results/explore/kernel_test_{which}_s{seed}.json', 'w'), indent=1)
