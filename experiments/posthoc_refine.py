"""Post-hoc exact correction of final sampler output with pi-invariant subspace i-SIR sweeps (MW-32).
Usage: python experiments/posthoc_refine.py RUN_DIR [n_sweeps]"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aem.targets import get_target
from aem.explore import refine_buffer_explore
from aem.metrics import compute_all

run = sys.argv[1]; sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
t = get_target('mw32'); g = torch.Generator().manual_seed(0)
x = torch.tensor(np.load(os.path.join(run, 'samples.npy')), dtype=torch.float64)
ref = t.sample(2000, seed=12345).numpy()
def rep(z, tag, ne):
    m = compute_all(t, z.numpy(), ref); zn = z.numpy()
    print(tag, f'evals={ne:.3g}', {k: round(v, 3) for k, v in m.items() if k in ('w2', 'energy_w1', 'mmd', 'pattern_tv')},
          'DW+', round(float((zn[:, 0::2] > 0).mean()), 3), 'Gvar', round(float(zn[:, 1::2].var()), 3), flush=True)
rep(x, 'raw', 0); tot = 0
for s in range(1, sweeps + 1):
    x, r, ne = refine_buffer_explore(t, x, n_sweeps=1, K=4, sigma_min=0.3, sigma_max=3.0, b_max=1, p_full=0.0,
                                     n_blocks=32, generator=g)
    tot += ne
    if s in (1, 2, 5, 10, 20, 40): rep(x, f'sweep{s}', tot)
