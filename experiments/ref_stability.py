"""Compare the first and second halves (in time) of a cached MALA reference run: energy histograms (W1, mean, std)
and pooled distance statistics. Usage: python -m experiments.ref_stability <cache.pt> <target> [out.json]"""
import json
import sys

import torch
from scipy.stats import wasserstein_distance

from aem.targets import get_target

path, name = sys.argv[1], sys.argv[2]
b = torch.load(path)
C, T = b['n_chains'], b['steps'] // b['thin']
x = b['samples'].view(C, T, -1).double()
tg = get_target(name)
res = {'acceptance': b['acceptance'], 'n_chains': C, 'kept_per_chain': T}
halves = {}
for k, part in (('first', x[:, :T // 2]), ('second', x[:, T // 2:])):
    p = part.reshape(-1, x.shape[-1])
    e = tg.energy(p).cpu().numpy()
    d = tg.interatomic_distances(p).cpu().numpy().ravel()
    halves[k] = (e, d)
    res[k] = {'E_mean': float(e.mean()), 'E_std': float(e.std()), 'E_q05': float(torch.tensor(e).quantile(0.05)),
              'E_q95': float(torch.tensor(e).quantile(0.95)), 'dist_mean': float(d.mean()), 'dist_std': float(d.std())}
res['energy_w1_halves'] = float(wasserstein_distance(halves['first'][0], halves['second'][0]))
res['dist_w1_halves'] = float(wasserstein_distance(halves['first'][1], halves['second'][1]))
# chain-split baseline: first vs second half of chains within the second time half (sampling noise level)
n = x.shape[0] // 2
ea = tg.energy(x[:n, T // 2:].reshape(-1, x.shape[-1])).cpu().numpy()
eb = tg.energy(x[n:, T // 2:].reshape(-1, x.shape[-1])).cpu().numpy()
res['energy_w1_chain_halves'] = float(wasserstein_distance(ea, eb))
print(json.dumps(res, indent=1))
if len(sys.argv) > 3:
    json.dump(res, open(sys.argv[3], 'w'), indent=1)
