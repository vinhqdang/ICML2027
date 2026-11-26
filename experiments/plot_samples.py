"""Marginal histograms on MW-32 (seed 0): a Gaussian coordinate and a double-well coordinate, iDEM vs AEM vs truth."""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aem.targets import get_target

t = get_target('mw32')
ref = t.sample(20000, seed=1).numpy()
runs = {'iDEM': 'idem', 'AEM (explore→anchor)': 'aem_warm'}
cols = {'iDEM': '#c44e52', 'AEM (explore→anchor)': '#4c72b0'}
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
g_bins = np.linspace(-6, 6, 61); dw_bins = np.linspace(-3, 3, 61)
axes[0].hist(ref[:, 1::2].ravel(), g_bins, density=True, color='k', alpha=.25, label='exact')
axes[1].hist(ref[:, 0::2].ravel(), dw_bins, density=True, color='k', alpha=.25, label='exact')
er = t.energy(torch.tensor(ref)).numpy()
e_bins = np.linspace(-190, 0, 60)
axes[2].hist(er, e_bins, density=True, color='k', alpha=.25, label='exact')
for name, m in runs.items():
    x = np.concatenate([np.load(os.path.join(ROOT, f'results/main/mw32/{m}/seed{s}/samples.npy')) for s in range(3)])
    axes[0].hist(x[:, 1::2].ravel(), g_bins, density=True, histtype='step', color=cols[name], lw=1.3, label=name)
    axes[1].hist(x[:, 0::2].ravel(), dw_bins, density=True, histtype='step', color=cols[name], lw=1.3, label=name)
    axes[2].hist(t.energy(torch.tensor(x)).numpy(), e_bins, density=True, histtype='step', color=cols[name], lw=1.3)
axes[0].set_title('Gaussian coordinates', fontsize=8); axes[1].set_title('double-well coordinates', fontsize=8)
axes[2].set_title('energy', fontsize=8)
for a in axes: a.tick_params(labelsize=7); a.set_yticks([])
axes[0].legend(fontsize=5.5, frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(ROOT, 'paper/fig_mw_marginals.pdf')); fig.savefig(os.path.join(ROOT, 'paper/fig_mw_marginals.png'), dpi=150)
