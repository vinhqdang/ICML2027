"""Plot the estimator study: relative MSE and bias^2 against sigma for iDEM and AEM.

Usage: python experiments/plot_estimator.py results/estimator_gmm40.csv results/estimator_mw32.csv
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

R = 32  # repeats in the study; bias2 is debiased by var/(R-1)

STYLE = {
    ('idem', 10): ('#c44e52', '--'), ('idem', 100): ('#dd8452', '--'), ('idem', 500): ('#937860', '--'),
    ('aem', 0): ('#8c8c8c', '-'), ('aem', 10): ('#4c72b0', '-'), ('aem', 100): ('#55a868', '-'),
    ('aem', 500): ('#8172b3', '-'),
}


def label(est, K):
    return f"{'iDEM' if est == 'idem' else 'AEM'} K={K}"


def main(paths):
    for path in paths:
        df = pd.read_csv(path)
        df['bias2_c'] = (df['bias2'] - df['var'] / (R - 1)).clip(lower=0)
        norm = df['bias2'] / df['rel_bias2']
        df['rel_bias2_c'] = df['bias2_c'] / norm
        tgt = df['target'].iloc[0]
        anchors = list(dict.fromkeys(df['anchor_type']))
        fig, axes = plt.subplots(2, 2 * len(anchors), figsize=(3.2 * 2 * len(anchors), 5.4), sharex=True)
        for a_i, anc in enumerate(anchors):
            for k_i, kind in enumerate(['tsi', 'dsi']):
                col = 2 * a_i + k_i
                sub = df[(df.anchor_type == anc) & (df.kind == kind)]
                for (est, K), g in sub.groupby(['estimator', 'K']):
                    c, ls = STYLE[(est, K)]
                    g = g.sort_values('sigma')
                    axes[0, col].plot(g.sigma, g.rel_mse, color=c, ls=ls, label=label(est, K))
                    axes[1, col].plot(g.sigma, g.rel_bias2_c.clip(lower=1e-7), color=c, ls=ls)
                axes[1, col].set_ylim(bottom=5e-8)
                axes[0, col].set_title(f'{anc} anchor, {kind.upper()}', fontsize=9)
                for r in range(2):
                    axes[r, col].set_xscale('log'); axes[r, col].set_yscale('log')
                    axes[r, col].grid(alpha=.3)
                axes[1, col].set_xlabel(r'$\sigma_t$')
        axes[0, 0].set_ylabel('relative MSE'); axes[1, 0].set_ylabel(r'relative bias$^2$')
        axes[0, 0].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        out = f'paper/fig_estimator_{tgt}.pdf'
        fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=120)
        print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1:])
