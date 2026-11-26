"""Smoke tests. Run from the repository root: python -m aem.tests_smoke"""
import json
import os
import sys
import time

import torch

from .estimators import score_estimate
from .targets import get_target
from .train import train


def _rel_err(est, truth):
    """Mean over points of ||est - truth|| / ||truth||."""
    num = (est - truth).norm(dim=-1)
    den = truth.norm(dim=-1).clamp_min(1e-30)
    return float((num / den).mean())


def test_analytic_vs_anchored():
    """(1) GMM40: exact-anchor AEM with K = 20000 matches the analytic noised score."""
    target = get_target('gmm40')
    gen = torch.Generator()
    gen.manual_seed(1)
    n = 64
    errs = {}
    for sigma in (0.5, 5.0, 50.0):
        sig = torch.full((n,), sigma, dtype=torch.float64)
        x0 = target.sample(n, generator=gen)
        x_t = x0 + sigma * torch.randn(n, target.dim, generator=gen, dtype=torch.float64)
        anchor = target.posterior_sample(x_t, sig, generator=gen)
        est = score_estimate(x_t, sig, target, K=20000, anchor=anchor, kind='tsi', generator=gen)
        errs[sigma] = _rel_err(est, target.noised_score(x_t, sig))
        print(f'  sigma={sigma:6.2f}  rel. error vs analytic = {errs[sigma]:.4f}')
    mean_err = sum(errs.values()) / len(errs)
    print(f'  averaged over sigma: {mean_err:.4f} (threshold 0.1)')
    assert mean_err < 0.1, f'analytic vs anchored score error too large: {mean_err}'


def test_unbiasedness():
    """(2) GMM40, sigma = 30, K = 10: averaging over 400 repeats, AEM (exact anchors) beats iDEM."""
    target = get_target('gmm40')
    gen = torch.Generator()
    gen.manual_seed(2)
    n, R, K, sigma = 64, 400, 10, 30.0
    x0 = target.sample(n, generator=gen)
    x_t = x0 + sigma * torch.randn(n, target.dim, generator=gen, dtype=torch.float64)
    truth = target.noised_score(x_t, torch.full((n,), sigma, dtype=torch.float64))

    x_rep = x_t.repeat(R, 1)  # row r*n + i is repeat r of point i
    sig_rep = torch.full((R * n,), sigma, dtype=torch.float64)
    anchors = target.posterior_sample(x_rep, sig_rep, generator=gen)  # fresh exact posterior anchor per repeat

    aem = score_estimate(x_rep, sig_rep, target, K, anchor=anchors, kind='tsi', generator=gen)
    idem = score_estimate(x_rep, sig_rep, target, K, anchor=None, kind='tsi', generator=gen)
    aem_mean = aem.reshape(R, n, target.dim).mean(0)
    idem_mean = idem.reshape(R, n, target.dim).mean(0)
    err_aem = _rel_err(aem_mean, truth)
    err_idem = _rel_err(idem_mean, truth)
    print(f'  AEM  K={K}: relative error of 400-repeat mean = {err_aem:.4f}')
    print(f'  iDEM K={K}: relative error of 400-repeat mean = {err_idem:.4f}')
    print(f'  ratio AEM / iDEM = {err_aem / err_idem:.3f}')
    assert err_aem < 0.5 * err_idem, 'AEM should be much closer to the analytic score than iDEM'


def test_train():
    """(3) Tiny training runs for both methods finish and write log.jsonl."""
    base = os.path.join('results', 'smoke')
    for method in ('idem', 'aem'):
        out = os.path.join(base, f'train_{method}')
        res = train(target='gmm40', method=method, K=16, n_outer=2, n_inner=20, sde_steps=50,
                    n_eval=200, n_new=500, batch_size=128, out_dir=out, seed=0)
        log = res['log']
        assert os.path.isfile(log), 'log.jsonl missing'
        with open(log) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        assert len(recs) >= 1, 'no log records'
        last = recs[-1]
        print(f'  {method}: {len(recs)} log record(s), last iter={last["iter"]}, '
              f'energy evals={last["n_energy_evals"]}, w2={last["w2"]:.3f}, mmd={last["mmd"]:.4f}')
        assert os.path.isfile(os.path.join(out, 'model.pt'))


def main():
    tests = [
        ('(1) analytic vs anchored score, GMM40', test_analytic_vs_anchored),
        ('(2) unbiasedness, GMM40 sigma=30', test_unbiasedness),
        ('(3) train smoke, gmm40 idem and aem', test_train),
    ]
    failed = []
    for name, fn in tests:
        print(name)
        t0 = time.time()
        try:
            fn()
            print(f'  PASS ({time.time() - t0:.1f} s)')
        except AssertionError as e:
            failed.append(name)
            print(f'  FAIL ({time.time() - t0:.1f} s): {e}')
    if failed:
        print('FAILED:', failed)
        sys.exit(1)
    print('all smoke tests passed')


if __name__ == '__main__':
    main()
