"""Sample-quality metrics comparing generated samples with exact reference samples."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance


def _np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _subsample(a, m, rng):
    if len(a) > m:
        return a[rng.choice(len(a), m, replace=False)]
    return a[:m]


def w2(x, y, max_points=1000, seed=0):
    """Exact 2-Wasserstein between two empirical sets of equal size (subsampled to <= max_points)."""
    x, y = _np(x), _np(y)
    rng = np.random.default_rng(seed)
    m = min(len(x), len(y), max_points)
    x, y = _subsample(x, m, rng), _subsample(y, m, rng)
    cost = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def energy_w1(ex, ey):
    """1D W1 between two sets of energy values, after dropping non-finite and clipping to a sane range."""
    ex, ey = _np(ex).ravel(), _np(ey).ravel()
    ex, ey = ex[np.isfinite(ex)], ey[np.isfinite(ey)]
    ex, ey = np.clip(ex, -1e8, 1e8), np.clip(ey, -1e8, 1e8)
    return float(wasserstein_distance(ex, ey))


def mmd_rbf(x, y, max_points=2000, seed=0):
    """Unbiased MMD^2 with a Gaussian kernel exp(-|a-b|^2 / h), h = median squared distance of the pooled sample."""
    x, y = _np(x), _np(y)
    rng = np.random.default_rng(seed)
    x, y = _subsample(x, max_points, rng), _subsample(y, max_points, rng)
    z = np.concatenate([x, y])
    sq = (z**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * z @ z.T, 0.0)
    n, m = len(x), len(y)
    iu = np.triu_indices(n + m, 1)
    h = np.median(d2[iu])
    h = h if h > 0 else 1.0
    K = np.exp(-d2 / h)
    kxx, kyy, kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    off = lambda M, k: (M.sum() - np.trace(M)) / (k * (k - 1))
    return float(off(kxx, n) + off(kyy, m) - 2.0 * kxy.mean())


def _mode_metrics(assign, n_modes):
    n = len(assign)
    counts = np.bincount(assign, minlength=n_modes).astype(np.float64)
    expected_share = n / n_modes
    coverage = float(np.mean(counts >= 0.01 * expected_share))
    p = counts / max(n, 1)
    tv = 0.5 * float(np.abs(p - 1.0 / n_modes).sum())
    return coverage, tv


def _pattern_codes(signs):
    """Integer code of the first 4 double-well sign bits, in [0, 16)."""
    bits = (signs[:, :4] > 0).astype(np.int64)
    return bits @ np.array([1, 2, 4, 8])


def compute_all(target, samples, ref):
    """Metrics for `samples` against exact reference samples `ref`. Keys depend on the target.

    Note: w2 is computed on raw coordinates. For molecular targets it is not invariant to rotations or particle
    permutations; it is kept unchanged for comparability with the other targets. dist_w1 is the invariant check.
    """
    xs = torch.as_tensor(_np(samples))
    xr = torch.as_tensor(_np(ref))
    with torch.no_grad():
        es = target.energy(xs).cpu().numpy()
        er = target.energy(xr).cpu().numpy()
    out = {
        'w2': w2(xs, xr),
        'energy_w1': energy_w1(es, er),
        'mmd': mmd_rbf(xs, xr),
    }
    if hasattr(target, 'mode_assign'):
        with torch.no_grad():
            a = target.mode_assign(xs).cpu().numpy()
        cov, tv = _mode_metrics(a, target.n_comp)
        out['mode_coverage'] = cov
        out['mode_tv'] = tv
    if hasattr(target, 'mode_signs'):
        with torch.no_grad():
            cs = _pattern_codes(target.mode_signs(xs).cpu().numpy())
            cr = _pattern_codes(target.mode_signs(xr).cpu().numpy())
        ps = np.bincount(cs, minlength=16) / len(cs)
        pr = np.bincount(cr, minlength=16) / len(cr)
        out['pattern_tv'] = 0.5 * float(np.abs(ps - pr).sum())
    if hasattr(target, 'interatomic_distances'):  # molecular: 1D W1 between pooled pairwise-distance samples
        with torch.no_grad():
            ds = target.interatomic_distances(xs).cpu().numpy()
            dr = target.interatomic_distances(xr).cpu().numpy()
        out['dist_w1'] = float(wasserstein_distance(ds.ravel(), dr.ravel()))
    return out
