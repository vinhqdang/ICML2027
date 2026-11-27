"""Target densities pi(x) ∝ exp(-E(x)) with exact samplers and, where available, exact noised scores.

Conventions
- energy(x) accepts [N, d] and returns [N]. Parameters are stored in float64 and cast to the input dtype.
- sample(n, seed=0, generator=None) returns exact samples [n, d].
- posterior_sample(x_t, sigma, generator=None) draws x0 ~ pi(x0 | x_t) exactly, for x_t = x0 + sigma * eps.
- noised_score(x_t, sigma) returns grad log p_t(x_t) for the VE-noised target, in closed form or by quadrature.
- tempered() / mode_dropped() return the corrupted targets used for anchor-corruption experiments.
- project(x) maps noise onto the sampling space. Identity for the free targets; the zero-centre-of-mass projector
  for the molecular targets (dw4, lj13). Samplers apply it to every Gaussian perturbation and score.
- Molecular targets (dw4, lj13) also have interatomic_distances(x) and sample() from a cached reference run
  (python -m aem.mcmc --target dw4). Their Gaussian densities live on d_eff = dim - spatial_dim dimensions.
"""
import math
import os

import torch

# (sigma_min, sigma_max, clip) per target. clip=None disables per-particle gradient clipping.
DEFAULTS = {
    'gmm40': (0.01, 70.0, 1e3),
    'mw32': (0.01, 3.0, None),
    'funnel10': (0.01, 10.0, 1e3),
    'gmm_d': (0.01, 20.0, None),
    'dw4': (0.01, 3.0, None),
    'lj13': (0.01, 2.0, 1e3),
}

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _ref_path(name):
    """Cache file of exact reference samples. The environment variable AEM_DATA_DIR overrides the folder."""
    return os.path.join(os.environ.get('AEM_DATA_DIR', _DATA_DIR), f'{name}_ref.pt')


class _Unconstrained:
    """Targets on the full R^dim: the projection onto the sampling space is the identity."""

    def project(self, x):
        return x


def effective_dim(target):
    """Dimension of the space the samplers work in. Gaussian densities and their normalisers use this."""
    return getattr(target, 'd_eff', target.dim)


def as_col(sigma, B):
    """Broadcast a scalar, [B] or [B, 1] noise level to [B, 1]."""
    s = torch.as_tensor(sigma)
    if s.ndim == 0:
        return s.reshape(1, 1).expand(B, 1)
    if s.ndim == 1:
        return s[:, None]
    return s.reshape(B, 1)


def _gen(seed=None, generator=None):
    """Explicit generator > new generator seeded with `seed` > None (global RNG)."""
    if generator is not None:
        return generator
    if seed is None:
        return None
    g = torch.Generator(device=torch.get_default_device())
    g.manual_seed(int(seed))
    return g


def _inv_cdf(logp, grid, u):
    """Inverse-CDF sampling on per-row grids with linear interpolation.

    logp, grid: [N, G] (grid ascending along the last axis); u: [N, M] uniforms in [0, 1).
    Returns [N, M].
    """
    p = torch.exp(logp - logp.amax(-1, keepdim=True))
    cdf = torch.cumsum(p, -1)
    cdf = cdf / cdf[..., -1:]
    G = grid.shape[-1]
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp(1, G - 1)
    c0, c1 = cdf.gather(-1, idx - 1), cdf.gather(-1, idx)
    g0, g1 = grid.gather(-1, idx - 1), grid.gather(-1, idx)
    frac = ((u - c0) / (c1 - c0).clamp_min(1e-300)).clamp(0.0, 1.0)
    return g0 + frac * (g1 - g0)


class GMM(_Unconstrained):
    """Equal-weight mixture of isotropic Gaussians N(mu_i, std^2 I).

    energy = -(logsumexp_i log N(x; mu_i, std^2 I) - log M), i.e. -log of the mixture density up to a constant.
    """

    plot_bounds = (-50.0, 50.0)

    def __init__(self, means, std=1.0):
        self.means = torch.as_tensor(means, dtype=torch.float64)
        self.std = float(std)
        self.n_comp, self.dim = self.means.shape

    def _sq_dist(self, x):
        """Squared distances [N, M] via |x|^2 - 2 x.mu + |mu|^2 (one matmul, no [N, M, d] tensor)."""
        mu = self.means.to(x.dtype)
        d2 = (x**2).sum(-1, keepdim=True) - 2.0 * x @ mu.T + (mu**2).sum(-1)[None, :]
        return d2.clamp_min(0.0)

    def energy(self, x):
        d2 = self._sq_dist(x)
        log_comp = -0.5 * d2 / self.std**2 - 0.5 * self.dim * math.log(2 * math.pi * self.std**2)
        return -(torch.logsumexp(log_comp, -1) - math.log(self.n_comp))

    def mode_assign(self, x):
        """Index of the nearest component mean, [N]."""
        return self._sq_dist(x).argmin(-1)

    def sample(self, n, seed=0, generator=None):
        g = _gen(seed, generator)
        idx = torch.randint(self.n_comp, (n,), generator=g)
        noise = torch.randn(n, self.dim, generator=g, dtype=torch.float64)
        return self.means[idx] + self.std * noise

    def noised_score(self, x_t, sigma):
        """Exact grad log p_t(x_t) = sum_i r_i(x_t) (mu_i - x_t) / v, with v = std^2 + sigma^2."""
        x_t = x_t.to(torch.float64)
        sig = as_col(sigma, x_t.shape[0]).to(torch.float64)
        v = self.std**2 + sig**2
        d2 = self._sq_dist(x_t)
        r = torch.softmax(-0.5 * d2 / v, -1)
        return (r @ self.means - x_t) / v

    def posterior_sample(self, x_t, sigma, generator=None):
        """Exact draw x0 ~ pi(x0 | x_t): pick a component by its posterior responsibility, then sample the Gaussian."""
        x_t = x_t.to(torch.float64)
        sig = as_col(sigma, x_t.shape[0]).to(torch.float64)
        g = _gen(None, generator)
        s2 = self.std**2
        tot = s2 + sig**2
        d2 = self._sq_dist(x_t)
        r = torch.softmax(-0.5 * d2 / tot, -1)
        comp = torch.multinomial(r, 1, generator=g)[:, 0]
        mu = self.means[comp]
        mean = mu + (s2 / tot) * (x_t - mu)
        var = s2 * sig**2 / tot
        return mean + var.sqrt() * torch.randn(x_t.shape, generator=g, dtype=torch.float64)

    def tempered(self):
        """Corrupted anchor target: same means, component std 2."""
        return GMM(self.means, std=2.0)

    def mode_dropped(self):
        """Corrupted anchor target: only the first half of the modes."""
        return GMM(self.means[: self.n_comp // 2], std=self.std)


class GMM40(GMM):
    """40 equal-weight unit-variance components, means uniform in [-40, 40]^2 (generator seed 0)."""

    plot_bounds = (-50.0, 50.0)

    def __init__(self):
        g = torch.Generator()
        g.manual_seed(0)
        means = (torch.rand(40, 2, generator=g, dtype=torch.float64, device='cpu') * 80.0 - 40.0).to(torch.get_default_device())
        super().__init__(means, std=1.0)


class GMMd(GMM):
    """n_comp equal-weight identity-covariance components, means uniform in [-scale, scale]^dim."""

    def __init__(self, dim=10, n_comp=10, scale=10.0, seed=0):
        g = torch.Generator()
        g.manual_seed(int(seed))
        means = ((torch.rand(n_comp, dim, generator=g, dtype=torch.float64, device='cpu') * 2.0 - 1.0) * scale).to(torch.get_default_device())
        super().__init__(means, std=1.0)
        self.plot_bounds = (-1.5 * scale, 1.5 * scale)


def _dw(x):
    return x**4 - 6.0 * x**2 - 0.5 * x


class ManyWell(_Unconstrained):
    """ManyWell target. Coordinates (2i, 2i+1): double well and Gaussian.

    energy = beta * sum_i [x_{2i}^4 - 6 x_{2i}^2 - 0.5 x_{2i} + 0.5 x_{2i+1}^2].
    beta = 1 is the target; beta = 1/2 is the temperature-2 tempered target exp(-E/2).
    """

    plot_bounds = (-3.0, 3.0)
    GRID_QUAD = 20001  # quadrature grid for the exact noised score
    GRID_POST = 512  # per-row grid for posterior sampling

    def __init__(self, dim=32, beta=1.0):
        assert dim % 2 == 0, 'ManyWell needs an even dimension'
        self.dim = dim
        self.beta = float(beta)
        self.half = dim // 2

    def energy(self, x):
        xe, xo = x[:, 0::2], x[:, 1::2]
        return self.beta * (_dw(xe) + 0.5 * xo**2).sum(-1)

    def mode_signs(self, x):
        """Sign of each double-well coordinate, [N, dim/2] with entries +-1."""
        return torch.where(x[:, 0::2] >= 0, 1, -1)

    def tempered(self):
        """Corrupted anchor target: temperature 2, density ∝ exp(-E/2)."""
        return ManyWell(self.dim, beta=self.beta / 2.0)

    def sample(self, n, seed=0, generator=None):
        g = _gen(seed, generator)
        grid = torch.linspace(-4.0, 4.0, 200001, dtype=torch.float64)[None]
        logp = -self.beta * _dw(grid)
        u = torch.rand(1, n * self.half, generator=g, dtype=torch.float64)
        xe = _inv_cdf(logp, grid, u).reshape(n, self.half)
        xo = torch.randn(n, self.half, generator=g, dtype=torch.float64) / math.sqrt(self.beta)
        out = torch.empty(n, self.dim, dtype=torch.float64)
        out[:, 0::2] = xe
        out[:, 1::2] = xo
        return out

    def posterior_sample(self, x_t, sigma, generator=None):
        """Exact draw from pi(x0 | x_t). Coordinates factorise: 1D grid inverse-CDF for double wells, Gaussian closed form."""
        x_t = x_t.to(torch.float64)
        B = x_t.shape[0]
        sig = as_col(sigma, B).to(torch.float64)
        g = _gen(None, generator)
        xe, xo = x_t[:, 0::2], x_t[:, 1::2]
        half_w = 8.0 * sig
        lo = (xe - half_w).clamp(min=-4.5, max=4.5 - 1e-6)
        hi = torch.maximum((xe + half_w).clamp(max=4.5), lo + 1e-6)
        frac = torch.linspace(0.0, 1.0, self.GRID_POST, dtype=torch.float64)
        grid = lo[..., None] + (hi - lo)[..., None] * frac
        logp = -self.beta * _dw(grid) - (grid - xe[..., None]) ** 2 / (2.0 * sig[..., None] ** 2)
        N = B * self.half
        u = torch.rand(N, 1, generator=g, dtype=torch.float64)
        xe_new = _inv_cdf(logp.reshape(N, -1), grid.reshape(N, -1), u).reshape(B, self.half)
        v = 1.0 / self.beta
        post_var = v * sig**2 / (v + sig**2)
        xo_new = xo * v / (v + sig**2) + post_var.sqrt() * torch.randn(B, self.half, generator=g, dtype=torch.float64)
        out = torch.empty(B, self.dim, dtype=torch.float64)
        out[:, 0::2] = xe_new
        out[:, 1::2] = xo_new
        return out

    def noised_score(self, x_t, sigma):
        """Exact grad log p_t(x_t) at beta = 1 (the target, not the corrupted one).

        Coordinates are independent, so grad log p_t,j = (E[x0_j | x_t,j] - x_t,j) / sigma^2.
        Double wells: posterior mean by quadrature on a fixed grid. Gaussians: -x / (1 + sigma^2).
        """
        x_t = x_t.to(torch.float64)
        B = x_t.shape[0]
        sig = as_col(sigma, B).to(torch.float64)
        xe, xo = x_t[:, 0::2], x_t[:, 1::2]
        grid = torch.linspace(-4.5, 4.5, self.GRID_QUAD, dtype=torch.float64)
        logp = -self.beta * _dw(grid)
        se = torch.empty_like(xe)
        for s in range(0, B, 32):
            e = slice(s, s + 32)
            L = logp - (xe[e][..., None] - grid) ** 2 / (2.0 * sig[e][..., None] ** 2)
            w = torch.softmax(L, -1)
            m = (w * grid).sum(-1)
            se[e] = (m - xe[e]) / sig[e] ** 2
        v = 1.0 / self.beta
        so = -xo / (v + sig**2)
        out = torch.empty_like(x_t)
        out[:, 0::2] = se
        out[:, 1::2] = so
        return out


class Funnel(_Unconstrained):
    """Funnel: x0 ~ N(0, 9), x_i | x0 ~ N(0, exp(x0)) for i = 1..dim-1. Energy drops constants."""

    plot_bounds = (-9.0, 9.0)

    def __init__(self, dim=10):
        self.dim = dim

    def energy(self, x):
        x0, rest = x[:, 0], x[:, 1:]
        return x0**2 / (2 * 9.0) + (rest**2).sum(-1) / (2 * torch.exp(x0)) + (self.dim - 1) / 2 * x0

    def sample(self, n, seed=0, generator=None):
        g = _gen(seed, generator)
        x0 = 3.0 * torch.randn(n, 1, generator=g, dtype=torch.float64)
        rest = torch.exp(0.5 * x0) * torch.randn(n, self.dim - 1, generator=g, dtype=torch.float64)
        return torch.cat([x0, rest], 1)


class _Molecular:
    """Molecules: n_particles points in spatial_dim dimensions, stored as one flat vector x of length dim.

    The energies are invariant to rigid translations, so the samplers work in the zero-centre-of-mass subspace of
    dimension d_eff = dim - spatial_dim. project(x) is the orthogonal projector onto that subspace.
    Subclasses set name, n_particles, spatial_dim, dim, and implement energy(x) (x: [N, dim] -> [N]).
    """

    name = None

    @property
    def d_eff(self):
        return self.dim - self.spatial_dim

    def _pos(self, x):
        return x.reshape(*x.shape[:-1], self.n_particles, self.spatial_dim)

    def project(self, x):
        """Remove the centre of mass: subtract the mean over particles, for any leading shape [..., dim]."""
        p = self._pos(x)
        return (p - p.mean(dim=-2, keepdim=True)).reshape(x.shape)

    def _pair_sqdist(self, x):
        """Squared distances for all pairs i < j, [..., n(n-1)/2]."""
        p = self._pos(x)
        i, j = torch.triu_indices(self.n_particles, self.n_particles, 1, device=x.device)
        return ((p[..., i, :] - p[..., j, :]) ** 2).sum(-1)

    def interatomic_distances(self, x):
        """Pairwise distances d_ij for i < j, [N, n(n-1)/2]."""
        return self._pair_sqdist(x).sqrt()

    def _reference(self):
        """Cached exact reference samples (float64, CPU). Raises a clear error when the cache is missing."""
        if getattr(self, '_ref', None) is None:
            path = _ref_path(self.name)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f'No reference samples for {self.name!r} at {path}. Generate them first with '
                    f'`python -m aem.mcmc --target {self.name}` (add --device cuda for the full run).')
            ref = torch.load(path, map_location='cpu', weights_only=True)['samples'].to(torch.float64)
            assert ref.shape[-1] == self.dim, f'cache {path} has dimension {ref.shape[-1]}, expected {self.dim}'
            self._ref = ref
        return self._ref

    def sample(self, n, seed=None, generator=None):
        """n reference samples drawn from the cache (without replacement while n <= cache size)."""
        ref = self._reference().to(torch.get_default_device())
        M = ref.shape[0]
        g = _gen(seed, generator)
        if n <= M:
            idx = torch.randperm(M, generator=g, device=ref.device)[:n]
        else:
            idx = torch.randint(M, (n,), generator=g, device=ref.device)
        return ref[idx].clone()


class DW4(_Molecular):
    """DW-4 (iDEM benchmark): 4 particles in 2D with pairwise double-well energy in the distance d_ij.

    E(x) = 1/(2 tau) sum_{i<j} [a (d_ij - d0) + b (d_ij - d0)^2 + c (d_ij - d0)^4], a = 0, b = -4, c = 0.9, d0 = 4.
    """

    name = 'dw4'

    def __init__(self, n_particles=4, spatial_dim=2, a=0.0, b=-4.0, c=0.9, d0=4.0, tau=1.0):
        self.n_particles, self.spatial_dim = n_particles, spatial_dim
        self.dim = n_particles * spatial_dim
        self.a, self.b, self.c, self.d0, self.tau = a, b, c, d0, tau

    def energy(self, x):
        d = self._pair_sqdist(x).clamp_min(1e-12).sqrt()  # the floor keeps the gradient finite at d = 0
        u = d - self.d0
        return (self.a * u + self.b * u**2 + self.c * u**4).sum(-1) / (2.0 * self.tau)


class LJ13(_Molecular):
    """LJ-13 (iDEM benchmark): 13 particles in 3D with Lennard-Jones pair energy and a harmonic centring term.

    E(x) = eps/(2 tau) sum_{i<j} [(rm/d_ij)^12 - 2 (rm/d_ij)^6] + c/2 sum_i ||x_i - mean(x)||^2.
    Distances are floored at min_dist (default 1e-2) so that the energy and its gradient stay finite.
    """

    name = 'lj13'

    def __init__(self, n_particles=13, spatial_dim=3, eps=1.0, rm=1.0, tau=1.0, c=0.5, min_dist=1e-2):
        self.n_particles, self.spatial_dim = n_particles, spatial_dim
        self.dim = n_particles * spatial_dim
        self.eps, self.rm, self.tau, self.c = eps, rm, tau, c
        self.min_dist = min_dist

    def energy(self, x):
        p = self._pos(x)
        r2 = self._pair_sqdist(x).clamp_min(self.min_dist**2)
        s6 = (self.rm**2 / r2) ** 3  # (rm / d)^6
        pair = (s6 * s6 - 2.0 * s6).sum(-1)  # (rm / d)^12 - 2 (rm / d)^6
        conf = ((p - p.mean(dim=-2, keepdim=True)) ** 2).sum((-1, -2))
        return self.eps / (2.0 * self.tau) * pair + 0.5 * self.c * conf


def get_target(name, **kw):
    """Build a target by name: 'gmm40', 'mw32', 'funnel10', 'gmm_d', 'dw4' or 'lj13'."""
    if name == 'gmm40':
        return GMM40(**kw)
    if name == 'mw32':
        return ManyWell(**{'dim': 32, **kw})
    if name == 'funnel10':
        return Funnel(**{'dim': 10, **kw})
    if name == 'gmm_d':
        return GMMd(**{'dim': 10, **kw})
    if name == 'dw4':
        return DW4(**kw)
    if name == 'lj13':
        return LJ13(**kw)
    raise ValueError(f'unknown target {name!r}')
