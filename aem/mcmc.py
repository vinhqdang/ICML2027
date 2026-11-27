"""Reference samples for the molecular targets (dw4, lj13) from long parallel MALA runs.

Many chains run in parallel on the zero-centre-of-mass subspace. Each step is a Metropolis-adjusted Langevin proposal
    y = P(x - h/2 grad E(x) + sqrt(h) xi),   P = target.project (orthogonal projector), xi ~ N(0, I),
with the exact Metropolis-Hastings correction on that subspace. Step sizes h are per chain and are adapted towards
60% acceptance during burn-in (Robbins-Monro on log h), then frozen. After burn-in, every `thin`-th state of each
chain is kept and the states are cached to aem/data/<name>_ref.pt (float32, shape [N, dim]).

Run: python -m aem.mcmc --target dw4 --chains 2000 --burnin 20000 --steps 10000 --thin 50 --device cuda
"""
import argparse
import math
import os
import time

import torch

from .targets import _ref_path, get_target

TARGET_ACCEPT = 0.6


def _logp_grad(target, x):
    """Energies [C] and gradients [C, dim] of the energy at x (no graph kept)."""
    with torch.enable_grad():
        xr = x.detach().clone().requires_grad_(True)
        E = target.energy(xr)
        (g,) = torch.autograd.grad(E.sum(), xr)
    return E.detach(), g.detach()


def _mala_step(target, x, E, g, h, gen):
    """One MALA step for all chains. h: [C, 1] per-chain step sizes. Returns new (x, E, g) and the accept flags [C]."""
    P = target.project
    xi = torch.randn(x.shape, generator=gen, dtype=x.dtype)
    mu_x = x - 0.5 * h * P(g)
    y = P(mu_x + torch.sqrt(h) * xi)  # the noise is already in the subspace, P only guards round-off
    Ey, gy = _logp_grad(target, y)
    mu_y = y - 0.5 * h * P(gy)
    log_q_fwd = -((y - mu_x) ** 2).sum(-1) / (2.0 * h[:, 0])
    log_q_bwd = -((x - mu_y) ** 2).sum(-1) / (2.0 * h[:, 0])
    log_alpha = (E - Ey) + log_q_bwd - log_q_fwd
    ok = torch.isfinite(Ey) & torch.isfinite(gy).all(-1) & torch.isfinite(log_alpha)
    log_alpha = torch.where(ok, log_alpha, torch.full_like(log_alpha, -float('inf')))
    u = torch.rand(x.shape[0], generator=gen, dtype=x.dtype)
    acc = torch.log(u) < log_alpha
    x = torch.where(acc[:, None], y, x)
    E = torch.where(acc, Ey, E)
    g = torch.where(acc[:, None], gy, g)
    return x, E, g, acc


def initial_states(target, n_chains, gen):
    """Starting points on the zero-centre-of-mass subspace.

    dw4: standard normal times 2, centred. lj13: uniform in a ball of radius 1.5 per particle, centred, then gradient
    descent with a clipped step to remove overlaps.
    """
    dim, n, s = target.dim, target.n_particles, target.spatial_dim
    if target.name == 'dw4':
        return target.project(2.0 * torch.randn(n_chains, dim, generator=gen, dtype=torch.float64))
    if target.name == 'lj13':
        direc = torch.randn(n_chains, n, s, generator=gen, dtype=torch.float64)
        direc = direc / direc.norm(dim=-1, keepdim=True)
        radius = 1.5 * torch.rand(n_chains, n, 1, generator=gen, dtype=torch.float64) ** (1.0 / s)
        x = target.project((direc * radius).reshape(n_chains, dim))
        for _ in range(500):
            E, g = _logp_grad(target, x)
            g = target.project(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
            nrm = g.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            x = target.project(x - 0.01 * g * torch.clamp(1.0 / nrm, max=1.0))  # step length <= 0.01
        return x
    raise ValueError(f'no initialisation rule for {target.name!r}')


def run_chains(target, n_chains=2000, burnin=20000, steps=10000, thin=50, seed=0, target_accept=TARGET_ACCEPT,
               h0=1e-2, x0=None, verbose=True, log_every=2000):
    """Run MALA chains. Returns (samples [n_chains * steps // thin, dim] float32 on CPU, info dict).

    burnin steps adapt the per-chain step sizes; steps post-burn-in steps are run with frozen step sizes and every
    thin-th state is kept. info holds the acceptance rates and the step sizes.
    """
    t0 = time.time()
    gen = torch.Generator(device=torch.get_default_device())
    gen.manual_seed(int(seed))
    x = target.project(x0.to(torch.float64) if x0 is not None else initial_states(target, n_chains, gen))
    C = x.shape[0]
    E, g = _logp_grad(target, x)
    log_h = torch.full((C, 1), math.log(h0), dtype=torch.float64, device=x.device)

    for t in range(burnin):
        x, E, g, acc = _mala_step(target, x, E, g, log_h.exp(), gen)
        gain = 0.5 / math.sqrt(t + 10.0)  # decaying Robbins-Monro gain
        log_h = log_h + gain * (acc.to(x.dtype)[:, None] - target_accept)
        if verbose and log_every and (t + 1) % log_every == 0:
            print(f'  burn-in {t + 1}/{burnin}: acc={float(acc.double().mean()):.3f} '
                  f'median h={float(log_h.exp().median()):.2e}  ({time.time() - t0:.0f} s)', flush=True)

    h = log_h.exp()
    kept, acc_sum = [], 0.0
    for t in range(1, steps + 1):
        x, E, g, acc = _mala_step(target, x, E, g, h, gen)
        acc_sum += float(acc.double().mean())
        if t % thin == 0:
            kept.append(x.to(torch.float32).cpu().clone())
        if verbose and log_every and t % log_every == 0:
            print(f'  sampling {t}/{steps}: running acc={acc_sum / t:.3f}  ({time.time() - t0:.0f} s)', flush=True)

    samples = torch.stack(kept, dim=1).reshape(-1, x.shape[1])  # chain-major within each kept time
    info = {
        'acceptance': acc_sum / max(steps, 1),
        'step_size_median': float(h.median()),
        'step_size_min': float(h.min()),
        'step_size_max': float(h.max()),
        'n_chains': C,
        'seconds': time.time() - t0,
    }
    return samples, info


def build_reference(name, n_chains=2000, burnin=20000, steps=10000, thin=50, seed=0, device=None, out_path=None,
                    verbose=True):
    """Run the chains for target `name` and write the cache. Returns the path written."""
    if device is not None:
        torch.set_default_device(device)
    target = get_target(name)
    samples, info = run_chains(target, n_chains=n_chains, burnin=burnin, steps=steps, thin=thin, seed=seed,
                               verbose=verbose)
    mean_dist = float(target.interatomic_distances(samples.to(torch.float64)).mean())
    mean_energy = float(target.energy(samples.to(torch.float64)).mean())
    path = out_path or _ref_path(name)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    blob = {
        'samples': samples.cpu().contiguous(),
        'name': name,
        'n_chains': int(n_chains),
        'burnin': int(burnin),
        'steps': int(steps),
        'thin': int(thin),
        'seed': int(seed),
        'acceptance': float(info['acceptance']),
        'mean_pair_distance': mean_dist,
        'mean_energy': mean_energy,
    }
    torch.save(blob, path)
    if verbose:
        print(f'wrote {samples.shape[0]} samples to {path}')
        print(f'  acceptance (sampling phase) = {info["acceptance"]:.3f}, step size median = '
              f'{info["step_size_median"]:.2e}, mean pair distance = {mean_dist:.3f}, '
              f'mean energy = {mean_energy:.3f}, time = {info["seconds"]:.0f} s')
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate cached reference samples for a molecular target by MALA.')
    p.add_argument('--target', required=True, choices=['dw4', 'lj13'])
    p.add_argument('--chains', type=int, default=2000)
    p.add_argument('--burnin', type=int, default=20000)
    p.add_argument('--steps', type=int, default=10000, help='steps after burn-in')
    p.add_argument('--thin', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=None, help='output file (default aem/data/<target>_ref.pt)')
    a = p.parse_args(argv)
    build_reference(a.target, n_chains=a.chains, burnin=a.burnin, steps=a.steps, thin=a.thin, seed=a.seed,
                    device=a.device, out_path=a.out)


if __name__ == '__main__':
    main()
