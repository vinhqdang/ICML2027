"""Tests for the molecular targets (dw4, lj13), the zero-centre-of-mass projection, the EGNN score network,
the MALA reference sampler and training with arch=egnn. CPU, small sizes.

Run from the repository root: python -m aem.tests_molecular
The reference-sample tests write to a temporary folder (AEM_DATA_DIR), so no real cache is read or overwritten.
"""
import json
import os
import shutil
import sys
import tempfile
import time

import torch

from .model import EGNNScore
from .mcmc import build_reference, initial_states, run_chains
from .targets import get_target
from .train import train


def _rel_diff(a, b):
    """Largest |a - b| relative to the scale of the values (energies may be large for overlapping LJ configs)."""
    return float(((a - b).abs() / (1.0 + b.abs())).max())


def _random_rotation(s, gen):
    q, r = torch.linalg.qr(torch.randn(s, s, generator=gen, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))  # Haar-distributed orthogonal matrix


def _configs(target, n, gen):
    """Random configurations: normal scatter for dw4; lj13 mixes near-minimum and scattered configurations."""
    if target.name == 'lj13':
        near = initial_states(target, n // 2, gen)
        scat = target.project(1.5 * torch.randn(n - n // 2, target.dim, generator=gen, dtype=torch.float64))
        return torch.cat([near, scat])
    return target.project(2.0 * torch.randn(n, target.dim, generator=gen, dtype=torch.float64))


def test_energy_invariances():
    """(a) energies are finite on random inputs and invariant to translation, rotation and permutation."""
    gen = torch.Generator()
    gen.manual_seed(0)
    for name in ('dw4', 'lj13'):
        t = get_target(name)
        x = _configs(t, 200, gen)
        E = t.energy(x)
        assert torch.isfinite(E).all(), f'{name}: non-finite energy'
        xf = torch.randn(200, t.dim, generator=gen, dtype=torch.float32) * 1.5
        assert torch.isfinite(t.energy(xf)).all(), f'{name}: non-finite float32 energy'
        p = x.reshape(-1, t.n_particles, t.spatial_dim)
        shift = torch.randn(t.spatial_dim, generator=gen, dtype=torch.float64) * 3.0
        Et = t.energy((p + shift).reshape(x.shape))
        Q = _random_rotation(t.spatial_dim, gen)
        Er = t.energy((p @ Q).reshape(x.shape))
        perm = torch.randperm(t.n_particles, generator=gen)
        Ep = t.energy(p[:, perm].reshape(x.shape))
        dt, dr, dp = _rel_diff(Et, E), _rel_diff(Er, E), _rel_diff(Ep, E)
        print(f'  {name}: finite; max rel. change under translation {dt:.2e}, rotation {dr:.2e}, '
              f'permutation {dp:.2e}')
        assert max(dt, dr, dp) < 1e-9, f'{name}: energy not invariant'


def test_projection():
    """(b) project() gives zero centre of mass, is idempotent, and the free targets use the identity."""
    gen = torch.Generator()
    gen.manual_seed(1)
    for name in ('dw4', 'lj13'):
        t = get_target(name)
        x = torch.randn(64, t.dim, generator=gen, dtype=torch.float64)
        px = t.project(x)
        com = px.reshape(-1, t.n_particles, t.spatial_dim).mean(1).abs().max()
        assert float(com) < 1e-12, f'{name}: centre of mass not removed ({float(com)})'
        assert _rel_diff(t.project(px), px) < 1e-12, f'{name}: projection not idempotent'
        assert t.interatomic_distances(x).shape == (64, t.n_particles * (t.n_particles - 1) // 2)
        print(f'  {name}: max |COM| after projection {float(com):.1e}')
    for name in ('gmm40', 'mw32', 'funnel10', 'gmm_d'):
        t = get_target(name)
        x = torch.randn(4, t.dim, generator=gen, dtype=torch.float64)
        assert t.project(x) is x, f'{name}: project must be the identity'
    print('  free targets: project is the identity object')


def test_egnn_symmetry():
    """(c) EGNNScore output is translation invariant, rotation equivariant and has zero centre of mass."""
    gen = torch.Generator()
    gen.manual_seed(2)
    for n, s in ((4, 2), (13, 3)):
        net = EGNNScore(n, s, hidden=32, n_layers=2, time_emb=16, sigma_min=0.01, sigma_max=3.0).double()
        net.eval()
        x = torch.randn(6, n * s, generator=gen, dtype=torch.float64) * 1.3
        t = torch.rand(6, generator=gen, dtype=torch.float64)
        out = net(x, t).detach()
        p_out = out.reshape(6, n, s)
        assert float(p_out.mean(1).abs().max()) < 1e-12, 'output has nonzero centre of mass'
        shift = torch.randn(s, generator=gen, dtype=torch.float64).repeat(n)
        d_tr = float((net(x + shift, t) - out).abs().max())
        Q = _random_rotation(s, gen)
        x_rot = (x.reshape(6, n, s) @ Q).reshape(6, n * s)
        d_rot = float((net(x_rot, t) - (p_out @ Q).reshape(6, n * s)).abs().max())
        print(f'  n={n}, s={s}: translation error {d_tr:.2e}, rotation-equivariance error {d_rot:.2e}')
        assert d_tr < 1e-10, 'EGNN output is not translation invariant'
        assert d_rot < 1e-10, 'EGNN output is not rotation equivariant'
    net = EGNNScore(4, 2, hidden=16, n_layers=1, time_emb=8).double()
    assert net(torch.randn(3, 8, dtype=torch.float64), torch.rand(3, dtype=torch.float64)).shape == (3, 8)


def test_mcmc_dw4():
    """(d) MALA on DW4: 200 chains, 2000 steps. Acceptance in (0.3, 0.9); mean pairwise distance near 4."""
    t = get_target('dw4')
    samples, info = run_chains(t, n_chains=200, burnin=2000, steps=2000, thin=50, seed=3, verbose=False)
    acc = info['acceptance']
    mean_d = float(t.interatomic_distances(samples.double()).mean())
    com = float(samples.double().reshape(-1, 4, 2).mean(1).abs().max())
    print(f'  acceptance {acc:.3f}, mean pairwise distance {mean_d:.3f} (target ~4), max |COM| {com:.1e}, '
          f'time {info["seconds"]:.1f} s')
    assert 0.3 < acc < 0.9, f'acceptance {acc} outside (0.3, 0.9)'
    assert abs(mean_d - 4.0) < 0.5, f'mean pairwise distance {mean_d} not near 4'
    assert com < 1e-5, 'samples not on the zero-centre-of-mass subspace'


def test_reference_cache_and_missing_error():
    """Builds a tiny cache in the temporary AEM_DATA_DIR, samples from it, and checks the missing-cache error."""
    tmp = os.environ['AEM_DATA_DIR']
    t = get_target('dw4')
    # without a cache the error names the command to run
    empty = tempfile.mkdtemp(dir=tmp)
    os.environ['AEM_DATA_DIR'] = empty
    try:
        get_target('dw4').sample(10, seed=0)
        raise AssertionError('expected FileNotFoundError without a cache')
    except FileNotFoundError as e:
        assert 'python -m aem.mcmc --target dw4' in str(e)
    finally:
        os.environ['AEM_DATA_DIR'] = tmp
    path = build_reference('dw4', n_chains=64, burnin=200, steps=500, thin=10, seed=0, verbose=False)
    assert path.startswith(tmp), f'cache written outside the test folder: {path}'
    a = t.sample(100, seed=7)
    b = t.sample(100, seed=7)
    c = t.sample(100, seed=8)
    assert a.shape == (100, 8) and torch.equal(a, b) and not torch.equal(a, c), 'sample() seeding broken'
    big = t.sample(5000, seed=1)  # more than the cache: sampled with replacement
    assert big.shape == (5000, 8) and torch.isfinite(big).all()
    print(f'  cache {os.path.basename(path)}: {64 * 50} samples; seeded sampling reproducible')


def test_train_egnn():
    """(e) train.py: 2 outer iterations on dw4 with arch=egnn (IDEM and AEM with replay and refinement)."""
    base = tempfile.mkdtemp(dir=os.environ['AEM_DATA_DIR'])
    # n_new > 256 so the refinement blocks have reference chains (refine_buffer needs more chains than one block)
    common = dict(target='dw4', arch='egnn', K=8, n_outer=2, n_inner=5, sde_steps=20, n_eval=200, n_new=300,
                  batch_size=32, hidden=32, n_layers=2, time_emb=16, eval_every=1, n_isir=1)
    for tag, extra in (('idem', dict(method='idem')),
                       ('aem_replay_refine', dict(method='aem', replay=1, refine=1, refine_K=8, anchor_warmup=1))):
        out = os.path.join(base, tag)
        t0 = time.time()
        res = train(out_dir=out, seed=0, **common, **extra)
        with open(res['log']) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        last = recs[-1]
        assert len(recs) == 2, f'expected 2 log records, got {len(recs)}'
        assert 'dist_w1' in last and last['dist_w1'] == last['dist_w1'], 'dist_w1 missing or NaN'
        assert os.path.isfile(os.path.join(out, 'model.pt'))
        print(f'  {tag}: 2 outer iterations in {time.time() - t0:.1f} s, w2={last["w2"]:.3f}, '
              f'dist_w1={last["dist_w1"]:.3f}, mmd={last["mmd"]:.4f}')


def main():
    tmp = tempfile.mkdtemp(prefix='aem_mol_test_')
    os.environ['AEM_DATA_DIR'] = tmp  # all reference caches for the tests live here and are deleted afterwards
    tests = [
        ('(a) energy finiteness and invariances', test_energy_invariances),
        ('(b) centre-of-mass projection', test_projection),
        ('(c) EGNN translation invariance and rotation equivariance', test_egnn_symmetry),
        ('(d) MALA reference sampler on DW4', test_mcmc_dw4),
        ('(e0) reference cache: build, sample, missing-cache error', test_reference_cache_and_missing_error),
        ('(e) train.py with arch=egnn, 2 outer iterations on dw4', test_train_egnn),
    ]
    failed = []
    try:
        for name, fn in tests:
            print(name)
            t0 = time.time()
            try:
                fn()
                print(f'  PASS ({time.time() - t0:.1f} s)')
            except AssertionError as e:
                failed.append(name)
                print(f'  FAIL ({time.time() - t0:.1f} s): {e}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        print('FAILED:', failed)
        sys.exit(1)
    print('all molecular tests passed')


if __name__ == '__main__':
    main()
