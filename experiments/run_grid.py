"""Run the main training grid in parallel worker processes (one torch thread each).

Usage: python experiments/run_grid.py --workers 4 [--only gmm40,gmm16] [--seeds 0,1,2]
Each run writes results/main/<target>/<method>/seed<k>/log.jsonl; runs whose log already has the final
iteration are skipped, so the script can be restarted.
"""
import argparse
import itertools
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMON = ['--kind', 'mix', '--K', '100', '--n_outer', '40', '--n_inner', '100', '--n_eval', '2000',
          '--eval_every', '10']
TARGETS = {
    'gmm40': ['--target', 'gmm40', '--sde_steps', '200', '--sde_steps_eval', '500'],
    'gmm16': ['--target', 'gmm_d', '--dim', '16', '--sde_steps', '200', '--sde_steps_eval', '500'],
    'mw32': ['--target', 'mw32', '--sde_steps', '300', '--sde_steps_eval', '1000', '--tau_max', '0.5'],
    'funnel10': ['--target', 'funnel10', '--sde_steps', '300', '--sde_steps_eval', '1000'],
}
METHODS = {
    'idem': ['--method', 'idem'],
    'aem': ['--method', 'aem'],
    'aem_warm': ['--method', 'aem', '--anchor_warmup', '20'],
    'aem_warm_replay': ['--method', 'aem', '--anchor_warmup', '20', '--replay', '1'],
}


def done(out):
    p = os.path.join(out, 'log.jsonl')
    if not os.path.exists(p):
        return False
    lines = [l for l in open(p) if l.strip()]
    return bool(lines) and json.loads(lines[-1])['iter'] == 40


def run(job):
    tgt, meth, seed = job
    out = os.path.join(ROOT, 'results', 'main', tgt, meth, f'seed{seed}')
    if done(out):
        return f'skip {tgt} {meth} {seed}'
    os.makedirs(out, exist_ok=True)
    cmd = ['python', '-m', 'aem.train', *COMMON, *TARGETS[tgt], *METHODS[meth], '--seed', str(seed),
           '--out_dir', out]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with open(os.path.join(out, 'stdout.txt'), 'w') as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    return f'{"ok" if r.returncode == 0 else "FAIL"} {tgt} {meth} {seed}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--only', default='gmm40,gmm16,mw32')
    p.add_argument('--methods', default='idem,aem,aem_warm,aem_warm_replay')
    p.add_argument('--seeds', default='0,1,2')
    a = p.parse_args()
    jobs = list(itertools.product(a.only.split(','), a.methods.split(','), [int(s) for s in a.seeds.split(',')]))
    # longest targets first
    order = {'mw32': 0, 'funnel10': 1, 'gmm16': 2, 'gmm40': 3}
    jobs.sort(key=lambda j: (order[j[0]], j[2]))
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(run, jobs):
            print(msg, flush=True)


if __name__ == '__main__':
    main()
