"""Write the job list of the main GPU grid (results/main_gpu) for experiments/gpu_runner.py.

Job name = <target>/<method>/seed<k>. Everything except the method flags is shared across methods of a target.
Usage: python experiments/main_gpu_jobs.py [--seeds 0,1,2] [--targets ...] [--methods ...] > jobs.txt
"""
import argparse

# (n_outer, anchor_warmup, eval_every) per target; the molecular runs are shorter because the EGNN is slower
BUDGET = {'gmm16': (80, 40, 20), 'mw32': (80, 40, 20), 'dw4': (40, 20, 10), 'lj13': (30, 15, 10)}
COMMON = '--kind mix --K 100 --n_inner 100 --n_eval 2000'
TARGETS = {
    'gmm16': '--target gmm_d --dim 16 --sde_steps 200 --sde_steps_eval 500 --refine_sigma_max 1.0',
    'mw32': '--target mw32 --sde_steps 300 --sde_steps_eval 1000 --tau_max 0.5 --refine_sigma_max 1.0',
    'dw4': '--target dw4 --arch egnn --sde_steps 200 --sde_steps_eval 500 --refine_sigma_max 0.5',
    # LJ-13: the EGNN on 13 particles is ~10x more expensive, so it uses width 128 and a coarser SDE grid
    'lj13': '--target lj13 --arch egnn --hidden 128 --sde_steps 100 --sde_steps_eval 250 --refine_sigma_max 0.3',
}
METHODS = {
    'idem': '--method idem',
    'aem_cold': '--method aem',
    'aem_ea': '--method aem --anchor_warmup {warm}',
    'aem_ea_refine': '--method aem --anchor_warmup {warm} --refine 1',
}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--targets', default=','.join(TARGETS))
    p.add_argument('--methods', default=','.join(METHODS))
    a = p.parse_args()
    for s in a.seeds.split(','):
        for t in a.targets.split(','):
            for m in a.methods.split(','):
                n, warm, ev = BUDGET[t]
                meth = METHODS[m].format(warm=warm)
                print(f'{t}/{m}/seed{s} {COMMON} --n_outer {n} --eval_every {ev} {TARGETS[t]} {meth} --seed {s}')
