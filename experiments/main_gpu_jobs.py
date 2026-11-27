"""Write the job list of the main GPU grid (results/main_gpu) for experiments/gpu_runner.py.

Job name = <target>/<method>/seed<k>. Everything except the method flags is shared across methods of a target.
Usage: python experiments/main_gpu_jobs.py [--seeds 0,1,2] [--targets ...] [--methods ...] > jobs.txt
"""
import argparse

N_OUTER, WARM = 80, 40
COMMON = f'--kind mix --K 100 --n_outer {N_OUTER} --n_inner 100 --n_eval 2000 --eval_every 20'
TARGETS = {
    'gmm16': '--target gmm_d --dim 16 --sde_steps 200 --sde_steps_eval 500 --refine_sigma_max 1.0',
    'mw32': '--target mw32 --sde_steps 300 --sde_steps_eval 1000 --tau_max 0.5 --refine_sigma_max 1.0',
    'dw4': '--target dw4 --arch egnn --sde_steps 200 --sde_steps_eval 500 --refine_sigma_max 0.5',
    'lj13': '--target lj13 --arch egnn --sde_steps 200 --sde_steps_eval 500 --refine_sigma_max 0.3',
}
METHODS = {
    'idem': '--method idem',
    'aem_cold': '--method aem',
    'aem_ea': f'--method aem --anchor_warmup {WARM}',
    'aem_ea_refine': f'--method aem --anchor_warmup {WARM} --refine 1',
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
                print(f'{t}/{m}/seed{s} {COMMON} {TARGETS[t]} {METHODS[m]} --seed {s}')
