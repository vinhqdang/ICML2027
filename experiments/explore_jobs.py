"""Write the GPU job list for the exploration study (consumed by experiments/gpu_runner.py on the VM).
Usage: python experiments/explore_jobs.py SEEDS > jobs.txt   (SEEDS like 0,1,2)
Common settings match results/gpu_pilot2 (K=100, 60 outer x 100 inner, kind=mix)."""
import sys

seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else '0').split(',')]
which = sys.argv[2].split(',') if len(sys.argv) > 2 else None
COMMON = '--kind mix --K 100 --n_outer 60 --n_inner 100 --eval_every 10 --n_eval 2000'
TGT = {
    'gmm16': '--target gmm_d --dim 16 --sde_steps 200 --sde_steps_eval 500',
    'mw32': '--target mw32 --sde_steps 300 --sde_steps_eval 1000 --tau_max 0.5',
}
XR = '--method aem --refine 1 --refine_K 32 --refine_mode subspace --refine_start 1 --explore_walkers 2000'
METH = {
    'idem': '--method idem',
    'aemwarmref': '--method aem --anchor_warmup 30 --refine 1',  # explore -> anchor + whole-vector refinement
    'xr': XR + ' --sub_blocks 32 --sub_K 4 --sub_bmax 1',        # anchored from the start + XR
    'xr_lite': XR.replace('--explore_walkers 2000', '--explore_walkers 500') + ' --sub_blocks 8 --sub_K 4 --sub_bmax 1',
    'xr_heavy': XR + ' --sub_blocks 96 --sub_K 4 --sub_bmax 1',  # 3x the coordinate updates
    'xr_nowalk': XR.replace('--explore_walkers 2000', '--explore_walkers 0') + ' --sub_blocks 32 --sub_K 4 --sub_bmax 1',
    'idem_matched': '--method idem --n_inner 192',               # iDEM at XR's energy budget (~295M evals)
}
for t, targs in TGT.items():
    for m, margs in METH.items():
        if which and m not in which:
            continue
        for s in seeds:
            print(f'{t}_{m}_s{s} {COMMON} {targs} {margs} --seed {s}')
