"""Run a list of aem.train configurations on a Colab GPU VM, two at a time.
Usage (on the VM): python experiments/gpu_runner.py jobs.txt OUTROOT
Each line of jobs.txt: <name> <aem.train args...>"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

jobs = [l.split(maxsplit=1) for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
root = sys.argv[2]
par = int(sys.argv[3]) if len(sys.argv) > 3 else 2


def run(job):
    name, args = job
    out = os.path.join(root, name)
    if os.path.exists(os.path.join(out, 'samples.npy')):
        return f'skip {name}'
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, '-m', 'aem.train', '--device', 'cuda', *args.split(), '--out_dir', out]
    with open(os.path.join(out, 'stdout.txt'), 'w') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return f'{"ok" if r.returncode == 0 else "FAIL"} {name}'


with ThreadPoolExecutor(par) as ex:
    for m in ex.map(run, jobs):
        print(m, flush=True)
