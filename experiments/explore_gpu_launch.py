"""Executed on the Colab VM (colab exec -f): unpack /content/B/code.tgz into /content/B/code and start
experiments/gpu_runner.py on /content/B/jobs.txt in the background. Output root /content/B/runs."""
import os
import subprocess
import tarfile

os.makedirs('/content/B/code', exist_ok=True)
with tarfile.open('/content/B/code.tgz') as t:
    t.extractall('/content/B/code')
par = open('/content/B/par.txt').read().strip() if os.path.exists('/content/B/par.txt') else '3'
log = open('/content/B/runner.log', 'a')
p = subprocess.Popen(['python', 'experiments/gpu_runner.py', '/content/B/jobs.txt', '/content/B/runs', par],
                     cwd='/content/B/code', stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print('started', p.pid)
