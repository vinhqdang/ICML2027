"""Keep every `every`-th saved time of each chain of a cached MALA reference run (to shrink the file).
Usage: python -m experiments.thin_ref <in.pt> <out.pt> <every>"""
import sys

import torch

src, dst, every = sys.argv[1], sys.argv[2], int(sys.argv[3])
b = torch.load(src)
C, T = b['n_chains'], b['steps'] // b['thin']
x = b['samples'].view(C, T, -1)[:, ::every].contiguous()
b['samples'] = x.reshape(-1, x.shape[-1])
b['thin_extra'] = every
print(src, '->', dst, tuple(b['samples'].shape))
torch.save(b, dst)
