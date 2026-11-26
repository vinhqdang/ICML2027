# Anchored Energy Matching (AEM)

Code and manuscript for *Anchored Energy Matching: Unbiased Regression Targets for Diffusion Samplers of Boltzmann Densities* (ICML 2027 submission draft).

**Idea.** Energy-matching diffusion samplers (iDEM and successors) regress a score network on a self-normalised importance sampling (SNIS) estimate of the noised score. The population optimum of that loss does not depend on the replay buffer, so the true score is not a fixed point of training. AEM adds the clean buffer sample that generated each noised point as an extra "anchor" particle (conditional importance sampling / i-SIR). This makes the target exactly unbiased whenever the buffer is exact, and it contracts the bias when the buffer is not.

## Layout
- `aem/`: library code. `estimators.py` has the iDEM and AEM score estimators, `replay.py` the replay (buffer-KDE) proposals, `train.py` iDEM/AEM training (including the explore-then-anchor and oracle-buffer modes), `targets.py` the benchmark targets, and `metrics.py` the evaluation metrics.
- `experiments/`: the training grid (`run_grid.py`), aggregation and plotting, the estimator studies (`replay_check.py`, `dim_sweep.py`) and the oracle score-error evaluation.
- `results/`: estimator-study CSVs and training logs.
- `paper/`: LaTeX sources (ICML 2026 style; the 2027 style is not yet released). Build with `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
- `docs/research_plan.md`: the original research plan.

## Reproduce
```bash
pip install torch numpy scipy matplotlib pandas
python -m aem.estimator_study --target gmm40 --out results/
python experiments/run_grid.py --workers 4
python experiments/aggregate.py
```
Everything runs on CPU.
