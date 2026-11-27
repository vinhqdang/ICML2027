# Exploration without biased targets: design notes (Agent B)

## Literature check (Nov 2026)
- PTSD (Progressive Tempering Sampler with Diffusion, ICML 2025): trains a diffusion model per temperature,
  generates lower-temperature samples from higher-temperature models, refines with MCMC. Temperature ladder, separate
  networks.
- CREPE (arXiv 2509.23265): replica exchange across diffusion noise levels of a *pretrained* model (data setting).
- "Exact global MCMC with denoising diffusion" (arXiv 2609.00279), "MCMC with diffusion paths" (2607.11631),
  "Noise, denoise, correct" (NeurIPS 2026): noise-then-denoise proposals with the learned model, corrected by MH /
  i-SIR on path space. This is our idea 4; it is crowded and needs path densities, so we did not pursue it.
- Search-guided diffusion samplers (NeurIPS 2025): novelty-driven MCMC to collect off-policy samples.
- Classical: AIS (Neal 2001), SMC samplers (Del Moral et al. 2006), FAB (AIS with a flow base), parallel tempering,
  evolutionary Monte Carlo / crossover moves (Liang & Wong 2000), block multiple-try Metropolis (So 2006),
  i-SIR as Gibbs (Andrieu et al. 2018; Samsonov et al. 2022, local-global kernels).
No prior work we found uses an *independent exploration population only as the reference set of an exact
(pi-invariant) augmented i-SIR kernel on a diffusion-sampler replay buffer*, nor subspace (blocked) augmented i-SIR
moves on such a buffer, combined with an unbiased (anchored) training target.

## Diagnosis
The oracle experiment (paper, tab:oracle) shows that AEM trained on an exact buffer gets MW-32 mode weights right
(0.837 vs 0.842). So the 0.97-0.99 major-well fraction of every method is a *buffer* problem. Two separate gaps:
1. Local multimodality (MW-32): the buffer needs moves that flip single coordinates. Whole-vector moves cannot:
   their weights are products over 32 coordinates.
2. Global discovery (GMM-16, cold start): moves need proposals that land in modes the buffer does not contain.
   Whole-vector i-SIR with buffer-KDE proposals only reaches modes already in the buffer.

## Method: decoupled exploration with exact subspace i-SIR moves (XR)
Principle: exploration enters ONLY through the proposal (reference set) of a pi-invariant kernel on the buffer.
The exploration process may be arbitrarily biased; its bias cannot reach the fixed point of anchored training.

### Kernel (aem/explore.py: refine_subspace_block)
For chain x, block S (mask m), noise sigma, reference set R independent of x:
1. x_t,S = x_S + sigma eps_S (exact conditional draw of the auxiliary variable).
2. K proposals: y_{-S} = x_{-S}; y_S ~ q_S = alpha N(x_t,S, sigma^2 I) + (1-alpha) sum_j r_j N(R_j,S, tau^2 I),
   r_j ∝ N(x_t,S; R_j,S, (sigma^2+tau^2) I), tau = min(sigma, tau_max).
3. Resample among {x} ∪ {y_k} with w(y) = exp(-E(y)) N(x_t,S; y_S, sigma^2 I) / q_S(y_S).

Invariance: on the extended space (x, x_t,S) with density pi(x) N(x_t,S; x_S, sigma^2 I), step 1 is an exact Gibbs
update of x_t,S | x. Given x_t,S and x_{-S}, the conditional of x_S is
pi(x_S | x_{-S}) N(x_t,S; x_S, sigma^2) up to a constant, and steps 2-3 are an i-SIR kernel for it with proposal q_S,
which does not depend on x_S (it depends on x_t,S, R and x_{-S} only through y_{-S}=x_{-S}). i-SIR leaves its target
invariant for any proposal (Andrieu, Lee & Vihola 2018). Hence each update leaves pi invariant, for every block S,
every sigma, and every R that is independent of the current chain (S, sigma, R drawn independently of x). A sweep
composes such kernels. With S = all coordinates this is exactly aem.refine.refine_block.

Block choice (generic, not target-specific): with probability p_full the whole vector, otherwise a uniformly random
subset with log-uniform size on [1, b_max]. Small blocks fix local multimodality; KDE proposals on a subspace copy
the S-coordinates of another chain or walker (an exact "crossover"/recombination move).

### Exploration population (aem/explore.py: annealed_walkers)
W walkers start from the VE prior N(0, sigma_max^2 I) and follow the geometric path
gamma_b ∝ N(0, sigma_max^2 I)^{1-b} exp(-b E), b: 0 -> 1 (200 levels, one MALA step per level, adaptive step size).
They use E and grad E only and never see the buffer or the network, so they are independent of the chains. Their
endpoints form a fraction explore_frac of the reference set R (the rest are other buffer chains). Their law need not
be pi: the i-SIR weights correct the mode weights; walkers only need to cover the basins. (AIS weights are available
but not needed.)

### Consistency of the whole sampler
Anchored targets are unbiased when the buffer law is pi (AEM theorem). The buffer evolves by
(i) appending model samples (FIFO) and (ii) pi-invariant XR sweeps. If the model is exact, (i) preserves pi and (ii)
preserves pi, so pi is still a fixed point of the joint (buffer, network) dynamics -- unlike iDEM, whose target is
biased for every buffer, and unlike a biased-exploration phase, which must be switched off.
When the model is not exact, XR sweeps contract the buffer law towards pi (in TV, at the kernel's mixing rate),
including among modes found only by the walkers.

### Cost
Per outer iteration: refine sweeps x N_buffer x (K_refine + 1) energies, plus W x (levels + 1) energy+gradient
evaluations for the walkers. Default (2 sweeps, N=10^4, K=32, W=2000, 200 levels): 0.66M + 0.40M = 1.06M, against
2.56M for the training targets (256 x 100 x 100), i.e. +41%. We compare with iDEM at matched energy budget.
