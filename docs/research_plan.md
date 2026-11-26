# Research plan: Anchored Energy Matching (AEM)

## Problem
We want to sample from a Boltzmann density pi(x) ∝ exp(-E(x)) when we have E and its gradient but no data.
Energy-based diffusion samplers (iDEM, Akhound-Sadegh et al., ICML 2024, and the follow-ups iEFM, BNEM, CVSI, IWSM)
regress a score network s_theta(x_t, t) onto a Monte Carlo estimate of the noised target score

    grad log p_t(x_t) = E_{pi(x0 | x_t)}[ grad log pi(x0) ]           (TSI)
                      = E_{pi(x0 | x_t)}[ (x0 - x_t) / sigma_t^2 ]      (DSI)

using self-normalised importance sampling (SNIS) with K proposals x0^k ~ N(x_t, sigma_t^2 I).

## Gap
SNIS is biased at finite K. The bias is largest at high noise, which is exactly where mode-finding happens.
It does not vanish when the sampler is perfect: even if the replay buffer holds exact samples from pi,
the iDEM regression target is still a biased field b_K != grad log p_t. So the true score is NOT a
fixed point of iDEM training. Prior fixes (more samples, bootstrapping, control variates) reduce this
bias or variance but keep the same structure.

## Key idea
Every training pair already contains a sample from the posterior: x_t is made by noising a buffer sample x0*,
so if the buffer were exact, x0* | x_t ~ pi(x0 | x_t). Add x0* as an extra *anchor* particle to the SNIS
estimate (conditional importance sampling, the T=1 case of conditional SMC / i-SIR):

    S_A = ( w(x0*) g(x0*) + sum_k w(x0^k) g(x0^k) ) / ( w(x0*) + sum_k w(x0^k) ),   w(x) = exp(-E(x)).

## Claims to prove
1. Exact unbiasedness for every K >= 0 when the anchor is an exact posterior sample, so the true score is a
   fixed point of the training operator. K = 0 recovers target/denoising score matching. K -> inf recovers iDEM.
2. Bias contraction with an approximate anchor (buffer law rho != pi): the anchored target is the posterior mean
   after one i-SIR step from rho(.|x_t). Doeblin minorisation gives
   |bias| <= 2 ||g||_inf (1 - eps_K) TV(rho(.|x_t), pi(.|x_t)), with eps_K = K / (K - 1 + 2 M_t).
   Iterating the kernel n times gives (1 - eps_K)^n.
3. Graceful degradation: a poor anchor (low weight) leaves the estimator approximately equal to iDEM. A good anchor
   in a mode dominates when proposals miss all the modes (large sigma), which removes the dominant bias.

## Experiments
- E1, estimator level: bias, variance and MSE of the score estimate against sigma and K on GMM-40, MW-32 and Funnel,
  using exact anchors and corrupted anchors (tempered or mode-dropped).
- E2, full sampler: iDEM vs AEM (and the DSI/TSI variants) on GMM-40 (2D), Funnel (10D), MW-32 (32D) and GMM in
  higher dimension. Metrics: sample W2, energy W1, MMD, mode coverage. Compute is matched.
- E3, ablations: K, number of i-SIR steps n, buffer size, anchor-weight diagnostics.
