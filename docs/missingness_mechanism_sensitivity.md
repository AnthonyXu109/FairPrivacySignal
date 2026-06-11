# Missingness-Mechanism Sensitivity

This diagnostic holds the overall behavioral-signal availability rate at 56% while changing which events retain signal. It separates signal quantity from incidence using controlled uniform-random, observed-context, and signal-dependent mechanisms.

| Mechanism                              | Overall availability   | Low-signal availability   | Not-low availability   | Baseline NDCG@3   | Aggregate recovery   | Low-signal recovery   |
|:---------------------------------------|:-----------------------|:--------------------------|:-----------------------|:------------------|:---------------------|:----------------------|
| MCAR-like: Uniform random              | 56.0%                  | 56.3%                     | 55.8%                  | 0.520 +/- 0.010   | +0.010 +/- 0.008     | +0.010 +/- 0.009      |
| MAR-like: Observed-context conditioned | 56.0%                  | 53.2%                     | 57.6%                  | 0.523 +/- 0.009   | +0.006 +/- 0.003     | +0.005 +/- 0.007      |
| MNAR-like: Signal-dependent            | 56.0%                  | 32.8%                     | 69.4%                  | 0.535 +/- 0.011   | +0.008 +/- 0.004     | +0.013 +/- 0.002      |

## Current result

The overall holdout availability rate is fixed at 56% for every mechanism, but low-signal availability changes from 56.3% under the uniform-random mechanism to 32.8% under the signal-dependent mechanism. The signal-dependent baseline has higher overall NDCG@3 because high-engagement events are preferentially retained, showing why matched aggregate availability does not imply matched subgroup incidence or comparable ranking difficulty.

## Interpretation limits

The MCAR-like, MAR-like, and MNAR-like labels describe synthetic mechanism analogues. They are not empirical missingness diagnoses, causal estimates, or implementations of inverse-propensity learning. The signal-dependent path intentionally uses the value later suppressed to create an adversarial incidence pattern.
