# Methodology

## Problem

For entity \(i\) and candidate \(j\), the ranking model predicts relevance from
permitted context \(x_{ij}\) and a behavioral signal \(b_{ij}\). A policy mask
\(m_{ij}\) determines whether the behavioral value is available when the model is
served.

The objective is to improve ranking quality when \(m_{ij}=0\) without providing
the hidden raw value to the serving ranker.

## Evaluation split

Synthetic households are divided into training and holdout partitions. Every event
for one household remains in the same partition. Aggregate mappings and
reconstruction models are fitted from training households only.

## Policy-Aware Signal Recovery

The method has two paths:

1. **Complete signal loss:** use thresholded, noise-stressed cohort aggregates.
2. **Partial signal loss:** learn a behavioral-signal reconstruction from permitted
   context, preserve observed values, substitute unavailable values, and fuse the
   result with train-fitted aggregates.

Five-fold household-grouped cross-fitting generates reconstructed training
features. A final reconstruction model fitted on all training households generates
holdout proxies.

## Comparators

The primary ablation includes:

- full-signal oracle
- no-recovery signal-loss baseline
- explicit missingness indicator
- flat privacy-safe aggregates
- signal reconstruction alone
- policy-aware recovery

Every comparison uses the same generated population, household split, downstream
model family, and seed.

## Metrics

The main ranking metric is NDCG@3. Results also report AUC, low-signal NDCG@3,
the low-signal ranking gap, behavioral availability, and a comparison-only privacy
exposure proxy.

For method \(r\), the fraction of the full-signal utility gap closed is:

\[
\frac{\mathrm{NDCG}_r-\mathrm{NDCG}_{loss}}
{\mathrm{NDCG}_{full}-\mathrm{NDCG}_{loss}}.
\]

The repository reports mean and standard deviation across five paired synthetic
data seeds. Required validation checks enforce that the primary method improves
both overall and low-signal NDCG@3 in every paired seed without increasing the
exposure proxy.

## Interpretation

The evaluation demonstrates controlled synthetic recovery, not real-world causal
impact or deployment effectiveness. See [limitations.md](limitations.md) and
[privacy_design.md](privacy_design.md).
