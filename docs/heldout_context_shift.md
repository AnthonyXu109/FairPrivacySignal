# Heldout Context-Shift Stress Test

This paired diagnostic tests aggregate recovery when synthetic context covariates drift after the household-level train/test split. Training households, labels, and service candidates remain fixed. Only holdout-side income, unemployment, broadband access, food-access risk, health need, housing pressure, and underserved-score context features move across three controlled stress levels. A deterministic share of holdout household income-band and urbanicity buckets is also remapped so the aggregate layer must use different training-fitted cohort lookups.

Privacy-safe aggregates remain fitted from unshifted training households only. Each recovery value is an aggregate-minus-baseline NDCG@3 difference for the same scenario, shift level, and synthetic-data seed. The table reports mean +/- standard deviation across three paired seeds.

| Scenario           | Heldout context shift   | Baseline NDCG@3   | Aggregate NDCG@3   | Overall recovery   | Low-signal recovery   | Fallback event share   | Remapped context-bucket share   |
|:-------------------|:------------------------|:------------------|:-------------------|:-------------------|:----------------------|:-----------------------|:--------------------------------|
| Severe signal loss | Reference               | 0.507 +/- 0.007   | 0.518 +/- 0.008    | +0.012 +/- 0.006   | +0.013 +/- 0.007      | 0.5%                   | 0.0%                            |
| Severe signal loss | Moderate                | 0.507 +/- 0.007   | 0.514 +/- 0.013    | +0.007 +/- 0.008   | +0.009 +/- 0.001      | 1.7%                   | 40.9%                           |
| Severe signal loss | Pronounced              | 0.507 +/- 0.007   | 0.511 +/- 0.014    | +0.004 +/- 0.010   | +0.003 +/- 0.005      | 2.8%                   | 80.8%                           |
| Policy restricted  | Reference               | 0.529 +/- 0.007   | 0.539 +/- 0.004    | +0.010 +/- 0.003   | +0.005 +/- 0.004      | 0.5%                   | 0.0%                            |
| Policy restricted  | Moderate                | 0.529 +/- 0.007   | 0.535 +/- 0.007    | +0.007 +/- 0.006   | +0.006 +/- 0.004      | 1.7%                   | 40.9%                           |
| Policy restricted  | Pronounced              | 0.529 +/- 0.007   | 0.531 +/- 0.008    | +0.002 +/- 0.006   | -0.001 +/- 0.003      | 2.8%                   | 80.8%                           |

## Interpretation limits

This is a synthetic covariate-drift proxy with fixed labels. It is not a temporal validation study, does not estimate real-world distribution shift, and does not establish deployment robustness. The diagnostic is intended to make one additional failure mode inspectable under controlled conditions.
