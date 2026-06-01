# Community-Held-Out Robustness Diagnostic

This diagnostic compares the primary household-level holdout with a stricter synthetic community-held-out stress test. In the stricter path, training and evaluation communities are disjoint. Privacy-safe aggregates remain fitted from training households only before scoring holdout events.

Each recovery value is a paired aggregate-minus-baseline NDCG@3 difference for the same signal-loss scenario, split strategy, and synthetic-data seed. The table reports mean +/- standard deviation across five seeds.

| Scenario           | Evaluation split   | Overall recovery   | Low-signal recovery   | Fallback event share   | Unseen cohort share   | Held-out community share   |
|:-------------------|:-------------------|:-------------------|:----------------------|:-----------------------|:----------------------|:---------------------------|
| Severe signal loss | Household holdout  | +0.015 +/- 0.008   | +0.015 +/- 0.009      | 0.5%                   | 0.0%                  | 0.0%                       |
| Severe signal loss | Community holdout  | +0.020 +/- 0.007   | +0.024 +/- 0.005      | 0.8%                   | 0.0%                  | 100.0%                     |
| Policy restricted  | Household holdout  | +0.012 +/- 0.004   | +0.008 +/- 0.009      | 0.5%                   | 0.0%                  | 0.0%                       |
| Policy restricted  | Community holdout  | +0.014 +/- 0.003   | +0.015 +/- 0.004      | 0.8%                   | 0.0%                  | 100.0%                     |

## Interpretation limits

This is a synthetic grouped-holdout diagnostic. It checks whether the benchmark claim survives a stricter separation of generated community contexts, but it is not a real geographic, temporal, or deployment validation study. The household-level holdout remains the primary benchmark protocol.
