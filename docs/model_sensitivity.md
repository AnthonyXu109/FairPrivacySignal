# Model Sensitivity Diagnostic

This diagnostic compares the interpretable logistic primary baseline with histogram gradient boosting. Both models receive the same synthetic-data draws, household-level train/test split, signal-loss scenarios, and privacy-safe aggregate features.

## Paired aggregate-recovery deltas

| Model                       | Scenario           | Overall recovery   | Low-signal recovery   |
|:----------------------------|:-------------------|:-------------------|:----------------------|
| Logistic baseline           | Severe signal loss | +0.012 +/- 0.006   | +0.013 +/- 0.007      |
| Logistic baseline           | Policy restricted  | +0.010 +/- 0.003   | +0.005 +/- 0.004      |
| Histogram gradient boosting | Severe signal loss | -0.000 +/- 0.002   | -0.001 +/- 0.012      |
| Histogram gradient boosting | Policy restricted  | -0.001 +/- 0.004   | -0.003 +/- 0.007      |

## Scenario scores

| Model                       | Scenario                       | Overall NDCG@3   | Low-signal NDCG@3   |
|:----------------------------|:-------------------------------|:-----------------|:--------------------|
| Logistic baseline           | Full signal                    | 0.557 +/- 0.011  | 0.495 +/- 0.014     |
| Logistic baseline           | Severe loss                    | 0.507 +/- 0.007  | 0.434 +/- 0.018     |
| Logistic baseline           | Severe loss + aggregates       | 0.518 +/- 0.008  | 0.447 +/- 0.012     |
| Logistic baseline           | Policy restricted              | 0.529 +/- 0.007  | 0.456 +/- 0.008     |
| Logistic baseline           | Policy restricted + aggregates | 0.539 +/- 0.004  | 0.461 +/- 0.005     |
| Histogram gradient boosting | Full signal                    | 0.557 +/- 0.005  | 0.493 +/- 0.009     |
| Histogram gradient boosting | Severe loss                    | 0.524 +/- 0.008  | 0.456 +/- 0.009     |
| Histogram gradient boosting | Severe loss + aggregates       | 0.524 +/- 0.006  | 0.455 +/- 0.015     |
| Histogram gradient boosting | Policy restricted              | 0.533 +/- 0.005  | 0.444 +/- 0.006     |
| Histogram gradient boosting | Policy restricted + aggregates | 0.533 +/- 0.004  | 0.442 +/- 0.002     |

## Interpretation limits

Histogram gradient boosting is a lightweight model-class sensitivity check, not a ranking-specific objective and not a replacement for the interpretable primary baseline. Differences across models show that aggregate-recovery results should be reported with model context rather than treated as model-invariant.
