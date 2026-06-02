# Pairwise Ranking-Objective Sensitivity Diagnostic

This diagnostic compares the interpretable pointwise logistic primary baseline with a lightweight linear pairwise ranker. The comparator creates ordered relevant-versus-nonrelevant service pairs within each synthetic household, then learns a linear score from feature differences.

Both objectives receive the same synthetic-data draws, household-level train/test split, signal-loss scenarios, and training-fitted privacy-safe aggregate features.

## Paired aggregate-recovery deltas

| Training objective     | Scenario           | Overall recovery   | Low-signal recovery   |
|:-----------------------|:-------------------|:-------------------|:----------------------|
| Pointwise logistic     | Severe signal loss | +0.012 +/- 0.006   | +0.013 +/- 0.007      |
| Pointwise logistic     | Policy restricted  | +0.010 +/- 0.003   | +0.005 +/- 0.004      |
| Linear pairwise ranker | Severe signal loss | +0.011 +/- 0.006   | +0.013 +/- 0.005      |
| Linear pairwise ranker | Policy restricted  | +0.011 +/- 0.004   | +0.004 +/- 0.003      |

## Scenario scores

| Training objective     | Scenario                       | Overall NDCG@3   | Low-signal NDCG@3   |
|:-----------------------|:-------------------------------|:-----------------|:--------------------|
| Pointwise logistic     | Full signal                    | 0.557 +/- 0.011  | 0.495 +/- 0.014     |
| Pointwise logistic     | Severe loss                    | 0.507 +/- 0.007  | 0.434 +/- 0.018     |
| Pointwise logistic     | Severe loss + aggregates       | 0.518 +/- 0.008  | 0.447 +/- 0.012     |
| Pointwise logistic     | Policy restricted              | 0.529 +/- 0.007  | 0.456 +/- 0.008     |
| Pointwise logistic     | Policy restricted + aggregates | 0.539 +/- 0.004  | 0.461 +/- 0.005     |
| Linear pairwise ranker | Full signal                    | 0.556 +/- 0.012  | 0.486 +/- 0.013     |
| Linear pairwise ranker | Severe loss                    | 0.507 +/- 0.007  | 0.434 +/- 0.018     |
| Linear pairwise ranker | Severe loss + aggregates       | 0.518 +/- 0.008  | 0.447 +/- 0.013     |
| Linear pairwise ranker | Policy restricted              | 0.529 +/- 0.007  | 0.455 +/- 0.008     |
| Linear pairwise ranker | Policy restricted + aggregates | 0.539 +/- 0.005  | 0.460 +/- 0.006     |

## Interpretation limits

The pairwise comparator is a lightweight linear sensitivity check inspired by pairwise learning-to-rank formulations. It is not an implementation of a neural ranking architecture, does not optimize a listwise objective, and does not replace the interpretable primary baseline.
