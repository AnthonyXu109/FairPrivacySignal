# Disparate Uncertainty Audit

This diagnostic fits household-bootstrap ensembles on a fixed holdout split. It reports ensemble-mean ranking utility, prediction variability, and the share of Top-3 services that agree with the ensemble-mean ranking.

| Experiment                     | Overall NDCG@3   | Low-signal score std   | Not-low score std   | Low-signal Top-3 agreement   | Not-low Top-3 agreement   |
|:-------------------------------|:-----------------|:-----------------------|:--------------------|:-----------------------------|:--------------------------|
| Full signal                    | 0.557 +/- 0.011  | 0.0101                 | 0.0101              | 0.963                        | 0.985                     |
| Severe loss                    | 0.505 +/- 0.009  | 0.0110                 | 0.0109              | 0.944                        | 0.944                     |
| Severe loss + aggregates       | 0.517 +/- 0.007  | 0.0169                 | 0.0171              | 0.914                        | 0.915                     |
| Policy restricted              | 0.529 +/- 0.006  | 0.0110                 | 0.0109              | 0.942                        | 0.961                     |
| Policy restricted + aggregates | 0.537 +/- 0.004  | 0.0159                 | 0.0159              | 0.909                        | 0.927                     |

## Paired aggregate effects

| scenario           |   overall_ndcg_recovery |   low_signal_uncertainty_change |   low_signal_top3_agreement_change |
|:-------------------|------------------------:|--------------------------------:|-----------------------------------:|
| policy_restricted  |                 +0.0084 |                         +0.0049 |                            -0.0323 |
| severe_signal_loss |                 +0.0114 |                         +0.0060 |                            -0.0303 |

## Current result

Aggregate recovery improves ensemble-mean overall NDCG@3 by `+0.0114` under severe signal loss and `+0.0084` under policy restriction. However, mean low-signal prediction variability increases by `+0.0060` and `+0.0049`, respectively, while mean low-signal Top-3 agreement changes by `-0.0303` and `-0.0323`. The score-standard-deviation gap between groups is small and not consistently directional in this configuration.

## Interpretation limits

Bootstrap prediction standard deviation is a training-resample instability diagnostic. It is not a calibrated posterior, a confidence interval for an individual event, or an implementation of Equal-Opportunity Ranking. Top-3 agreement measures membership stability, not ranking correctness.
