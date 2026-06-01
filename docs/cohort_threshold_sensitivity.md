# Cohort-Threshold Sensitivity

FairPrivacySignal applies k-thresholding before using privacy-safe cohort
aggregates. When a cohort contains fewer than `k` households, the benchmark
suppresses its cohort-level aggregate and uses a broad service-level fallback.

## Design

The sensitivity diagnostic fixes the generated synthetic dataset and DP-style noise
configuration while sweeping the minimum cohort-size threshold:

| Parameter | Values |
|---|---|
| Minimum cohort size (`k`) | `25`, `50`, `100`, `200`, `400`, `800` |
| DP-style aggregate-noise scale | `1.0` |
| Aggregate-noise seed | `42` |
| Signal-loss scenarios | Severe signal loss, policy restricted |
| Metrics | Fallback event share, overall NDCG@3 recovery, low-signal NDCG@3 recovery |

The default benchmark threshold is `k=50`. The sweep intentionally includes more
restrictive values so that reviewers can see how fallback coverage and utility
change when more cohort aggregates are suppressed.

## Interpretation

This diagnostic makes the threshold mechanism inspectable. It does not prove that a
particular threshold is appropriate for a real deployment, and k-thresholding alone
does not provide a formal privacy guarantee. Real settings require domain-specific
privacy analysis, cohort definitions, and threat modeling.
