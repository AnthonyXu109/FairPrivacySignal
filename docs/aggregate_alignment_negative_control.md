# Aggregate-Alignment Negative Control

This diagnostic tests whether observed recovery depends on service-aligned aggregate structure. The negative-control path cyclically permutes service categories in the training reference before privacy-safe aggregates are constructed. Aggregate distributions and train-only fitting are retained, but service-to-signal semantics are deliberately broken.

| Scenario           | Variant                     | Overall NDCG@3   | Overall recovery   | Low-signal recovery   |
|:-------------------|:----------------------------|:-----------------|:-------------------|:----------------------|
| Severe signal loss | No aggregate substitutes    | 0.507 +/- 0.007  | +0.000 +/- 0.000   | +0.000 +/- 0.000      |
| Severe signal loss | Aligned aggregates          | 0.518 +/- 0.008  | +0.012 +/- 0.006   | +0.013 +/- 0.007      |
| Severe signal loss | Service-permuted aggregates | 0.505 +/- 0.008  | -0.001 +/- 0.001   | -0.002 +/- 0.003      |
| Policy restricted  | No aggregate substitutes    | 0.529 +/- 0.007  | +0.000 +/- 0.000   | +0.000 +/- 0.000      |
| Policy restricted  | Aligned aggregates          | 0.539 +/- 0.004  | +0.010 +/- 0.003   | +0.005 +/- 0.004      |
| Policy restricted  | Service-permuted aggregates | 0.530 +/- 0.010  | +0.001 +/- 0.003   | +0.005 +/- 0.008      |

## Current result

Under severe signal loss, mean overall recovery changes from `+0.0115` with aligned aggregates to `-0.0014` after service permutation. Under the policy-restricted scenario, mean overall recovery changes from `+0.0102` to `+0.0009`. The policy-restricted low-signal comparison is less separated, so the negative control should be read metric by metric rather than as a uniform effect.

## Interpretation limits

This is a synthetic structural negative control. A weaker permuted result supports the interpretation that service alignment matters in this benchmark, but it does not establish causality or transfer to a real deployment.
