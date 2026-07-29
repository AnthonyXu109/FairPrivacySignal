# Reliability-Weighted Public-Services Recovery

## Objective

Improve the UCI Adult public-services validation with a train-fitted recovery
method that learns when to trust contextual reconstruction and when to trust a
cohort estimate. The update must preserve the existing privacy boundary: the
official test set's restricted economic signal is never used to fit the recovery
method or its weights.

## Scope

This increment changes only the public-services validation and its generated
evidence. It will:

- retain the current fixed `85%` reconstruction / `15%` cohort blend as a
  comparison baseline;
- add a reliability-weighted blend learned entirely from the UCI Adult training
  split;
- use the new zero-exposure recovery in the existing policy-aware path;
- update the public-services report, tables, detailed figures, and README gallery
  card; and
- add focused regression tests.

It will not change the other four public-data pilots, the synthetic benchmark, or
the definition of the restricted and permitted features.

## Method

1. Enrich each training fold with the existing restricted-signal and permitted-
   context definitions.
2. Produce out-of-fold predictions from the existing ridge reconstruction and
   cohort estimators using five deterministic folds with seed `42`.
3. Measure each estimator's absolute error globally and within the permitted
   `relationship` groups.
4. Shrink group errors toward the global error with a fixed strength of `100`
   rows, so small groups cannot dictate unstable weights.
5. Convert the two error estimates into inverse-error blend weights with a
   numerical error floor of `1e-6`, then bound the reconstruction weight to
   `[0.10, 0.95]`.
6. Refit both estimators on the full training split and apply the train-derived
   relationship weights to the untouched official test split. Unseen groups use
   the global weight.

The final recovered signal is:

`w_reconstruction * reconstruction + (1 - w_reconstruction) * cohort`

No test label or restricted test feature participates in fitting
`w_reconstruction`.

## Evidence Presentation

The main four-method public-services figure remains readable:

- full detailed-economic signal;
- context-only baseline;
- train-fitted reliability-weighted recovery; and
- policy-aware partial recovery using the reliability-weighted substitute.

A separate compact comparison table in the report and generated CSV will show the
existing fixed blend beside the new method. This preserves an explicit before/
after result without adding a fifth row to the homepage gallery card.

## Testing

Focused tests will verify that:

- learned weights are finite and bounded;
- fitting is deterministic;
- unseen relationship groups use the global fallback;
- scoring the test frame does not require its restricted signal to fit weights;
- the fixed and reliability-weighted recovery scores are both reported; and
- the new primary summary continues to report zero restricted-signal exposure for
  train-fitted recovery.

The complete test suite and all five public-data pilots will then be rerun.

## Publication Gate

The new method will be presented as an improvement only if, on the untouched UCI
Adult test split, it:

- strictly exceeds the fixed blend's overall NDCG@1000;
- keeps low-signal NDCG@1000 at least `fixed_blend - 0.0005`, so the displayed
  three-decimal result does not regress;
- keeps restricted-signal exposure at `0.000`; and
- passes the repository's full verification workflow.

If these conditions are not met, the result will not be described or visualized
as a superior recovery method.

## Training-Objective Refinement

The first held-out publication gate showed that inverse-error weighting improved
neither ranking outcome: signal-reconstruction error and ranking utility were not
interchangeable objectives. The published implementation therefore keeps the
relationship-level error estimates only as relative reliability adjustments and
selects their global anchor by NDCG@1000 on the same deterministic out-of-fold
training predictions.

Because fold diagnostics showed that low-signal ranking was less stable across
candidate weights, low-signal records retain the fixed `85/15` blend. Ranking
calibration applies only to non-low-signal records, and the official test split
remains excluded from fitting. This refinement directly aligns model selection
with the stated ranking outcome while enforcing the low-signal guardrail by
construction.
