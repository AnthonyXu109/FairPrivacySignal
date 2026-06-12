# Reviewer Guide

This guide provides a short audit path through FairPrivacySignal. It is intended
for readers who want to assess the methodological substance before inspecting
every implementation detail.

## Ten-Minute Review Path

### 1. Establish the scope

Read the project statement and evidence boundaries in the
[README](../README.md#scope) and
[limitations](limitations.md).

The project asks how much ranking utility can be recovered when individual
behavioral signals are reduced, without restoring unavailable raw values to the
serving model. It uses synthetic data and does not claim deployment validation or
formal privacy accounting.

### 2. Inspect the experimental system

Use the [method system map](../README.md#method) for the full path
from synthetic inputs to validation. Then inspect the
[benchmark design](benchmark_design.md) for the data-generating process,
signal-loss scenarios, train/test protocol, aggregate construction, models, and
metrics.

Three implementation details are especially important:

- holdout households do not contribute to aggregate feature construction
- signal reconstruction uses household-grouped cross-fitting for downstream
  training proxies
- the serving ranker never receives unavailable raw behavioral values
- privacy exposure, ranking utility, subgroup diagnostics, and allocation outcomes
  are reported as separate evidence surfaces

### 3. Check the central result

Start with [policy-aware signal recovery](policy_aware_signal_recovery.md), which
contains the method, ablation, paired five-seed results, and applicability
boundary. The generated [benchmark card](benchmark_card.md) summarizes the wider
evidence surface.

The central claim is deliberately narrow: the policy-conditioned recovery method
improves overall and low-signal NDCG@3 over the matching no-recovery baseline in
both tested signal-loss regimes without increasing the exposure proxy. The result
is not presented as full recovery, universal model behavior, or a solved fairness
problem.

### 4. Look for failure-focused evidence

The benchmark includes diagnostics designed to challenge the main result:

- [aggregate-alignment negative control](aggregate_alignment_negative_control.md):
  breaks service-to-signal semantics while preserving the aggregate pipeline
- [missingness-mechanism sensitivity](missingness_mechanism_sensitivity.md):
  fixes total signal quantity while changing which events retain signal
- [model sensitivity](model_sensitivity.md): checks whether recovery survives a
  different lightweight model class
- [uncertainty and ranking-stability audit](disparate_uncertainty_audit.md):
  checks whether utility recovery remains stable under household resampling
- [heldout context shift](heldout_context_shift.md): moves evaluation-side context
  while keeping training data and labels fixed
- [underserved quartile profile](underserved_recovery_profile.md): checks whether
  pooled gains hide context-specific regressions
- [cohort-threshold sensitivity](cohort_threshold_sensitivity.md): exposes the
  fallback-coverage and utility tradeoff

These diagnostics are useful because several results weaken, disappear, or become
uneven under stress. That behavior keeps the evidence falsifiable and the claims
bounded.

### 5. Verify provenance and reproducibility

The [validation report](validation_report.md) records required methodological
checks, while the [reproducibility guide](reproducibility.md) lists commands and
generated artifacts. The pull-request workflow runs the same complete verification
entry point:

```bash
bash scripts/verify_benchmark.sh
```

The pipeline rebuilds synthetic data, experiments, figures, generated summaries,
the benchmark card, and the validation report before required checks are enforced.

## Claim-Evidence Map

| Question | Primary evidence | Important boundary |
|---|---|---|
| Does signal loss reduce ranking utility? | [Multi-seed results](multiseed_results.md) | Synthetic scenario, not an impact estimate |
| Does the proposed method recover utility? | [Policy-aware signal recovery](policy_aware_signal_recovery.md) | Requires controlled offline access to historical signal for reconstruction |
| How much of the full-signal gap is closed? | [Policy-aware signal recovery](policy_aware_signal_recovery.md) | Partial recovery in a synthetic task |
| Which component provides the gain? | [Policy-aware method ablation](policy_aware_signal_recovery.md) and [aggregate feature ablation](recovery_feature_ablation.md) | Contributions vary by policy regime |
| Does recovery depend on meaningful aggregate structure? | [Alignment negative control](aggregate_alignment_negative_control.md) | Structural diagnostic, not causal identification |
| Does equal signal quantity imply equal incidence? | [Missingness-mechanism sensitivity](missingness_mechanism_sensitivity.md) | Controlled analogues, not empirical missingness diagnoses |
| Does utility recovery imply stable rankings? | [Uncertainty audit](disparate_uncertainty_audit.md) | Bootstrap instability, not calibrated posterior uncertainty |
| Are pooled gains uniform across contexts? | [Underserved quartile profile](underserved_recovery_profile.md) | Some subgroup-context deltas are weak or negative |
| Does the result survive stricter evaluation stress? | [Community holdout](community_holdout_robustness.md) and [context shift](heldout_context_shift.md) | Synthetic robustness tests, not geographic or temporal validation |
| What happens after capacity constraints? | [Capacity allocation](capacity_allocation.md) | Simplified allocation policies |
| Are methodological invariants enforced? | [Validation report](validation_report.md) | Checks validate the declared benchmark protocol |

## Reviewer Checklist

- Are signal-loss scenarios explicit and reproducible?
- Are aggregate features fitted without holdout-household leakage?
- Are results paired across multiple synthetic seeds?
- Are utility, low-signal diagnostics, privacy exposure, and allocation separated?
- Do ablations and negative controls test alternative explanations?
- Are model, threshold, community, and context-shift sensitivities visible?
- Do the stated limitations match the evidence actually produced?
