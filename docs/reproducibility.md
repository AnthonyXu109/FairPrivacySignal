# Reproducibility Guide

This guide explains how to reproduce the FairPrivacySignal benchmark results from a clean checkout.

## 1. Create a Python environment

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt

## 2. Run the full benchmark pipeline

    bash scripts/run_benchmark.sh

This command regenerates:

- synthetic public-service outreach data
- sanity-check figures
- signal-loss summaries
- baseline ranking metrics
- privacy-safe recovery metrics
- train-fitted aggregate preprocessing for holdout evaluation
- fairness-aware recovery diagnostics
- multi-seed benchmark results
- capacity-constrained allocation metrics
- capacity-sensitivity frontier metrics
- multi-seed capacity-sensitivity metrics
- score-matched subgroup calibration metrics
- public-reference calibration metrics from a tracked Census Bureau snapshot
- aggregate-noise sensitivity metrics across reproducible perturbations
- cohort-threshold sensitivity metrics for aggregate fallback coverage
- paired multi-seed recovery feature-ablation metrics
- paired aggregate-alignment negative-control metrics
- paired matched-rate missingness-mechanism metrics
- household-bootstrap prediction-instability and Top-3 stability metrics
- paired model-sensitivity metrics for logistic and histogram gradient boosting
- paired ranking-objective metrics for pointwise logistic, linear pairwise, and linear listwise training
- paired underserved-quartile recovery metrics across synthetic community contexts
- paired community-held-out robustness metrics across disjoint synthetic contexts
- paired heldout context-shift metrics across three controlled drift levels
- benchmark overview figure
- architecture diagram
- benchmark validation report
- reviewer-facing benchmark card

Signal-suppression scenarios and privacy-exposure weights are loaded from:

    config/policy_rules.json
    config/public_reference_targets.csv

## 3. Key result files

    docs/assets/multiseed_privacy_recovery_ndcg.png
    docs/assets/multiseed_fairness_gap.png
    docs/assets/benchmark_overview.png
    docs/assets/privacy_utility_tradeoff.png
    docs/assets/privacy_recovery_fairness_gap.png
    docs/assets/capacity_allocation_precision.png
    docs/assets/capacity_allocation_selection_gap.png
    docs/assets/capacity_allocation_low_signal_share.png
    docs/assets/capacity_sensitivity_frontier.png
    docs/assets/multiseed_capacity_sensitivity.png
    docs/assets/score_matched_calibration.png
    docs/assets/public_reference_calibration.png
    docs/assets/aggregate_noise_sensitivity.png
    docs/assets/cohort_threshold_sensitivity.png
    docs/assets/recovery_feature_ablation.png
    docs/assets/aggregate_alignment_negative_control.png
    docs/assets/missingness_mechanism_sensitivity.png
    docs/assets/disparate_uncertainty_audit.png
    docs/assets/model_sensitivity.png
    docs/assets/pairwise_ranking_sensitivity.png
    docs/assets/underserved_recovery_profile.png
    docs/assets/community_holdout_robustness.png
    docs/assets/heldout_context_shift.png
    docs/multiseed_results.md
    docs/model_sensitivity.md
    docs/aggregate_alignment_negative_control.md
    docs/missingness_mechanism_sensitivity.md
    docs/disparate_uncertainty_audit.md
    docs/heldout_context_shift.md
    docs/validation_report.md
    docs/benchmark_card.md
    outputs/tables/multiseed_privacy_recovery_summary.csv
    outputs/tables/privacy_recovery_metrics.csv
    outputs/tables/capacity_allocation_metrics.csv
    outputs/tables/capacity_sensitivity_metrics.csv
    outputs/tables/multiseed_capacity_sensitivity_raw.csv
    outputs/tables/multiseed_capacity_sensitivity_summary.csv
    outputs/tables/score_matched_calibration_bins.csv
    outputs/tables/score_matched_calibration_summary.csv
    outputs/tables/public_reference_calibration.csv
    outputs/tables/aggregate_noise_sensitivity_raw.csv
    outputs/tables/aggregate_noise_sensitivity_summary.csv
    outputs/tables/cohort_threshold_sensitivity.csv
    outputs/tables/recovery_feature_ablation_raw.csv
    outputs/tables/recovery_feature_ablation_summary.csv
    outputs/tables/aggregate_alignment_negative_control_raw.csv
    outputs/tables/aggregate_alignment_negative_control_summary.csv
    outputs/tables/missingness_mechanism_sensitivity_raw.csv
    outputs/tables/missingness_mechanism_sensitivity_summary.csv
    outputs/tables/disparate_uncertainty_audit_raw.csv
    outputs/tables/disparate_uncertainty_audit_summary.csv
    outputs/tables/disparate_uncertainty_audit_paired.csv
    outputs/tables/model_sensitivity_raw.csv
    outputs/tables/model_sensitivity_summary.csv
    outputs/tables/model_sensitivity_paired_recovery.csv
    outputs/tables/pairwise_ranking_sensitivity_raw.csv
    outputs/tables/pairwise_ranking_sensitivity_summary.csv
    outputs/tables/pairwise_ranking_sensitivity_paired_recovery.csv
    outputs/tables/underserved_recovery_profile_raw.csv
    outputs/tables/underserved_recovery_profile_paired.csv
    outputs/tables/underserved_recovery_profile_summary.csv
    outputs/tables/community_holdout_robustness_raw.csv
    outputs/tables/community_holdout_robustness_paired.csv
    outputs/tables/community_holdout_robustness_summary.csv
    outputs/tables/heldout_context_shift_raw.csv
    outputs/tables/heldout_context_shift_paired.csv
    outputs/tables/heldout_context_shift_summary.csv
    outputs/tables/benchmark_validation_checks.csv

## 4. Run the regression tests

    python -m pytest -q

## 5. Run the complete verification sequence

To run the regression tests, Python compilation check, full benchmark pipeline, and
final validation gate together:

    bash scripts/verify_benchmark.sh

The repository workflow in
[`benchmark-checks.yml`](../.github/workflows/benchmark-checks.yml) runs the same
verification sequence for pull requests and `main` updates. It uploads the generated
validation report so that the completed methodological checks remain inspectable.

## 6. Notes

Generated CSV files under `data/synthetic/` and `outputs/` are intentionally not tracked by Git. The repository tracks source code, documentation, and selected figures used in the README.
