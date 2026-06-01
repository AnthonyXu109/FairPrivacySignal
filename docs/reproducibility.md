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
- fairness-aware recovery diagnostics
- multi-seed benchmark results
- capacity-constrained allocation metrics
- capacity-sensitivity frontier metrics
- multi-seed capacity-sensitivity metrics
- score-matched subgroup calibration metrics
- public-reference calibration metrics from a tracked QuickFacts snapshot
- aggregate-noise sensitivity metrics across reproducible perturbations
- benchmark overview figure
- architecture diagram
- benchmark validation report

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
    docs/multiseed_results.md
    docs/validation_report.md
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
