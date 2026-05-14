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
- multi-seed benchmark results
- architecture diagram

## 3. Key result files

    docs/assets/multiseed_privacy_recovery_ndcg.png
    docs/assets/privacy_utility_tradeoff.png
    docs/assets/privacy_recovery_fairness_gap.png
    docs/multiseed_results.md
    outputs/tables/multiseed_privacy_recovery_summary.csv
    outputs/tables/privacy_recovery_metrics.csv

## 4. Notes

Generated CSV files under `data/synthetic/` and `outputs/` are intentionally not tracked by Git. The repository tracks source code, documentation, and selected figures used in the README.
