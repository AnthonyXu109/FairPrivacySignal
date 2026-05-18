# Experiment Matrix

This document summarizes the current and planned FairPrivacySignal experiments.

## Current Experiments

| Experiment | Purpose | Individual behavioral signal | Privacy-safe aggregates | Main metric |
|---|---|---:|---:|---|
| Full signal raw baseline | Upper-bound utility comparison | yes | no | NDCG@3 |
| Severe signal loss | Stress test when all behavioral signal is removed | no | no | NDCG@3 |
| Severe loss + privacy-safe aggregates | Test utility recovery using aggregate substitutes | no | yes | NDCG@3 |
| Policy restricted | Test consent and sensitive-cohort suppression | partial | no | NDCG@3 |
| Policy restricted + privacy-safe aggregates | Test hybrid recovery under policy restrictions | partial | yes | NDCG@3 |

## Current Diagnostics

| Diagnostic | Purpose |
|---|---|
| Privacy exposure score | Compare remaining individual-level behavioral exposure across scenarios |
| Low-signal NDCG@3 | Measure utility for low-signal households |
| Low-signal gap | Detect unequal ranking outcomes between low-signal and not-low-signal households |
| Multi-seed standard deviation | Check robustness across synthetic data draws |

## Planned Experiments

| Planned experiment | Purpose |
|---|---|
| Fairness-aware recovery | Reduce low-signal ranking gaps with sample weighting or group-aware calibration |
| Capacity-aware service ranking | Add public-service capacity constraints to make the scenario more realistic |
| Policy-rule configuration | Move consent/sensitive-cohort suppression rules into a readable configuration file |
| Public aggregate calibration | Calibrate synthetic community distributions with public aggregate datasets |
| Stronger learning-to-rank model | Compare interpretable logistic baseline with ranking-specific models |
