# Experiment Matrix

This document summarizes the current and planned FairPrivacySignal experiments.

## Current Experiments

| Experiment | Purpose | Individual behavioral signal | Privacy-safe aggregates | Main metric |
|---|---|---:|---:|---|
| Full signal raw baseline | Upper-bound utility comparison | yes | no | NDCG@3 |
| Severe signal loss | Stress test when all behavioral signal is removed | no | no | NDCG@3 |
| Severe loss + privacy-safe aggregates | Test utility recovery using aggregate substitutes | no | yes | NDCG@3 |
| Severe loss + fairness-aware recovery | Blend global and low-signal-specific predictions after aggregate recovery | no | yes | Low-signal NDCG@3 gap |
| Policy restricted | Test consent and sensitive-cohort suppression | partial | no | NDCG@3 |
| Policy restricted + privacy-safe aggregates | Test hybrid recovery under policy restrictions | partial | yes | NDCG@3 |
| Policy restricted + fairness-aware recovery | Test low-signal recovery while retaining policy-permitted signals | partial | yes | Low-signal NDCG@3 gap |

## Current Allocation Extension

| Experiment | Purpose | Main metrics |
|---|---|---|
| Utility-only capacity allocation | Select the highest-scoring candidates within limited service capacity | Allocated relevance rate, low-signal selection rate |
| Fairness-constrained capacity allocation | Reserve a minimum share of outreach capacity for low-signal households | Allocated relevance rate, selection-rate gap, allocated low-signal share |

## Current Diagnostics

| Diagnostic | Purpose |
|---|---|
| Privacy exposure score | Compare remaining individual-level behavioral exposure across scenarios |
| Low-signal NDCG@3 | Measure utility for low-signal households |
| Low-signal gap | Detect unequal ranking outcomes between low-signal and not-low-signal households |
| Multi-seed standard deviation | Check robustness across synthetic data draws |
| Allocated relevance rate | Measure utility after service-capacity constraints are applied |
| Selection-rate gap | Compare low-signal and not-low-signal allocation rates |

## Planned Experiments

| Planned experiment | Purpose |
|---|---|
| Multi-seed capacity sensitivity | Test whether allocation tradeoffs persist across random draws and capacity levels |
| Subgroup calibration | Compare predicted and observed relevance across low-signal and not-low-signal groups |
| Policy-rule configuration | Move consent/sensitive-cohort suppression rules into a readable configuration file |
| Public aggregate calibration | Calibrate synthetic community distributions with public aggregate datasets |
| Stronger learning-to-rank model | Compare interpretable logistic baseline with ranking-specific models |
