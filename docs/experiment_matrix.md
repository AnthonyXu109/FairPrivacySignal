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
| Capacity sensitivity sweep | Vary outreach capacity and low-signal allocation-floor strength | Allocation frontier, selection-rate gap, allocated relevance cost |
| Multi-seed capacity sensitivity | Repeat the allocation sweep across synthetic draws | Mean allocation frontier, standard deviation, mean selection-rate gap |

## Current Policy Configuration

| Configuration | Purpose |
|---|---|
| Behavioral-signal scenario rules | Declare consent, sensitive-cohort, and severe-loss suppression flags in readable JSON |
| Privacy-exposure proxy weights | Keep the comparison diagnostic explicit and validate that weights sum to 1.0 |

## Current Diagnostics

| Diagnostic | Purpose |
|---|---|
| Privacy exposure score | Compare remaining individual-level behavioral exposure across scenarios |
| Low-signal NDCG@3 | Measure utility for low-signal households |
| Low-signal gap | Detect unequal ranking outcomes between low-signal and not-low-signal households |
| Multi-seed standard deviation | Check robustness across synthetic data draws |
| Allocated relevance rate | Measure utility after service-capacity constraints are applied |
| Selection-rate gap | Compare low-signal and not-low-signal allocation rates |
| Score-matched subgroup calibration | Compare observed relevance for low-signal and not-low-signal events within shared predicted-score bins |
| Public-reference calibration | Compare selected synthetic community priors with tracked Census QuickFacts anchors without automatic fitting |
| Aggregate-noise sensitivity | Repeat privacy-safe recovery across stress scales and noise realizations while holding the synthetic dataset fixed |
| Cohort-threshold sensitivity | Sweep the minimum cohort size and measure fallback coverage plus utility recovery |
| Recovery feature ablation | Compare no substitutes, engagement aggregates, cohort-context aggregates, and their combination with paired multi-seed deltas |
| Model sensitivity diagnostic | Compare the interpretable logistic primary baseline with lightweight histogram gradient boosting across paired scenarios and seeds |

## Planned Experiments

| Planned experiment | Purpose |
|---|---|
| Expanded public-reference coverage | Add carefully mapped public aggregate anchors while preserving explicit limitations |
| Stronger learning-to-rank model | Compare interpretable logistic baseline with ranking-specific models |
