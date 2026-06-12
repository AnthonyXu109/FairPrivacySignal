# Experiment Matrix

This document summarizes the current and planned FairPrivacySignal experiments.

## Current Experiments

### Primary recovery method

| Experiment | Purpose | Serving-time behavioral signal | Recovery input | Main metric |
|---|---|---:|---|---|
| Policy-aware recovery under complete loss | Use the stable train-fitted aggregate path when no event-level signal remains | no | Thresholded, noise-stressed aggregates | NDCG@3 and full-signal gap closed |
| Policy-aware recovery under partial restriction | Preserve permitted values and reconstruct unavailable values before aggregate fusion | partial | Cross-fitted reconstruction plus aggregates | NDCG@3 and full-signal gap closed |
| Missingness-indicator comparator | Test whether explicitly marking unavailable values explains the gain | partial or none | Availability indicator only | NDCG@3 |
| Reconstruction-only ablation | Isolate the contribution of cross-fitted feature reconstruction | reconstructed | Permitted context and candidate attributes | NDCG@3 |

The primary method uses five household-grouped folds for reconstruction and five
paired synthetic-data seeds for final evaluation.

### Supporting baselines

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
| Train-fitted aggregate preprocessing | Learn cohort statistics and service-level fallbacks from training households before holdout scoring |

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
| Public-reference calibration | Compare selected synthetic community priors with tracked Census Bureau anchors without automatic fitting |
| Aggregate-noise sensitivity | Repeat privacy-safe recovery across stress scales and noise realizations while holding the synthetic dataset fixed |
| Cohort-threshold sensitivity | Sweep the minimum cohort size and measure fallback coverage plus utility recovery |
| Recovery feature ablation | Compare no substitutes, engagement aggregates, cohort-context aggregates, and their combination with paired multi-seed deltas |
| Aggregate-alignment negative control | Preserve train-only aggregate construction while cyclically permuting reference service categories to test whether recovery depends on service-aligned structure |
| Model sensitivity diagnostic | Compare the interpretable logistic primary baseline with lightweight histogram gradient boosting across paired scenarios and seeds |
| Ranking-objective sensitivity | Compare pointwise logistic training with lightweight linear pairwise and listwise rankers across paired scenarios and seeds |
| Underserved quartile recovery profile | Check whether pooled recovery hides quartile-specific regressions across synthetic community contexts |
| Community-held-out robustness diagnostic | Compare household-level holdout with a stricter paired split that keeps training and evaluation communities disjoint |
| Heldout context-shift stress test | Move bounded synthetic context covariates and deterministic context buckets on the evaluation side while keeping training households and labels fixed |
| Matched-rate missingness-mechanism sensitivity | Hold overall behavioral availability fixed while comparing uniform-random, observed-context, and signal-dependent incidence |
| Disparate-uncertainty and ranking-stability audit | Use household-bootstrap ensembles to compare score variability and Top-3 membership stability by signal group |

## Planned Experiments

| Planned experiment | Purpose |
|---|---|
| Public-reference uncertainty metadata | Add source-universe and uncertainty details while preserving explicit limitations |
| Broader drift mechanisms | Extend the controlled covariate proxy with additional documented shift families |
