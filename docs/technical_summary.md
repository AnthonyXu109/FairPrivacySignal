# FairPrivacySignal Technical Summary

**Project:** FairPrivacySignal  
**Public repository:** https://github.com/AnthonyXu109/FairPrivacySignal  
**Archived DOI:** https://doi.org/10.5281/zenodo.20130952  
**Version:** v0.1.1  

## 1. Project Purpose

FairPrivacySignal is a public, non-confidential synthetic-data benchmark for evaluating privacy, utility, and fairness tradeoffs in AI ranking and matching systems under signal loss.

The project studies a common challenge in modern AI systems: ranking and matching models often rely on individual-level behavioral signals, but privacy requirements, consent restrictions, data minimization rules, and policy constraints may reduce access to those signals. When signal loss occurs, model utility can decline, and low-signal or underserved participants may be disproportionately affected.

FairPrivacySignal uses a synthetic public-service outreach scenario to demonstrate how privacy-safe aggregate and contextual features can partially recover ranking utility while reducing exposure to individual-level behavioral data.

## 2. Synthetic Public-Service Scenario

The benchmark models a public-service outreach system that ranks relevant services for synthetic households and communities. Example services include:

- food assistance
- preventive health outreach
- housing support
- job training
- education support
- transportation support

The benchmark does not model or identify any real person, household, community, or organization. All event-level data is synthetic.

## 3. Core Technical Components

FairPrivacySignal currently includes:

1. **Synthetic data generation**  
   Generates synthetic households, communities, services, and household-service relevance labels.

2. **Signal-loss simulation**  
   Simulates full-signal, severe signal-loss, consent-restricted, and policy-restricted scenarios.

3. **Privacy-safe feature transformation**  
   Implements cohort aggregation, minimum cohort thresholds, contextual features, and DP-style noise for aggregate signals.

4. **Ranking evaluation**  
   Trains baseline ranking models and evaluates utility using AUC and NDCG@3.

5. **Privacy and fairness diagnostics**  
   Tracks average privacy exposure score and low-signal ranking gaps.

6. **Fairness-aware recovery evaluation**
   Blends global and low-signal-specific predictions to measure whether explicit recovery strategies reduce low-signal ranking gaps.

7. **Capacity-constrained allocation**
   Compares utility-only and fairness-constrained allocation when outreach opportunities are limited.

8. **Multi-seed reproducibility checks**
   Repeats privacy-recovery evaluation across synthetic-data and DP-style aggregate-noise seeds.

9. **Score-matched subgroup calibration**
   Compares observed relevance rates for low-signal and not-low-signal events within shared predicted-score bins.

10. **Multi-seed allocation sensitivity**
    Repeats the allocation-floor sweep across synthetic draws and reports mean tradeoffs with standard deviations.

11. **Readable policy-rule configuration**
    Loads validated signal-suppression flags and privacy-exposure weights from a small JSON file.

12. **Benchmark validation gate**
    Enforces stable methodological invariants and writes a reviewer-readable validation report.

13. **Public-reference calibration diagnostic**
    Compares selected synthetic community priors with tracked Census Bureau anchors
    while reporting gaps rather than automatically fitting the generator.

14. **Aggregate-noise sensitivity analysis**
    Repeats privacy-safe recovery across DP-style noise stress scales and reproducible
    perturbations while holding the synthetic dataset fixed.

15. **Continuous benchmark verification**
    Runs regression tests, Python compilation checks, the reproducible benchmark
    pipeline, and the methodological validation gate for repository updates.

16. **Cohort-threshold sensitivity analysis**
    Sweeps the k-threshold for cohort aggregates and reports fallback coverage
    alongside overall and low-signal utility recovery.

17. **Generated benchmark card**
    Summarizes experimental scale, coverage, key results, sensitivity checkpoints,
    validation status, and evidence links from auditable pipeline outputs.

18. **Recovery feature ablation**
    Separates engagement aggregates from cohort-context aggregates and reports
    paired multi-seed recovery deltas against no-aggregate baselines.

19. **Model sensitivity diagnostic**
    Compares the interpretable logistic primary baseline with lightweight histogram
    gradient boosting across paired signal-loss and recovery scenarios.

20. **Underserved quartile recovery profile**
    Groups distinct synthetic communities by underserved score and reports paired
    aggregate-minus-baseline recovery across five synthetic draws.

21. **Train-fitted aggregate preprocessing**
    Learns cohort statistics and service-level fallbacks from training households
    before applying the same mapping to holdout events.

22. **Community-held-out robustness diagnostic**
    Compares the primary household-level holdout with a stricter paired synthetic
    split that keeps training and evaluation communities disjoint.

23. **Ranking-objective sensitivity diagnostic**
    Compares the pointwise logistic primary baseline with lightweight linear
    pairwise and listwise rankers trained on ordered pairs or complete service
    lists within synthetic households.

## 4. Current Benchmark Results

The current benchmark results show:

- Full-signal ranking achieved the highest utility.
- Severe signal loss reduced ranking utility.
- Privacy-safe aggregate features partially recovered utility under severe signal loss.
- A hybrid policy-restricted setting using available policy-permitted signals plus privacy-safe aggregate features improved ranking utility over the policy-restricted baseline.
- Fairness-aware recovery variants produced mixed low-signal gap effects across the
  current synthetic scenarios, without eliminating the gap.
- A paired feature ablation showed that the engagement aggregate provides most of
  the observed recovery in the current synthetic configuration, while combined
  cohort-context features add a modest increment under severe signal loss.
- A lightweight model comparison showed that aggregate recovery is not
  model-invariant: it is clear for the logistic primary baseline but smaller or
  absent for histogram gradient boosting under the current synthetic configuration.
- A paired underserved-quartile profile showed that positive pooled recovery can
  coexist with uneven low-signal effects across synthetic community contexts,
  including negative quartile-specific deltas.
- A paired community-held-out diagnostic showed that aggregate recovery remained
  visible when training and evaluation communities were disjoint in a stricter
  synthetic stress test.
- A paired ranking-objective diagnostic reports aggregate recovery separately for
  pointwise logistic, linear pairwise, and linear listwise training.
- Capacity-constrained allocation exposed a measurable tradeoff between allocated relevance and low-signal representation.
- Low-signal fairness gaps are explicitly tracked as a diagnostic to prevent utility recovery from hiding unequal effects on low-signal or underserved populations.

These results support the project’s core hypothesis: privacy-preserving AI systems should be evaluated not only for privacy exposure reduction, but also for downstream utility and fairness impact.

## 5. Privacy and Confidentiality

FairPrivacySignal does not use real personal data, private datasets, proprietary systems, internal business metrics, confidential architecture, or confidential implementation details from any organization.

The project is designed as an educational and research-oriented synthetic benchmark. It does not provide production-grade privacy guarantees, and its DP-style noise mechanism is included only as a simplified demonstration of aggregate privacy-preserving feature design.

## 6. Broader Relevance

Although the benchmark uses public-service outreach as its demonstration setting, the same technical pattern is relevant to many ranking and matching systems, including:

- public benefits outreach
- healthcare resource matching
- education program recommendation
- nonprofit service delivery
- local marketplace discovery
- small-business access systems
- other privacy-sensitive AI decision pipelines

The broader goal is to make privacy-preserving and fairness-aware ranking evaluation more transparent, reproducible, and accessible.

## 7. Future Work

Planned improvements include:

- expanding synthetic scenarios beyond public-service outreach
- adding public-reference uncertainty metadata with explicit synthetic-to-public mappings
- improving documentation for independent technical review
- collecting external expert feedback
