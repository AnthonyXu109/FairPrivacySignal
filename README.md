# FairPrivacySignal

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20130952.svg)](https://doi.org/10.5281/zenodo.20130952)

FairPrivacySignal is a public, non-confidential synthetic-data benchmark for studying privacy-preserving and fairness-aware AI ranking and matching systems under signal loss.

Many AI systems rely on user-level behavioral signals to decide which services, resources, or recommendations should be shown to people or communities. Privacy requirements, consent restrictions, data minimization rules, and reduced access to tracking signals can remove or weaken those signals. While these protections are important, signal loss can also reduce model utility and may disproportionately affect small, low-signal, or underserved participants.

This project demonstrates a reproducible public-service outreach scenario: matching communities or households to relevant public services such as preventive health outreach, food assistance, housing support, job training, and education resources.

The same technical pattern can apply to public agencies, healthcare outreach, nonprofit service delivery, education programs, local marketplaces, and small-business discovery systems.

## What this project demonstrates

- Synthetic public-service ranking and matching data generation
- Privacy-driven signal-loss simulation
- Consent-aware and policy-aware feature suppression
- Cohort aggregation and k-thresholding
- Differential-privacy-style noise for aggregate features
- Contextual and geography-level signals
- Utility metrics such as AUC and NDCG@K
- Fairness metrics for low-signal or underserved participants
- Privacy exposure scoring
- Multi-seed reproducibility checks

## Data and confidentiality

This project uses synthetic data and public aggregate references only. It does not use real personal data, private datasets, proprietary systems, internal business metrics, or confidential implementation details from any organization.

This repository is an educational and research-oriented synthetic benchmark.

## Intended use

FairPrivacySignal is intended to illustrate engineering patterns for evaluating privacy, utility, and fairness tradeoffs in AI ranking and matching systems. It is not a production privacy system and does not provide formal privacy guarantees unless explicitly stated.

## Reproduce the benchmark

FairPrivacySignal includes a one-command benchmark pipeline. From a clean checkout:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    bash scripts/run_benchmark.sh

The full reproducibility guide is available in [`docs/reproducibility.md`](docs/reproducibility.md).

## System architecture

FairPrivacySignal is organized as a reproducible benchmark pipeline: synthetic public-service data generation, privacy-driven signal-loss simulation, policy and consent-based feature suppression, privacy-safe aggregate recovery, ranking evaluation, and fairness diagnostics.

![FairPrivacySignal architecture](docs/assets/architecture_diagram.png)

## Results

FairPrivacySignal demonstrates a privacy-utility-fairness tradeoff in a synthetic public-service outreach setting. The benchmark shows that low-signal households are more concentrated in underserved communities, signal loss can reduce ranking utility, and privacy-safe aggregate/contextual features can partially recover utility while keeping individual behavioral exposure reduced.

### 1. Low-signal households concentrate in underserved communities

![Low signal by underserved bucket](docs/assets/low_signal_by_underserved_bucket.png)

### 2. Privacy-safe aggregate features partially recover ranking utility

![Privacy recovery NDCG](docs/assets/privacy_recovery_ndcg.png)

### 3. Privacy-utility tradeoff across signal-loss scenarios

![Privacy utility tradeoff](docs/assets/privacy_utility_tradeoff.png)

## Multi-seed benchmark results

To make the benchmark more robust, FairPrivacySignal evaluates the privacy-recovery experiment across five synthetic data seeds.

![Multi-seed privacy recovery NDCG](docs/assets/multiseed_privacy_recovery_ndcg.png)

The multi-seed results show that severe signal loss consistently reduces ranking utility, while privacy-safe aggregate and contextual features partially recover NDCG@3 under both severe signal-loss and policy-restricted scenarios.

| Scenario | Privacy exposure | NDCG@3 | Low-signal NDCG@3 | Low-signal gap |
|---|---:|---:|---:|---:|
| Full signal raw baseline | 0.925 ± 0.002 | 0.555 ± 0.011 | 0.490 ± 0.014 | 0.095 ± 0.009 |
| Severe signal loss | 0.475 ± 0.002 | 0.504 ± 0.007 | 0.430 ± 0.014 | 0.108 ± 0.018 |
| Severe loss + privacy-safe aggregates | 0.475 ± 0.002 | 0.520 ± 0.007 | 0.448 ± 0.015 | 0.106 ± 0.018 |
| Policy restricted | 0.728 ± 0.007 | 0.526 ± 0.007 | 0.451 ± 0.008 | 0.109 ± 0.010 |
| Policy restricted + privacy-safe aggregates | 0.728 ± 0.007 | 0.539 ± 0.006 | 0.460 ± 0.007 | 0.115 ± 0.005 |

These results support the project’s utility-recovery claim. The fairness gap remains explicitly reported as a diagnostic rather than presented as solved.

## Fairness diagnostics

FairPrivacySignal also tracks low-signal ranking gaps to ensure that utility recovery does not hide unequal effects on low-signal or underserved populations. This diagnostic is intentionally reported separately from the utility-recovery claim.

![Privacy recovery fairness gap](docs/assets/privacy_recovery_fairness_gap.png)

## Notes on synthetic data

All results are based on synthetic data. The benchmark is designed to illustrate engineering patterns for evaluating privacy, utility, and fairness tradeoffs; it is not intended to model any real community or provide production-grade privacy guarantees.

## Archived release

FairPrivacySignal v0.1.1 has been archived on Zenodo with DOI: [10.5281/zenodo.20130952](https://doi.org/10.5281/zenodo.20130952).
