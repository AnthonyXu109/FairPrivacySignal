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
- Readable policy-rule configuration with validation
- Cohort aggregation and k-thresholding
- Differential-privacy-style noise for aggregate features
- Aggregate-noise sensitivity analysis across reproducible perturbations
- Contextual and geography-level signals
- Utility metrics such as AUC and NDCG@K
- Fairness metrics for low-signal or underserved participants
- Fairness-aware recovery variants for low-signal ranking diagnostics
- Score-matched subgroup calibration diagnostics
- Public-reference calibration diagnostic with a tracked Census QuickFacts snapshot
- Capacity-constrained allocation under limited outreach slots
- Multi-seed allocation sensitivity analysis
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

## Benchmark methodology

For a detailed explanation of the benchmark design, assumptions, signal-loss scenarios, privacy-safe recovery layer, metrics, and limitations, see [`docs/benchmark_design.md`](docs/benchmark_design.md).

The current and planned experiment matrix is summarized in [`docs/experiment_matrix.md`](docs/experiment_matrix.md).

The public research directions that inform the benchmark are summarized in [`docs/related_work.md`](docs/related_work.md).

The auditable signal-suppression configuration is documented in [`docs/policy_rules.md`](docs/policy_rules.md).

The pipeline also writes a machine-checked [`benchmark validation report`](docs/validation_report.md).

The selected synthetic context priors are compared with a tracked public-reference
snapshot in [`docs/public_reference_calibration.md`](docs/public_reference_calibration.md).

## System architecture

FairPrivacySignal is organized as a reproducible benchmark pipeline: synthetic public-service data generation, privacy-driven signal-loss simulation, policy and consent-based feature suppression, privacy-safe aggregate recovery, ranking evaluation, and fairness diagnostics.

![FairPrivacySignal architecture](docs/assets/architecture_diagram.png)

## Results

FairPrivacySignal demonstrates a privacy-utility-fairness tradeoff in a synthetic public-service outreach setting. The benchmark shows that low-signal households are more concentrated in underserved communities, signal loss can reduce ranking utility, and privacy-safe aggregate/contextual features can partially recover utility while keeping individual behavioral exposure reduced.

### Benchmark at a glance

![FairPrivacySignal benchmark overview](docs/assets/benchmark_overview.png)

This overview combines multi-seed recovery results with the capacity-constrained allocation experiment. It shows utility recovery after signal loss, the remaining low-signal ranking gaps, and the explicit tradeoff between allocated relevance and low-signal representation when outreach opportunities are limited.

### 1. Low-signal households concentrate in underserved communities

![Low signal by underserved bucket](docs/assets/low_signal_by_underserved_bucket.png)

### 2. Selected synthetic priors are explicit and inspectable

![Public-reference calibration diagnostic](docs/assets/public_reference_calibration.png)

The benchmark compares selected synthetic context priors with a tracked U.S. Census
Bureau QuickFacts snapshot. The visible gaps are intentional: this is a directional
diagnostic, not an automatic fit or a claim that the synthetic benchmark represents
a real population. See [`docs/public_reference_calibration.md`](docs/public_reference_calibration.md).

### 3. Privacy-safe aggregate features partially recover ranking utility

![Privacy recovery NDCG](docs/assets/privacy_recovery_ndcg.png)

### 4. Aggregate recovery remains inspectable across noise strengths

![Aggregate-noise sensitivity](docs/assets/aggregate_noise_sensitivity.png)

The benchmark varies the DP-style aggregate-noise stress scale and repeats each
setting across three reproducible noise realizations. This exposes whether the
recovery result depends on one favorable perturbation or parameter point. The sweep
is a stress diagnostic, not a formal privacy-budget analysis. See
[`docs/aggregate_noise_sensitivity.md`](docs/aggregate_noise_sensitivity.md).

### 5. Privacy-utility tradeoff across signal-loss scenarios

![Privacy utility tradeoff](docs/assets/privacy_utility_tradeoff.png)

## Multi-seed benchmark results

To make the benchmark more robust, FairPrivacySignal evaluates the privacy-recovery experiment across five synthetic data seeds. Each run also uses the corresponding seed for DP-style aggregate noise, so both the synthetic population and privacy-safe recovery layer vary reproducibly.

![Multi-seed privacy recovery NDCG](docs/assets/multiseed_privacy_recovery_ndcg.png)

The multi-seed results show that severe signal loss consistently reduces ranking utility, while privacy-safe aggregate and contextual features partially recover NDCG@3 under both severe signal-loss and policy-restricted scenarios. The fairness-aware variants produce modest improvements in the low-signal gap under the current synthetic configuration, but do not eliminate the gap.

| Scenario | Privacy exposure | NDCG@3 | Low-signal NDCG@3 | Low-signal gap |
|---|---:|---:|---:|---:|
| Full signal raw baseline | 0.925 ± 0.002 | 0.555 ± 0.011 | 0.490 ± 0.014 | 0.095 ± 0.009 |
| Severe signal loss | 0.475 ± 0.002 | 0.504 ± 0.007 | 0.430 ± 0.014 | 0.108 ± 0.018 |
| Severe loss + privacy-safe aggregates | 0.475 ± 0.002 | 0.520 ± 0.007 | 0.448 ± 0.015 | 0.106 ± 0.018 |
| Severe loss + fairness-aware recovery | 0.475 ± 0.002 | 0.521 ± 0.008 | 0.449 ± 0.017 | 0.104 ± 0.018 |
| Policy restricted | 0.728 ± 0.007 | 0.526 ± 0.007 | 0.451 ± 0.008 | 0.109 ± 0.010 |
| Policy restricted + privacy-safe aggregates | 0.728 ± 0.007 | 0.539 ± 0.006 | 0.460 ± 0.007 | 0.115 ± 0.005 |
| Policy restricted + fairness-aware recovery | 0.728 ± 0.007 | 0.540 ± 0.005 | 0.462 ± 0.004 | 0.113 ± 0.004 |

These results support the project’s utility-recovery claim. The fairness gap remains explicitly reported as a diagnostic rather than presented as solved.

## Fairness diagnostics

FairPrivacySignal also tracks low-signal ranking gaps to ensure that utility recovery does not hide unequal effects on low-signal or underserved populations. This diagnostic is intentionally reported separately from the utility-recovery claim.

![Multi-seed privacy recovery fairness gap](docs/assets/multiseed_fairness_gap.png)

### Score-matched subgroup calibration

![Score-matched subgroup calibration](docs/assets/score_matched_calibration.png)

Aggregate metrics can conceal differences among similarly scored candidates. This diagnostic places test-set events into shared predicted-score bins and compares observed relevance for low-signal and not-low-signal groups within those bins. It is a lightweight, seed-42 diagnostic rather than a formal fairness guarantee. See [`docs/fairness_metrics.md`](docs/fairness_metrics.md) for definitions and limitations.

## Capacity-constrained allocation

Ranking quality is only part of the problem when outreach slots, appointment capacity, staff time, or funding are limited. FairPrivacySignal includes a capacity-constrained allocation experiment that compares utility-only allocation with a fairness-constrained policy that reserves a minimum share of outreach capacity for low-signal households.

![Capacity-constrained allocation relevance rate](docs/assets/capacity_allocation_precision.png)

The allocation experiment makes the utility-fairness tradeoff visible rather than assuming one objective automatically improves the other. See [`docs/capacity_allocation.md`](docs/capacity_allocation.md) for the full setup, metrics, and figures.

### Capacity sensitivity frontier

![Multi-seed capacity sensitivity frontier](docs/assets/multiseed_capacity_sensitivity.png)

The sensitivity analysis sweeps multiple outreach-capacity levels and low-signal allocation-floor strengths across five synthetic seeds. The frontier and heatmaps show that a fairness constraint should be evaluated as a policy choice with measurable tradeoffs, not as a single fixed switch. The single-seed supporting figure remains available in [`docs/capacity_allocation.md`](docs/capacity_allocation.md).

## Notes on synthetic data

All results are based on synthetic data. The benchmark is designed to illustrate engineering patterns for evaluating privacy, utility, and fairness tradeoffs; it is not intended to model any real community or provide production-grade privacy guarantees.

## Archived release

FairPrivacySignal v0.1.1 has been archived on Zenodo with DOI: [10.5281/zenodo.20130952](https://doi.org/10.5281/zenodo.20130952).
