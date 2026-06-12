# FairPrivacySignal

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20130952.svg)](https://doi.org/10.5281/zenodo.20130952)
[![Benchmark checks](https://github.com/AnthonyXu109/FairPrivacySignal/actions/workflows/benchmark-checks.yml/badge.svg)](https://github.com/AnthonyXu109/FairPrivacySignal/actions/workflows/benchmark-checks.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FairPrivacySignal develops and evaluates a reusable recovery method for AI ranking
systems whose behavioral features are reduced by privacy, consent, or data
minimization rules.

The central question is practical:

> How much ranking utility can be recovered without restoring unavailable raw
> behavioral signals to the serving model?

The repository answers this with **Policy-Aware Signal Recovery**, a two-path
method that combines train-fitted privacy-safe aggregates with cross-fitted signal
reconstruction. The method is evaluated in a synthetic public-service matching
task, but its abstraction applies to healthcare outreach, education programs,
local marketplaces, nonprofit services, and other systems that rank candidates
under restricted behavioral data.

## Main result

![Policy-aware signal recovery results](docs/assets/policy_aware_signal_recovery.png)

Across five paired synthetic-data seeds:

| Signal-loss regime | No recovery NDCG@3 | Policy-aware recovery NDCG@3 | Full-signal gap closed | Low-signal NDCG@3 change |
|---|---:|---:|---:|---:|
| Complete behavioral-signal loss | 0.504 | 0.519 | 31.4% | +0.015 |
| Policy-restricted partial signal | 0.526 | 0.542 | 56.1% | +0.016 |

The privacy-exposure proxy is unchanged relative to the matching signal-loss
baseline. The method recovers part of the lost utility; it does not claim to exceed
the full-signal oracle or eliminate every subgroup gap.

See the full paired results and applicability boundary in
[`docs/policy_aware_signal_recovery.md`](docs/policy_aware_signal_recovery.md).

## Method

Policy-Aware Signal Recovery separates the trusted offline recovery process from
the serving ranker:

1. **Train-only aggregate path.** Historical engagement is converted into
   thresholded, noise-stressed cohort statistics using training households only.
2. **Cross-fitted reconstruction path.** A nonlinear model learns to reconstruct
   the training-only behavioral signal from policy-permitted context and candidate
   features. Household-grouped cross-fitting produces out-of-fold training
   proxies.
3. **Availability-aware substitution.** Observed behavioral values are retained
   when permitted; unavailable values are replaced by reconstructed proxies.
4. **Policy-conditioned fusion.** Complete-loss settings use the more stable
   aggregate path. Partial-loss settings fuse reconstructed and aggregate signals.
5. **Serving without hidden raw signal.** The downstream ranker never receives the
   unavailable individual behavioral value.

![FairPrivacySignal recovery method and evidence system](docs/assets/architecture_diagram.png)

This design is intended for settings where historical signal may be processed
inside a controlled offline training or aggregation environment but cannot be
exposed to the serving model. If policy prohibits use of the signal even during
offline fitting, the reconstruction path is not applicable.

## Why this is more than a model comparison

The repository contributes an executable recovery pipeline rather than only a set
of baselines:

- a policy-conditioned method with explicit deployment assumptions
- household-grouped cross-fitting to prevent test or own-record reconstruction
  leakage
- a same-model ablation against no recovery, a missingness indicator, flat
  aggregates, and reconstruction alone
- paired five-seed estimates of absolute utility, recovered utility, and the
  fraction of the full-signal gap closed
- required validation checks that fail if the primary method stops improving both
  overall and low-signal NDCG@3 without increasing the exposure proxy

## Transferable problem structure

The code operates on a general ranking pattern rather than a company-specific
dataset:

| Abstract role | Public-service example | Other applicable settings |
|---|---|---|
| Decision entity | Household or community | Patient, student, customer, small business |
| Ranked candidate | Outreach service | Intervention, program, course, provider, listing |
| Restricted signal | Historical service engagement | Prior interactions, response history, usage behavior |
| Permitted inputs | Context and candidate attributes | Operational, geographic, institutional, or consented features |
| Output | Ranked service list | Ranked opportunities, resources, or recommendations |

The repository demonstrates methodological portability, not real-world adoption.
Its data, metrics, and implementation are public, synthetic, employer-neutral, and
non-confidential.

## Evidence suite

The benchmark remains important, but it serves the method claim rather than being
the claim itself.

![FairPrivacySignal benchmark overview](docs/assets/benchmark_overview.png)

Key supporting checks include:

- [feature ablation](docs/recovery_feature_ablation.md) and
  [aggregate-alignment negative control](docs/aggregate_alignment_negative_control.md)
- [matched-rate missingness mechanisms](docs/missingness_mechanism_sensitivity.md)
- [model-class](docs/model_sensitivity.md) and
  [ranking-objective](docs/pairwise_ranking_sensitivity.md) sensitivity
- [community-held-out](docs/community_holdout_robustness.md) and
  [heldout context-shift](docs/heldout_context_shift.md) stress tests
- [aggregate-noise](docs/aggregate_noise_sensitivity.md) and
  [cohort-threshold](docs/cohort_threshold_sensitivity.md) sensitivity
- [low-signal fairness diagnostics](docs/fairness_metrics.md) and
  [capacity-constrained allocation](docs/capacity_allocation.md)

The generated [benchmark card](docs/benchmark_card.md),
[reviewer guide](docs/reviewer_guide.md), and
[validation report](docs/validation_report.md) provide compact audit paths.

## Reproduce

FairPrivacySignal runs on an ordinary laptop with NumPy, pandas, scikit-learn, and
Matplotlib.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
bash scripts/verify_benchmark.sh
```

The verification command runs regression tests, compilation checks, the complete
benchmark pipeline, figure generation, and required methodological validations.
See [`docs/reproducibility.md`](docs/reproducibility.md) for output locations and
runtime guidance.

## Scope

All event-level data is synthetic. The repository does not use personal data,
private datasets, proprietary systems, non-public organizational metrics, or
confidential implementation details from any organization.

The method does not provide formal differential privacy, prove protection against
model extraction, establish population representativeness, or demonstrate
deployment impact. These boundaries are documented in
[`docs/limitations.md`](docs/limitations.md) and
[`docs/privacy_design.md`](docs/privacy_design.md).

## Research context

The design is informed by public work on learning with privileged features,
learning-to-rank under unavailable serving features, missingness mechanisms,
privacy-aware aggregation, and fairness diagnostics. The relationship to prior
methods is documented in [`docs/related_work.md`](docs/related_work.md); the
repository does not claim a new state-of-the-art ranking algorithm.

## Citation

FairPrivacySignal v0.1.1 is archived on Zenodo:
[10.5281/zenodo.20130952](https://doi.org/10.5281/zenodo.20130952).

Citation metadata is available in [`CITATION.cff`](CITATION.cff). The repository is
released under the [`MIT License`](LICENSE).
