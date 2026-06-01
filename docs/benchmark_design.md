# FairPrivacySignal Benchmark Design

FairPrivacySignal is designed as a synthetic benchmark for studying privacy, utility, and fairness tradeoffs in AI ranking and matching systems under signal loss.

The goal is not to reproduce any real production system. The goal is to provide a transparent, reproducible, non-confidential benchmark that lets reviewers inspect how privacy constraints affect downstream ranking quality and low-signal groups.

## 1. Problem Formulation

Many ranking and matching systems rely on individual-level behavioral signals. These signals can improve model utility, but they may also create privacy exposure, especially when they are linkable, persistent, or sensitive.

FairPrivacySignal studies the following question:

> When individual-level behavioral signals are reduced or suppressed by privacy, consent, or policy constraints, can privacy-safe aggregate and contextual features recover part of the lost ranking utility while keeping raw behavioral exposure low?

The benchmark evaluates three linked dimensions:

- **Utility:** whether the system can rank relevant services accurately.
- **Privacy exposure:** how much individual-level behavioral signal remains available.
- **Fairness diagnostics:** whether low-signal groups experience worse ranking outcomes.

## 2. Synthetic Public-Service Outreach Scenario

The benchmark uses a synthetic public-service outreach task. A ranking system recommends relevant public services to synthetic households and communities.

Example service categories include:

- food assistance
- preventive health outreach
- housing support
- job training
- education support
- transportation support

This scenario is intentionally broader than advertising. The same technical pattern appears in public benefits outreach, healthcare resource matching, education program recommendation, nonprofit service delivery, local marketplace discovery, and other privacy-sensitive ranking systems.

## 3. Why Synthetic Data

The benchmark uses synthetic data for three reasons:

1. **Privacy:** no real person, household, community, user, advertiser, or organization is modeled or identified.
2. **Transparency:** every assumption is visible in code and can be inspected or modified.
3. **Controlled experiments:** signal loss, policy restrictions, and privacy-safe recovery can be tested under repeatable conditions.

The benchmark is not intended to claim that the synthetic population represents any real community. It is a controlled engineering testbed.

## 4. Data Generating Process

The synthetic generator creates four conceptual layers:

### 4.1 Communities

Synthetic communities include contextual attributes such as:

- urbanicity
- population
- median income
- unemployment rate
- broadband access
- food access risk
- health need score
- housing pressure
- underserved score

### 4.2 Households

Synthetic households are assigned to communities and include attributes such as:

- age group
- income band
- consent state
- internet access
- language access need
- disability proxy
- sensitive-cohort proxy
- low-signal status
- historical engagement count

### 4.3 Services

Synthetic services represent public-service categories with different targeting and capacity characteristics.

### 4.4 Household-Service Events

Each household-service pair receives a synthetic relevance label. Relevance is generated from a combination of:

- service-specific need
- contextual community characteristics
- household-level attributes
- service-specific historical engagement
- low-signal penalties
- random noise

Service-specific engagement is intentionally important because it allows signal loss to affect within-household ranking order, not merely global classification quality.

## 5. Public-Reference Calibration Diagnostic

The benchmark compares selected synthetic community-context priors against a
tracked U.S. Census Bureau QuickFacts snapshot. The current diagnostic reports
population-weighted synthetic averages for median income and broadband access
alongside national public-reference anchors.

This comparison exposes synthetic-prior gaps without automatically fitting the
generator to public values. It is directional only: community-level synthetic
context variables are not equivalent to national household-level statistics, and
the comparison does not establish representativeness. See
[`docs/public_reference_calibration.md`](public_reference_calibration.md).

## 6. Signal-Loss Scenarios

FairPrivacySignal evaluates multiple signal availability scenarios.

| Scenario | Description | Behavioral signal availability |
|---|---|---:|
| Full signal raw baseline | All synthetic individual-level behavioral features are available | high |
| Severe signal loss | Individual behavioral history is removed for all households | none |
| Consent restricted | Behavioral history is removed when consent is false | partial |
| Policy restricted | Behavioral history is removed for non-consented or sensitive-cohort households | partial |
| Privacy-safe recovery | Aggregate/contextual substitutes are added without restoring raw individual behavioral history | low to medium |

The executable scenario flags and privacy-exposure weights are stored in
[`config/policy_rules.json`](../config/policy_rules.json). See
[`docs/policy_rules.md`](policy_rules.md) for the readable rule table and validation
behavior.

## 7. Privacy-Safe Recovery Design

The privacy-safe recovery layer adds non-raw aggregate and contextual signals:

- cohort aggregation
- minimum cohort-size thresholds
- service-level fallback aggregates
- contextual community features
- DP-style noise on aggregate statistics

This layer is intentionally simple and inspectable. It is not a claim of production-grade differential privacy. It is a benchmark mechanism for studying how aggregate substitutes can recover useful signal after raw behavioral features are suppressed.

### 7.1 Fairness-Aware Recovery Variant

The fairness-aware variant trains a low-signal-specific model after privacy-safe aggregate recovery. Relevant low-signal examples receive additional training weight, and the low-signal-specific predictions are blended with predictions from the global model.

Extra positive weighting shifts the probability scale of a logistic model. Before blending, the implementation explicitly reverses that odds shift. This keeps the score scale auditable and avoids presenting a ranking-only improvement while silently degrading overall classification diagnostics.

This variant is an experimental baseline, not a claim that fairness gaps are solved. The benchmark reports its utility and low-signal metrics alongside the simpler privacy-safe aggregate baseline.

## 8. Ranking Model

The current benchmark uses an interpretable baseline model rather than a complex neural model.

This is intentional:

- it makes results easier to audit
- it avoids hiding the benchmark mechanism behind model complexity
- it allows reviewers to inspect whether the signal-loss and recovery effects are plausible
- it keeps the project runnable on ordinary laptops

Future versions can add gradient-boosted trees, learning-to-rank objectives, or neural ranking models, but the v0.x baseline prioritizes transparency.

## 9. Evaluation Metrics

FairPrivacySignal reports three categories of metrics.

### 9.1 Utility

- AUC
- NDCG@3

NDCG@3 is emphasized because the benchmark is a ranking task: the system must recommend the most relevant services near the top of the list.

### 9.2 Privacy Exposure

The benchmark uses an interpretable privacy exposure score. Higher scores indicate more individual-level behavioral signal remains available to the model.

This is not a formal privacy guarantee. It is a diagnostic proxy used to compare scenarios.

### 9.3 Low-Signal Fairness Diagnostics

The benchmark tracks low-signal NDCG@3 and the NDCG gap between not-low-signal and low-signal households.

The project does not claim that the current privacy-safe recovery layer solves fairness. Instead, it reports fairness gaps explicitly so that utility recovery does not hide unequal downstream effects.

### 9.4 Score-Matched Subgroup Calibration

The benchmark also places test-set events into shared predicted-score bins and
compares observed relevance rates for low-signal and not-low-signal groups within
those bins. It reports subgroup expected calibration error (ECE) and a weighted mean
absolute matched relevance gap.

This is a lightweight diagnostic inspired by ranking-calibration research. It does
not implement a formal matched-pair estimator or prove that a ranking policy is fair.

## 10. Multi-Seed Evaluation

The benchmark runs the privacy-recovery experiment across multiple synthetic seeds.

Each synthetic-data seed is also forwarded to the aggregate noise generator. This varies both the generated population and the DP-style aggregate noise while keeping each run reproducible.

This reduces the risk that results are driven by one favorable random draw and makes the benchmark more credible as an experimental artifact.

The current multi-seed result shows:

- severe signal loss consistently reduces ranking utility
- privacy-safe aggregate/contextual features partially recover ranking utility
- policy-restricted + privacy-safe recovery improves utility over the policy-restricted baseline
- fairness-aware variants produce modest low-signal gap improvements under the current synthetic configuration
- fairness gaps remain diagnostic and require further evaluation

## 11. What the Current Benchmark Shows

The current version supports a modest, evidence-aligned claim:

> Privacy-driven signal loss can reduce ranking utility in synthetic public-service matching. Privacy-safe aggregate and contextual features can partially recover utility while keeping individual behavioral exposure reduced. Low-signal fairness gaps should be separately measured rather than assumed to improve automatically.

## 12. What the Current Benchmark Does Not Claim

FairPrivacySignal does not claim that:

- it models any real community
- it provides production-grade privacy guarantees
- it proves fairness gaps are solved
- it reproduces any proprietary system
- it should be used directly for public-service eligibility decisions
- synthetic results alone prove real-world effectiveness or adoption

## 13. Planned Extensions

Near-term planned extensions include:

1. comparing the interpretable baseline with a ranking-specific model
2. expanding public-reference coverage while keeping mappings explicit
3. improving the technical whitepaper for independent expert review

## 14. Why This Matters

Privacy regulations and data minimization practices can reduce access to individual-level behavioral signals. This is often necessary for protecting users, but it can also degrade ranking quality and disproportionately affect low-signal groups.

A benchmark like FairPrivacySignal helps make these tradeoffs visible. It gives researchers, engineers, public-sector technologists, and reviewers a non-confidential way to reason about privacy-preserving ranking systems before deploying them in sensitive settings.

## 15. Capacity-Constrained Allocation Extension

FairPrivacySignal also includes a capacity-constrained allocation experiment. This extension models a practical public-service outreach constraint: even if a ranking system can score every household-service pair, real service providers often have limited outreach slots, appointment capacity, staff time, or funding.

The allocation experiment compares utility-only allocation with fairness-constrained allocation. Utility-only allocation selects the highest-scoring candidates, while fairness-constrained allocation reserves a minimum share of outreach capacity for low-signal households. This exposes a realistic utility-fairness tradeoff: improving low-signal representation can reduce selection-rate gaps, but may also reduce allocated relevance rate.

The benchmark repeats the allocation sensitivity sweep across five synthetic draws.
It reports mean tradeoffs and standard deviation whiskers so that the frontier is not
supported by one favorable seed.

See [`docs/capacity_allocation.md`](capacity_allocation.md) for details.

## 16. Benchmark Validation Gate

The one-command pipeline ends with machine-checked methodological invariants. These
checks cover signal-loss scenario completeness, privacy-exposure monotonicity,
bounded metrics, valid allocation counts, score-matched calibration coverage,
documented public-reference targets, and multi-seed completeness.

Required checks fail the pipeline when an invariant drifts. Informational checks
record current result behavior without blocking future experimentation. The current
report is available in [`docs/validation_report.md`](validation_report.md).
