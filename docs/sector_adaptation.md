# Sector Adaptation Roadmap

FairPrivacySignal validates a privacy-aware signal-recovery method on a synthetic
public-service outreach benchmark and extends the same system shape into
sector-specific public-data pilots. This roadmap tracks implemented and planned
external validations, not claims that the method has already been deployed or
proven in these sectors.

## Transfer Pattern

Each sector pilot should answer the same five questions:

1. What entity is being ranked or matched?
2. Which historical signal may become restricted by privacy, consent, retention,
   or data-minimization rules?
3. Which contextual features remain permitted at ranking time?
4. Does Policy-Aware Signal Recovery improve the matching signal-loss baseline
   without increasing the exposure proxy?
5. Are low-signal or under-observed participants helped, unchanged, or harmed?

The pilot should report the same evidence surface used in the main benchmark:

- full-signal oracle
- signal-loss baseline
- train-fitted or aggregate substitute
- policy-aware recovery
- overall ranking utility
- low-signal utility
- full-signal gap closed
- exposure proxy
- allocation or threshold sensitivity when decisions are capacity-limited

## Candidate Public-Data Pilots

| Sector | Public data candidate | Ranking or matching task | Restricted signal analogue | Low-signal group | Status |
|---|---|---|---|---|---|
| Healthcare outreach | Synthea synthetic EHR records | Rank care-gap outreach, screening, or follow-up options for synthetic patients | Prior encounters, portal response, medication or care-plan history | Sparse visit or engagement history | Strong first healthcare pilot because records are synthetic and non-confidential |
| Education | UCI Student Performance | Rank student-course records for support triage | Prior grades from earlier periods | Low-absence records with weaker administrative warning signals | Implemented as the second external public-data pilot |
| Public and nonprofit services | ACS PUMS plus public service/resource metadata | Rank benefit, outreach, or resource options for synthetic households | Prior service engagement, simulated because ACS is not an intervention log | Low-observability household or community contexts | Best treated as an expanded synthetic scenario anchored to public population data |
| Financial access | UCI German Credit | Rank credit applications for review or assistance triage | Checking-account status, credit-history status, and savings status | Thin-file applicants with one or fewer existing credits | Implemented as the third external public-data pilot |
| Marketplaces | MovieLens ratings data | Rank items for users under rating-history loss | Prior ratings, tags, or interactions | New, sparse-history, or niche-preference users/items | Implemented as the first external public-data pilot |

## Recommended Execution Order

1. **MovieLens marketplace pilot.** Implemented in
   [MovieLens marketplace validation](movielens_marketplace_validation.md). It is
   the cleanest public ranking analogue: users, items, historical ratings, sparse
   users, and top-k ranking are already native to the data.
2. **UCI Student Performance education pilot.** Implemented in
   [education student performance validation](education_student_performance_validation.md).
   UCI was selected before OULAD because it has a stable public download and a
   clear prior-grade signal-loss setup; OULAD remains useful for a future richer
   learning-activity pilot.
3. **Synthea healthcare pilot.** It keeps healthcare non-confidential while
   showing how the method maps to patient outreach and care-navigation tasks.
4. **UCI German Credit financial-access pilot.** Implemented in
   [finance German Credit validation](finance_german_credit_validation.md). It is
   a compact public credit dataset with a clear historical financial-signal loss
   setup. HMDA remains useful for a future policy-facing stress test, but its
   behavioral signal would need to be simulated or derived.
5. **ACS-anchored public-services scenario.** Extend the current synthetic
   public-service benchmark by calibrating household and community priors to ACS
   PUMS, while retaining synthetic labels and simulated service histories.

## Figure Plan

A sector extension should add one compact figure, not a wall of charts:

- panel 1: full-signal gap closed by sector
- panel 2: low-signal NDCG/AUC change by sector
- panel 3: exposure proxy compared with the signal-loss baseline
- panel 4: capacity or threshold frontier for domains with limited slots

The figure should separate "validated on this public dataset" from "adaptation
design only" so the repository does not overclaim.

## Acceptance Criteria

A sector pilot is ready to include in the main README only when it has:

- a scripted adapter under `fairprivacysignal/` or `scripts/`
- a deterministic small-data or documented-download path
- a generated table under `outputs/tables/`
- a generated figure under `docs/assets/`
- a short report under `docs/`
- a validation check that compares recovery with the matching signal-loss baseline
- an explicit limitation section naming what the public data cannot prove

## Public Data References

- Synthea synthetic patient generator: https://github.com/synthetichealth/synthea
- Open University Learning Analytics Dataset overview: https://analyse.kmi.open.ac.uk/open_dataset
- UCI Student Performance dataset: https://archive.ics.uci.edu/dataset/320/student+performance
- UCI German Credit dataset: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- CFPB HMDA data: https://www.consumerfinance.gov/data-research/hmda/
- ACS Public Use Microdata Sample: https://www.census.gov/programs-surveys/acs/microdata.html
- MovieLens datasets: https://grouplens.org/datasets/movielens/
