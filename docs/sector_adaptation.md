# Sector Adaptation Roadmap

FairPrivacySignal currently validates a privacy-aware signal-recovery method on a
synthetic public-service outreach benchmark. This roadmap translates the same
system shape into sector-specific public-data pilots. It is a plan for external
validation, not a claim that the method has already been deployed or proven in
these sectors.

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
- aggregate-only substitute
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
| Education | Open University Learning Analytics Dataset (OULAD) | Rank support, advising, or course-resource interventions for students | Prior virtual-learning-environment activity and assessment interactions | Sparse learning-platform activity | Strong first education pilot because the data has student-course interactions and outcomes |
| Public and nonprofit services | ACS PUMS plus public service/resource metadata | Rank benefit, outreach, or resource options for synthetic households | Prior service engagement, simulated because ACS is not an intervention log | Low-observability household or community contexts | Best treated as an expanded synthetic scenario anchored to public population data |
| Financial access | CFPB/FFIEC HMDA mortgage data | Rank review, assistance, or outreach pathways for mortgage applicants | Prior digital interaction or response history, simulated because HMDA is not clickstream data | Thin-context or under-observed applicant segments | Useful for public financial-access framing, but not a direct behavioral-history dataset |
| Marketplaces | MovieLens ratings data | Rank items for users under rating-history loss | Prior ratings, tags, or interactions | New, sparse-history, or niche-preference users/items | Strong first marketplace pilot because recommender signal loss is native to the dataset |

## Recommended Execution Order

1. **MovieLens marketplace pilot.** It is the cleanest public ranking analogue:
   users, items, historical ratings, sparse users, and top-k ranking are already
   native to the data.
2. **OULAD education pilot.** It offers a second domain where behavioral-history
   loss has a natural interpretation: student activity data may be incomplete or
   unavailable, while course and assessment context may remain usable.
3. **Synthea healthcare pilot.** It keeps healthcare non-confidential while
   showing how the method maps to patient outreach and care-navigation tasks.
4. **HMDA financial-access pilot.** Use it as a public financial decisioning
   stress test, but state clearly that the behavioral signal is simulated or
   derived because HMDA is a loan-application dataset, not a product-interaction
   log.
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
- CFPB HMDA data: https://www.consumerfinance.gov/data-research/hmda/
- ACS Public Use Microdata Sample: https://www.census.gov/programs-surveys/acs/microdata.html
- MovieLens datasets: https://grouplens.org/datasets/movielens/
