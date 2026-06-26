# Public Services Adult Census Validation

This public-data pilot adapts the FairPrivacySignal signal-loss pattern to a
public-service outreach setting using the UCI Adult/Census Income dataset. Adults
are ranked for low-income support outreach; detailed employment and economic
fields are treated as signals that may be unavailable under data minimization,
while coarse demographic and household context remains available.

The raw UCI files are downloaded at runtime and are not redistributed in this repository.

![Public-services public-data validation](assets/public_services_adult_validation.svg)

## Task

- **Ranked candidate:** adults for public or nonprofit support outreach
- **Restricted economic signal:** education, occupation, workclass, hours, and capital-gain/loss fields
- **Permitted context:** age, marital status, relationship, race, sex, and native country
- **Low-signal group:** adults below the median permitted-context score
- **Metric:** NDCG@1000, with binary relevance defined as income `<=50K`

## Results

| method | overall_ndcg_at_1000 | low_signal_ndcg_at_1000 | full_signal_gap_closed | economic_signal_exposure |
| --- | --- | --- | --- | --- |
| Full detailed-economic signal | 0.992 | 0.926 | 100.0% | 1.000 |
| Context-only baseline | 0.923 | 0.772 | 0.0% | 0.000 |
| Train-fitted signal recovery | 0.950 | 0.855 | 38.7% | 0.000 |
| Policy-aware partial recovery | 0.992 | 0.855 | 100.0% | 0.500 |

The train-fitted recovery path closes 38.7%
of the full-signal NDCG@1000 gap without exposing the restricted detailed
economic features at scoring time. The policy-aware partial path keeps detailed
economic signal for higher-signal records while substituting recovered signal for
low-signal records, closing 100.0%
of the same gap in this pilot.

## Interpretation

This is an external public-data validation of the system shape, not a deployed
benefits or nonprofit-service model. It shows how the method can be instantiated
in a census-like public-services workflow: define detailed economic signals,
suppress them at scoring time, substitute a train-fitted reconstruction with a
cohort stabilizer, and measure outreach-ranking recovery. The dataset is a public
income benchmark rather than a service-interaction log, so the availability policy
is simulated for evaluation.
