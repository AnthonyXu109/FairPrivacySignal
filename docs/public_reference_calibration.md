# Public-Reference Calibration Diagnostic

FairPrivacySignal includes a lightweight public-reference comparison for selected
synthetic community-context priors. The purpose is to make synthetic assumptions
more inspectable, not to claim that the benchmark represents a real population.

## Tracked reference snapshot

The benchmark stores two national-level reference anchors in
[`config/public_reference_targets.csv`](../config/public_reference_targets.csv).
Both values come from the [U.S. Census Bureau QuickFacts United States
page](https://www.census.gov/quickfacts/fact/table/US/PST045225) and cover the
2020-2024 period:

| Public reference anchor | QuickFacts value | Mapped synthetic context prior |
|---|---:|---|
| Median household income (in 2024 dollars) | $80,734 | Community-level `median_income` |
| Households with a broadband Internet subscription | 91.0% | Community-level `broadband_access` |

The snapshot records the source URL, source period, retrieval date, and mapping used
by the diagnostic. Keeping this small snapshot in the repository allows the
benchmark pipeline to run offline without requiring an API key.

## Interpretation

The pipeline calculates population-weighted synthetic community averages and
reports their gaps against the tracked reference anchors. The chart intentionally
shows gaps rather than automatically fitting synthetic distributions to the public
values.

These comparisons are directional only:

- the synthetic priors are community-level context variables
- the public values are national household-level summary statistics
- two aggregate anchors cannot establish representativeness
- the comparison does not validate synthetic labels or ranking outcomes

The diagnostic is useful because it makes selected synthetic assumptions explicit,
versioned, and reviewable while preserving the benchmark's controlled synthetic-data
design.
