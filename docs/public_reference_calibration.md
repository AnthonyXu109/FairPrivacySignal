# Public-Reference Calibration Diagnostic

FairPrivacySignal includes a lightweight public-reference comparison for selected
synthetic community-context priors. The purpose is to make synthetic assumptions
more inspectable, not to claim that the benchmark represents a real population.

## Tracked reference snapshot

The benchmark stores three national-level reference anchors in
[`config/public_reference_targets.csv`](../config/public_reference_targets.csv).
Income and broadband values come from the [U.S. Census Bureau QuickFacts United
States page](https://www.census.gov/quickfacts/fact/table/US/PST045225). The
unemployment-rate value comes from the Census Bureau's [2024 ACS 5-Year Data
Profile DP03](https://data.census.gov/table/ACSDP5Y2024.DP03?g=010XX00US). Each
anchor covers the 2020-2024 period:

| Public reference anchor | Census Bureau value | Mapped synthetic context prior |
|---|---:|---|
| Median household income (in 2024 dollars) | $80,734 | Community-level `median_income` |
| Households with a broadband Internet subscription | 91.0% | Community-level `broadband_access` |
| Civilian labor force unemployment rate (`DP03_0009PE`) | 5.2% | Community-level `unemployment_rate` |

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
- the public values are national summary statistics with different universes
- the ACS unemployment anchor may differ from Bureau of Labor Statistics labor
  force estimates because survey design and data collection differ
- three aggregate anchors cannot establish representativeness
- the comparison does not validate synthetic labels or ranking outcomes

The diagnostic is useful because it makes selected synthetic assumptions explicit,
versioned, and reviewable while preserving the benchmark's controlled synthetic-data
design.
