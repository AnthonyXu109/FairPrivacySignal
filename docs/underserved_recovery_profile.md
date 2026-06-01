# Underserved Quartile Recovery Profile

This diagnostic checks whether pooled ranking-recovery results hide uneven effects across synthetic community contexts. Communities are assigned to quartiles using their synthetic underserved score before the logistic primary baseline is evaluated.

Each recovery value is a paired aggregate-minus-baseline NDCG@3 difference for the same signal-loss scenario and synthetic-data seed. Quartiles are formed from distinct synthetic communities, not weighted by event volume. The table reports mean +/- standard deviation across five seeds.

| Scenario           | Underserved quartile   | Low-signal event share   | Overall recovery   | Low-signal recovery   |
|:-------------------|:-----------------------|:-------------------------|:-------------------|:----------------------|
| Severe signal loss | Q1 lower               | 31.7%                    | +0.021 +/- 0.025   | +0.030 +/- 0.031      |
| Severe signal loss | Q2                     | 36.0%                    | +0.028 +/- 0.012   | +0.025 +/- 0.009      |
| Severe signal loss | Q3                     | 39.4%                    | +0.008 +/- 0.006   | +0.008 +/- 0.024      |
| Severe signal loss | Q4 higher              | 42.3%                    | +0.003 +/- 0.011   | +0.001 +/- 0.023      |
| Policy restricted  | Q1 lower               | 31.7%                    | +0.013 +/- 0.017   | +0.010 +/- 0.029      |
| Policy restricted  | Q2                     | 36.0%                    | +0.023 +/- 0.012   | +0.023 +/- 0.009      |
| Policy restricted  | Q3                     | 39.4%                    | +0.010 +/- 0.003   | +0.004 +/- 0.019      |
| Policy restricted  | Q4 higher              | 42.3%                    | +0.003 +/- 0.014   | -0.002 +/- 0.016      |

## Interpretation limits

This profile is a synthetic heterogeneity diagnostic. Positive pooled recovery should not be read as uniform benefit: a quartile can show a negative low-signal recovery delta in the same configuration. The quartiles are benchmark constructs, not real-world demographic groups, and the diagnostic does not establish domain-specific fairness.
