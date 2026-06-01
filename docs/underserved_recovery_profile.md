# Underserved Quartile Recovery Profile

This diagnostic checks whether pooled ranking-recovery results hide uneven effects across synthetic community contexts. Communities are assigned to quartiles using their synthetic underserved score before the logistic primary baseline is evaluated.

Each recovery value is a paired aggregate-minus-baseline NDCG@3 difference for the same signal-loss scenario and synthetic-data seed. Quartiles are formed from distinct synthetic communities, not weighted by event volume. The table reports mean +/- standard deviation across five seeds.

| Scenario           | Underserved quartile   | Low-signal event share   | Overall recovery   | Low-signal recovery   |
|:-------------------|:-----------------------|:-------------------------|:-------------------|:----------------------|
| Severe signal loss | Q1 lower               | 31.7%                    | +0.017 +/- 0.024   | +0.023 +/- 0.036      |
| Severe signal loss | Q2                     | 36.0%                    | +0.033 +/- 0.014   | +0.035 +/- 0.014      |
| Severe signal loss | Q3                     | 39.4%                    | +0.007 +/- 0.004   | +0.004 +/- 0.023      |
| Severe signal loss | Q4 higher              | 42.3%                    | +0.007 +/- 0.008   | +0.009 +/- 0.021      |
| Policy restricted  | Q1 lower               | 31.7%                    | +0.012 +/- 0.016   | +0.009 +/- 0.031      |
| Policy restricted  | Q2                     | 36.0%                    | +0.027 +/- 0.011   | +0.027 +/- 0.012      |
| Policy restricted  | Q3                     | 39.4%                    | +0.008 +/- 0.003   | +0.002 +/- 0.020      |
| Policy restricted  | Q4 higher              | 42.3%                    | +0.004 +/- 0.013   | -0.001 +/- 0.013      |

## Interpretation limits

This profile is a synthetic heterogeneity diagnostic. Positive pooled recovery should not be read as uniform benefit: a quartile can show a negative low-signal recovery delta in the same configuration. The quartiles are benchmark constructs, not real-world demographic groups, and the diagnostic does not establish domain-specific fairness.
