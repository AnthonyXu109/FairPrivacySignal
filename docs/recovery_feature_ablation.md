# Recovery Feature Ablation

This diagnostic separates the privacy-safe recovery layer into inspectable feature groups. It compares no aggregate substitutes, an engagement aggregate only, cohort-context aggregates only, and their combined use.

Each recovery value is a paired difference against the no-aggregate baseline for the same signal-loss scenario and synthetic-data seed. The table reports mean +/- standard deviation across five seeds.

| Scenario           | Feature set                      | Overall NDCG@3   | Overall recovery   | Low-signal recovery   |
|:-------------------|:---------------------------------|:-----------------|:-------------------|:----------------------|
| Severe signal loss | No aggregate substitutes         | 0.504 +/- 0.007  | +0.000 +/- 0.000   | +0.000 +/- 0.000      |
| Severe signal loss | Engagement aggregate only        | 0.520 +/- 0.006  | +0.016 +/- 0.008   | +0.015 +/- 0.010      |
| Severe signal loss | Cohort context aggregates only   | 0.504 +/- 0.008  | +0.001 +/- 0.003   | +0.001 +/- 0.004      |
| Severe signal loss | Combined privacy-safe aggregates | 0.520 +/- 0.007  | +0.016 +/- 0.009   | +0.018 +/- 0.010      |
| Policy restricted  | No aggregate substitutes         | 0.526 +/- 0.007  | +0.000 +/- 0.000   | +0.000 +/- 0.000      |
| Policy restricted  | Engagement aggregate only        | 0.539 +/- 0.005  | +0.013 +/- 0.004   | +0.008 +/- 0.008      |
| Policy restricted  | Cohort context aggregates only   | 0.527 +/- 0.007  | +0.001 +/- 0.002   | +0.001 +/- 0.004      |
| Policy restricted  | Combined privacy-safe aggregates | 0.539 +/- 0.006  | +0.013 +/- 0.005   | +0.009 +/- 0.009      |

## Interpretation limits

The ablation isolates feature-group contributions within this synthetic benchmark. It does not prove that the same contributions will transfer to a real deployment, and it should not be interpreted as formal privacy accounting.
