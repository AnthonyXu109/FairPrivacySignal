# Multi-seed Evaluation Summary

This table reports mean ± standard deviation across five synthetic data seeds.

| Scenario                                    | Privacy exposure   | NDCG@3        | Low-signal NDCG@3   | Low-signal gap   |
|:--------------------------------------------|:-------------------|:--------------|:--------------------|:-----------------|
| Full signal raw baseline                    | 0.925 ± 0.002      | 0.555 ± 0.011 | 0.490 ± 0.014       | 0.095 ± 0.009    |
| Policy restricted                           | 0.728 ± 0.007      | 0.526 ± 0.007 | 0.451 ± 0.008       | 0.109 ± 0.010    |
| Policy restricted + privacy-safe aggregates | 0.728 ± 0.007      | 0.539 ± 0.006 | 0.460 ± 0.007       | 0.115 ± 0.005    |
| Severe signal loss                          | 0.475 ± 0.002      | 0.504 ± 0.007 | 0.430 ± 0.014       | 0.108 ± 0.018    |
| Severe loss + privacy-safe aggregates       | 0.475 ± 0.002      | 0.520 ± 0.007 | 0.448 ± 0.015       | 0.106 ± 0.018    |
