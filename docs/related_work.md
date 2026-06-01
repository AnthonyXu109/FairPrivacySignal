# Methodological Context

FairPrivacySignal is a synthetic engineering benchmark. It does not reproduce any
single prior system, and it does not claim that its simplified mechanisms solve the
broader research problems below. The benchmark design is informed by several public
research directions.

## Fairness-Utility Tradeoffs in Ranking

Fair ranking research often treats utility and fairness as competing objectives
rather than assuming that one metric can be optimized without affecting the other.

- [Toward Pareto Efficient Fairness-Utility Trade-off in Recommendation through Reinforcement Learning](https://arxiv.org/abs/2201.00140)
  motivates exploring a Pareto frontier so that decision-makers can inspect a range
  of tradeoffs rather than a single fixed preference.
- [Pareto-Optimal Fairness-Utility Amortizations in Rankings with a DBN Exposure Model](https://arxiv.org/abs/2205.07647)
  studies Pareto-optimal ranking policies under exposure constraints.

FairPrivacySignal uses a simpler allocation-floor sweep. It reports how allocated
relevance, low-signal representation, and selection-rate gaps move as a low-signal
allocation floor becomes stronger.

## Ranking Calibration Diagnostics

[Matched Pair Calibration for Ranking Fairness](https://arxiv.org/abs/2306.03775)
proposes comparing subgroup outcomes among similarly scored items to diagnose ranking
fairness. Inspired by that direction, FairPrivacySignal reports a lightweight
score-matched subgroup calibration diagnostic beyond aggregate group gaps. It bins
similarly scored test-set events and compares observed relevance between low-signal
and not-low-signal groups. This is not an implementation of the paper's full method.

## Standardized Benchmark Practice

Public benchmarks such as [WILDS](https://proceedings.mlr.press/v139/koh21a.html)
and [RobustBench](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a3c65c2974270fd093ee8a9bf8ae7d0b-Abstract-round2.html)
emphasize standardized evaluation protocols, reproducible scripts, and explicit
task definitions. [BetterBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/26889e8359e7ef8a7f5d77457364ca55-Abstract-Datasets_and_Benchmarks_Track.html)
also identifies replicability and statistical reporting as recurring benchmark
quality gaps.

FairPrivacySignal therefore prioritizes:

- a one-command regeneration pipeline
- a documented experiment matrix
- multi-seed mean and standard-deviation reporting
- paired feature-ablation reporting
- lightweight baseline model comparison
- tracked figures generated from auditable CSV outputs
- focused regression tests for experiment orchestration
- explicit limitations and non-confidential synthetic-data framing

## Public-Reference Anchors

The benchmark also uses a small tracked snapshot from the
[U.S. Census Bureau QuickFacts United States page](https://www.census.gov/quickfacts/fact/table/US/PST045225)
to make selected synthetic context priors inspectable. The comparison reports gaps
without claiming that synthetic communities are representative of a real
population. See [`docs/public_reference_calibration.md`](public_reference_calibration.md).
