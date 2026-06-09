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

## Learning-to-Rank Objectives

[Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
introduces RankNet and a pairwise probabilistic cost for ranking functions.
[Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/?p=153086)
describes pairwise methods as treating object pairs as learning instances and
introduces permutation and top-one probability models for listwise learning.

FairPrivacySignal adds lightweight linear pairwise and listwise comparators. Each
synthetic household yields relevant-versus-nonrelevant service pairs for the
pairwise path and a complete candidate-service list for the top-one softmax
listwise path. These are objective-sensitivity diagnostics inspired by
learning-to-rank formulations, not implementations of RankNet or the original
neural ListNet architecture.

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
- paired ranking-objective comparison
- paired quartile-level heterogeneity reporting
- tracked figures generated from auditable CSV outputs
- focused regression tests for experiment orchestration
- explicit limitations and non-confidential synthetic-data framing

## Distribution-Shift Evaluation

[WILDS](https://proceedings.mlr.press/v139/koh21a.html) highlights the importance
of standardized evaluation when training and test distributions differ.
FairPrivacySignal adds a narrow synthetic stress test in that spirit: after the
household-level split, it moves bounded context covariates and context buckets only
on the holdout side while keeping labels fixed. This is a controlled
covariate-drift proxy, not a reproduction of a WILDS dataset or a temporal
validation study.

## Holdout Evaluation Hygiene

The [scikit-learn common pitfalls guide](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
recommends splitting train and test data before learning preprocessing statistics,
then applying the learned transformation to both subsets. FairPrivacySignal follows
that pattern for privacy-safe aggregates: cohort statistics and service-level
fallbacks are fitted from training households before holdout scoring.

## Group-Held-Out Robustness

The scikit-learn
[`GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
documentation describes randomized train/test indices based on externally supplied
groups, including time-based group splits as one example. FairPrivacySignal uses
synthetic community IDs as grouping keys for an additional paired stress test. The
diagnostic keeps training and evaluation communities disjoint without presenting
the result as real geographic validation.

## Public-Reference Anchors

The benchmark also uses a small tracked snapshot from the
[U.S. Census Bureau QuickFacts United States page](https://www.census.gov/quickfacts/fact/table/US/PST045225)
and [2024 ACS 5-Year Data Profile DP03](https://data.census.gov/table/ACSDP5Y2024.DP03?g=010XX00US)
to make selected synthetic context priors inspectable. The comparison reports gaps
without claiming that synthetic communities are representative of a real
population. See [`docs/public_reference_calibration.md`](public_reference_calibration.md).
