# Limitations

FairPrivacySignal is a synthetic engineering benchmark. Its outputs are useful for
testing evaluation patterns, but they should not be interpreted as real-world impact
estimates.

## Synthetic Data

The generated households, communities, services, and relevance labels do not model
or identify real people or places. Synthetic results alone cannot establish that a
method will generalize to a real service-delivery setting.

## Privacy

The privacy exposure score is an interpretable comparison proxy. DP-style noise and
k-thresholded cohort aggregates illustrate a design pattern; they do not provide a
production privacy guarantee or a complete privacy accounting framework.

The aggregate-noise sensitivity sweep makes the simplified noise mechanism more
inspectable, but it remains a stress diagnostic. Its noise scale should not be
interpreted as epsilon or as formal privacy accounting.

The cohort-threshold sensitivity sweep makes fallback coverage visible, but it does
not establish that any threshold is appropriate for a real deployment.

Aggregate statistics are fitted on training households and mapped onto the holdout
split. This reduces preprocessing leakage, but the benchmark still uses a simplified
evaluation design rather than a full temporal or geographic validation study.
The community-held-out robustness diagnostic adds a stricter synthetic split with
disjoint training and evaluation communities, but it does not establish transfer
to a real place, time period, or service-delivery setting.

## Fairness

Low-signal ranking gaps, allocation gaps, and score-matched calibration metrics are
diagnostics. They do not prove that a ranking or allocation policy is fair. The
low-signal label is a benchmark construct and is not a substitute for
domain-specific evaluation with affected communities and subject-matter experts.

## Modeling Scope

The benchmark uses an interpretable logistic baseline and simplified
capacity-allocation policies. It does not model every feedback loop, operational
constraint, or long-term effect that could arise in deployed ranking systems.
The recovery feature ablation isolates contributions within this synthetic
configuration; it does not establish that the same contributions transfer to a
real deployment.
The model-sensitivity diagnostic compares two lightweight classifier families. It
does not establish robustness across all model classes or implement a
ranking-specific learning objective.
The underserved-quartile recovery profile exposes heterogeneity across synthetic
community contexts. Its quartiles are benchmark constructs, not real-world
demographic groups, and positive pooled recovery does not establish uniform benefit.
The community-held-out robustness diagnostic tests generated context separation,
not real-world geographic generalization.
