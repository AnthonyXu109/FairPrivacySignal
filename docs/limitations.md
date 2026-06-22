# Limitations

FairPrivacySignal is a public, synthetic benchmark and reference method. The scope
conditions below state what the controlled synthetic study does and does not claim, so
that results are not over-read. They are standard scientific scope statements for a
methodological benchmark; they do **not** imply that the underlying problem, or the
method itself, lacks real-world relevance. The real-world motivation and its documented,
economy-wide U.S. scale are summarized in the README section "Why this problem matters"
and in [`policy_context.md`](policy_context.md).

The outputs below should be read as measurements of method behavior under controlled,
fully disclosed synthetic conditions, not as estimates of the effect size a deployed
system would observe.

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

Policy-Aware Signal Recovery assumes historical behavioral signal may be processed
inside a controlled offline training or aggregation environment even when the
serving ranker cannot receive that signal. The reconstruction path is not
applicable when policy prohibits use of the signal during offline fitting. Model
weights and reconstructed values are not formally differentially private, and the
benchmark does not evaluate model-extraction or membership-inference risk.

The reconstruction model predicts a behavioral feature from permitted context; it
does not recover the original value exactly. Cross-fitting reduces own-record
training leakage and train-serving mismatch, but it does not prove transfer to a
different population or policy environment.

Aggregate statistics are fitted on training households and mapped onto the holdout
split. This reduces preprocessing leakage, but the benchmark still uses a simplified
evaluation design rather than a full temporal or geographic validation study.
The community-held-out robustness diagnostic adds a stricter synthetic split with
disjoint training and evaluation communities, but it does not establish transfer
to a real place, time period, or service-delivery setting.
The heldout context-shift stress test moves selected synthetic covariates and
deterministically remaps selected holdout context buckets after the household-level
split while keeping labels fixed. It exposes sensitivity to one controlled drift
family, but it is not a temporal validation study, a model of real-world shift, or
evidence of deployment robustness.

The matched-rate missingness diagnostic uses controlled MCAR-like, MAR-like, and
MNAR-like analogues. These labels describe how the synthetic masks are generated;
they do not identify a missingness mechanism in observed data or provide a
counterfactual correction method.

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
The aggregate-alignment negative control deliberately breaks service-to-signal
alignment while preserving the surrounding synthetic pipeline. A weaker permuted
result supports a structural interpretation within this benchmark, but it does
not identify a causal mechanism or establish transfer beyond the generated data.
The missingness-mechanism sensitivity experiment isolates signal quantity from
incidence in a semi-synthetic stress test. Its signal-dependent path is
intentionally adversarial and should not be interpreted as an estimated real-world
selection process.
The model-sensitivity diagnostic compares two lightweight classifier families. It
does not establish robustness across all model classes or implement a
ranking-specific learning objective.
The ranking-objective diagnostic adds lightweight linear pairwise and listwise
comparators, but it does not implement a neural ranking architecture or establish
that one training objective is universally better. The listwise comparator uses a
top-one softmax objective but is not an implementation of the original neural
ListNet architecture.
The underserved-quartile recovery profile exposes heterogeneity across synthetic
community contexts. Its quartiles are benchmark constructs, not real-world
demographic groups, and positive pooled recovery does not establish uniform benefit.
The uncertainty audit measures variation across household-bootstrap training
samples. It is not a calibrated posterior distribution, an event-level confidence
interval, or an implementation of Equal-Opportunity Ranking. Top-3 agreement
measures membership stability rather than correctness.
The community-held-out robustness diagnostic tests generated context separation,
not real-world geographic generalization.
