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

## Fairness

Low-signal ranking gaps, allocation gaps, and score-matched calibration metrics are
diagnostics. They do not prove that a ranking or allocation policy is fair. The
low-signal label is a benchmark construct and is not a substitute for
domain-specific evaluation with affected communities and subject-matter experts.

## Modeling Scope

The benchmark uses an interpretable logistic baseline and simplified
capacity-allocation policies. It does not model every feedback loop, operational
constraint, or long-term effect that could arise in deployed ranking systems.
