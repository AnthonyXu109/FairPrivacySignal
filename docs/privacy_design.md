# Privacy Design and Trust Boundary

FairPrivacySignal separates a controlled offline recovery process from the serving
ranker. This document states what each stage may access and what the repository
does not guarantee.

## Offline recovery boundary

The current method assumes historical behavioral signal may be processed in a
controlled offline environment for one or both of these operations:

- construction of thresholded cohort aggregates from training households
- fitting of a signal-reconstruction model on training households

The holdout population never contributes to either operation. Reconstruction
proxies used to train the downstream ranker are produced with five-fold
household-grouped cross-fitting.

## Serving boundary

At serving time:

- policy-permitted behavioral values may remain available
- unavailable values are not restored from the hidden source column
- complete-loss settings use train-fitted aggregate features
- partial-loss settings replace unavailable values with model-based
  reconstructions and combine them with aggregate features
- the downstream ranker receives permitted context, candidate attributes, and
  recovery outputs only

## Aggregate controls

The aggregate path includes:

- minimum cohort-size thresholds
- broad service-level fallback values
- train-only reference construction
- reproducible Laplace-noise stress tests

These controls make information flow inspectable. They are not a formal privacy
accounting mechanism.

## Non-goals

The repository does not claim:

- formal differential privacy
- protection against membership inference or model extraction
- that reconstructed values are anonymous
- that model parameters cannot encode information about training data
- compliance with any particular law, regulation, or organizational policy

If historical signal cannot be used even inside the offline boundary, the
reconstruction path must be disabled. That policy setting is represented by the
aggregate-only comparator.
