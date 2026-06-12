# Policy-Aware Signal Recovery

Policy-Aware Signal Recovery is the repository's primary recovery method for ranking systems whose behavioral features are available during controlled offline training but unavailable or partially unavailable when the ranker is served.

## Method

1. A reconstruction model learns the training-only behavioral signal from policy-permitted context and candidate features.
2. Household-grouped cross-fitting produces out-of-fold reconstructed signals for downstream model training, preventing each training event from being reconstructed by a model fitted on that event.
3. At serving time, observed behavioral values are retained when policy allows them; unavailable values are replaced by the learned reconstruction.
4. Under partial restriction, the reconstruction is fused with train-fitted, thresholded, noise-stressed cohort aggregates. Under complete loss, the method uses the more stable aggregate path because no observed event-level signal remains to anchor per-event reconstruction.
5. The downstream ranker never receives the hidden raw behavioral value for an unavailable event.

![Policy-aware signal recovery results](assets/policy_aware_signal_recovery.png)

## Paired results

| Signal-loss regime               | Method                      | Overall NDCG@3   | Low-signal NDCG@3   | Overall gap closed   | Positive seeds   |
|:---------------------------------|:----------------------------|:-----------------|:--------------------|:---------------------|:-----------------|
| Complete behavioral-signal loss  | Full-signal oracle          | 0.555 +/- 0.011  | 0.490 +/- 0.014     | -                    | -                |
| Complete behavioral-signal loss  | Signal-loss baseline        | 0.504 +/- 0.007  | 0.430 +/- 0.014     | -                    | -                |
| Complete behavioral-signal loss  | Missingness indicator       | 0.504 +/- 0.007  | 0.430 +/- 0.014     | 0.0%                 | 0%               |
| Complete behavioral-signal loss  | Flat aggregates             | 0.519 +/- 0.006  | 0.445 +/- 0.012     | 31.4%                | 100%             |
| Complete behavioral-signal loss  | Cross-fitted reconstruction | 0.519 +/- 0.007  | 0.447 +/- 0.013     | 30.6%                | 100%             |
| Complete behavioral-signal loss  | Policy-aware recovery       | 0.519 +/- 0.006  | 0.445 +/- 0.012     | 31.4%                | 100%             |
| Policy-restricted partial signal | Full-signal oracle          | 0.555 +/- 0.011  | 0.490 +/- 0.014     | -                    | -                |
| Policy-restricted partial signal | Signal-loss baseline        | 0.526 +/- 0.007  | 0.451 +/- 0.008     | -                    | -                |
| Policy-restricted partial signal | Missingness indicator       | 0.527 +/- 0.006  | 0.454 +/- 0.009     | 3.1%                 | 80%              |
| Policy-restricted partial signal | Flat aggregates             | 0.539 +/- 0.006  | 0.460 +/- 0.008     | 45.3%                | 100%             |
| Policy-restricted partial signal | Cross-fitted reconstruction | 0.534 +/- 0.005  | 0.462 +/- 0.006     | 27.3%                | 100%             |
| Policy-restricted partial signal | Policy-aware recovery       | 0.542 +/- 0.006  | 0.468 +/- 0.006     | 56.1%                | 100%             |

Across the five paired synthetic draws, policy-aware recovery improves overall NDCG@3 by `+0.015` under complete signal loss and `+0.015` under partial policy restriction relative to the matching no-recovery baselines. These changes close `31.4%` and `56.1%` of the respective synthetic full-signal utility gaps.

## Applicability boundary

This method applies when historical behavioral signal may be used inside a controlled offline training process but may not be exposed to the serving ranker. If a policy prohibits use of the signal even during offline model fitting, the reconstruction path is not applicable and the aggregate-only path remains the appropriate comparator. The implementation does not claim formal differential privacy or immunity to model-extraction attacks.
