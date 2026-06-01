# Aggregate-Noise Sensitivity

FairPrivacySignal includes an aggregate-noise sensitivity sweep for the privacy-safe
recovery layer. The experiment asks whether the observed utility recovery depends on
one favorable random perturbation or one fixed noise-strength setting.

## Design

The diagnostic fixes the generated synthetic dataset at seed `42` and varies only
the privacy-safe aggregate transformation:

| Parameter | Values |
|---|---|
| Aggregate-noise stress scale | `0.0`, `0.5`, `1.0`, `2.0`, `4.0` |
| Aggregate-noise seed | `7`, `42`, `101` |
| Signal-loss scenarios | Severe signal loss, policy restricted |
| Metrics | Overall NDCG@3, low-signal NDCG@3 |

For each scenario and noise scale, the benchmark reports the mean and standard
deviation across three reproducible noise realizations. The chart also shows the
corresponding no-aggregate scenario baseline and marks the default stress scale of
`1.0`.

## Interpretation

The transformation adds Laplace noise scaled by cohort size:

    dp_noise_scale / sqrt(cohort_size)

The sweep is a stress test for the simplified aggregate-recovery mechanism. It is
not an epsilon sweep, a formal privacy accounting analysis, or a claim of
production-grade differential privacy. Its purpose is narrower: make the effect of
the configured noise mechanism visible and reproducible instead of relying on one
parameter point.
