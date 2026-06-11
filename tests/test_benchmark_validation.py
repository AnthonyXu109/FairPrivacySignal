from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import benchmark_validation


def _signal_loss() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": benchmark_validation.EXPECTED_SIGNAL_SCENARIOS,
            "num_events": [100, 100, 100, 100],
            "behavioral_available_share": [1.0, 0.8, 0.6, 0.0],
            "avg_privacy_exposure_score": [0.95, 0.82, 0.70, 0.45],
        }
    )


def _recovery() -> pd.DataFrame:
    reference_scopes = [
        (
            "train_households_only"
            if experiment
            in benchmark_validation.EXPECTED_TRAIN_FITTED_RECOVERY_EXPERIMENTS
            else "not_applicable"
        )
        for experiment in benchmark_validation.EXPECTED_RECOVERY_EXPERIMENTS
    ]
    return pd.DataFrame(
        {
            "experiment": benchmark_validation.EXPECTED_RECOVERY_EXPERIMENTS,
            "overall_auc": [0.60] * 7,
            "overall_ndcg_at_3": [0.60, 0.50, 0.55, 0.56, 0.52, 0.54, 0.55],
            "low_signal_ndcg_at_3": [0.50] * 7,
            "not_low_signal_ndcg_at_3": [0.60] * 7,
            "aggregate_reference_scope": reference_scopes,
        }
    )


def _capacity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "allocated_relevance_rate": [0.50],
            "low_signal_selection_rate": [0.10],
            "not_low_signal_selection_rate": [0.20],
            "allocated_low_signal_share": [0.30],
            "num_allocated": [10],
            "num_candidate_events": [100],
        }
    )


def _score_calibration() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_matched_bins": [4],
            "low_signal_ece": [0.20],
            "not_low_signal_ece": [0.10],
            "mean_absolute_matched_relevance_gap": [0.15],
        }
    )


def _public_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "median_household_income",
                "broadband_subscription_share",
                "unemployment_rate",
            ],
            "reference_value": [80734, 0.91, 0.052],
            "synthetic_value": [65000, 0.80, 0.07],
            "synthetic_minus_reference": [-15734, -0.11, 0.018],
            "relative_gap_vs_reference": [-0.19, -0.12, 0.35],
            "synthetic_as_share_of_reference": [0.81, 0.88, 1.35],
            "source_url": [
                "https://www.census.gov/quickfacts/example",
                "https://www.census.gov/quickfacts/example",
                "https://data.census.gov/table/example",
            ],
        }
    )


def _aggregate_noise_sensitivity_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "noise_scale": noise_scale,
                "noise_seed": noise_seed,
                "baseline_overall_ndcg_at_3": 0.50,
                "aggregate_overall_ndcg_at_3": 0.53,
                "baseline_low_signal_ndcg_at_3": 0.40,
                "aggregate_low_signal_ndcg_at_3": 0.44,
            }
            for scenario in benchmark_validation.EXPECTED_AGGREGATE_NOISE_SCENARIOS
            for noise_scale in benchmark_validation.EXPECTED_AGGREGATE_NOISE_SCALES
            for noise_seed in benchmark_validation.EXPECTED_AGGREGATE_NOISE_SEEDS
        ]
    )


def _cohort_threshold_sensitivity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "min_cohort_size": threshold,
                "suppressed_event_share": suppressed_share,
                "suppressed_unique_cohort_share": suppressed_share,
                "baseline_overall_ndcg_at_3": 0.50,
                "aggregate_overall_ndcg_at_3": 0.53,
                "baseline_low_signal_ndcg_at_3": 0.40,
                "aggregate_low_signal_ndcg_at_3": 0.44,
            }
            for scenario in benchmark_validation.EXPECTED_AGGREGATE_NOISE_SCENARIOS
            for threshold, suppressed_share in [
                (25, 0.00),
                (50, 0.01),
                (100, 0.03),
                (200, 0.12),
                (400, 0.37),
                (800, 0.56),
            ]
        ]
    )


def _recovery_feature_ablation_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "variant": variant,
                "seed": seed,
                "overall_ndcg_at_3": 0.53,
                "low_signal_ndcg_at_3": 0.44,
            }
            for scenario in benchmark_validation.EXPECTED_AGGREGATE_NOISE_SCENARIOS
            for variant in benchmark_validation.EXPECTED_ABLATION_VARIANTS
            for seed in benchmark_validation.EXPECTED_SEEDS
        ]
    )


def _aggregate_alignment_negative_control_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "variant": variant,
                "seed": seed,
                "service_alignment": (
                    benchmark_validation.EXPECTED_ALIGNMENT_CONTROL_ALIGNMENTS[
                        variant
                    ]
                ),
                "overall_ndcg_at_3": 0.53,
                "low_signal_ndcg_at_3": 0.44,
                "num_test_events": 100,
                "aggregate_reference_scope": (
                    "not_applicable"
                    if variant == "no_aggregate_substitutes"
                    else "train_households_only"
                ),
            }
            for scenario in benchmark_validation.EXPECTED_AGGREGATE_NOISE_SCENARIOS
            for variant in benchmark_validation.EXPECTED_ALIGNMENT_CONTROL_VARIANTS
            for seed in benchmark_validation.EXPECTED_ALIGNMENT_CONTROL_SEEDS
        ]
    )


def _missingness_mechanism_sensitivity_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mechanism": mechanism,
                "variant": variant,
                "seed": seed,
                "behavioral_available_share": 0.56,
                "low_signal_available_share": 0.45,
                "not_low_signal_available_share": 0.62,
                "overall_ndcg_at_3": 0.53,
                "low_signal_ndcg_at_3": 0.44,
                "num_test_events": 100,
                "aggregate_reference_scope": (
                    "train_households_only"
                    if variant == "privacy_safe_aggregates"
                    else "not_applicable"
                ),
            }
            for mechanism in benchmark_validation.EXPECTED_MISSINGNESS_MECHANISMS
            for variant in benchmark_validation.EXPECTED_MISSINGNESS_VARIANTS
            for seed in benchmark_validation.EXPECTED_MISSINGNESS_SEEDS
        ]
    )


def _model_sensitivity_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "experiment": experiment,
                "seed": seed,
                "overall_auc": 0.60,
                "overall_ndcg_at_3": 0.53,
                "low_signal_ndcg_at_3": 0.44,
            }
            for model in benchmark_validation.EXPECTED_MODEL_SENSITIVITY_MODELS
            for experiment in benchmark_validation.EXPECTED_MODEL_SENSITIVITY_EXPERIMENTS
            for seed in benchmark_validation.EXPECTED_MODEL_SENSITIVITY_SEEDS
        ]
    )


def _pairwise_ranking_sensitivity_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "objective": objective,
                "experiment": experiment,
                "seed": seed,
                "overall_auc": 0.60,
                "overall_ndcg_at_3": 0.53,
                "low_signal_ndcg_at_3": 0.44,
                "not_low_signal_ndcg_at_3": 0.55,
                "num_training_pairs": (
                    100 if objective == "linear_pairwise" else 0
                ),
                "num_training_lists": (
                    100 if objective == "linear_listwise" else 0
                ),
                "aggregate_reference_scope": (
                    "train_households_only"
                    if experiment
                    in benchmark_validation.EXPECTED_TRAIN_FITTED_RECOVERY_EXPERIMENTS
                    else "not_applicable"
                ),
            }
            for objective in benchmark_validation.EXPECTED_RANKING_OBJECTIVES
            for experiment in benchmark_validation.EXPECTED_RANKING_EXPERIMENTS
            for seed in benchmark_validation.EXPECTED_RANKING_SEEDS
        ]
    )


def _underserved_recovery_profile_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "variant": variant,
                "underserved_quartile": quartile,
                "seed": seed,
                "overall_ndcg_at_3": 0.53,
                "low_signal_ndcg_at_3": 0.44,
                "low_signal_share": 0.35,
                "num_test_events": 100,
                "num_low_signal_events": 35,
                "num_communities": 30,
            }
            for scenario in benchmark_validation.EXPECTED_AGGREGATE_NOISE_SCENARIOS
            for variant in benchmark_validation.EXPECTED_UNDERSERVED_PROFILE_VARIANTS
            for quartile in benchmark_validation.EXPECTED_UNDERSERVED_QUARTILES
            for seed in benchmark_validation.EXPECTED_SEEDS
        ]
    )


def _community_holdout_robustness_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "split_strategy": split_strategy,
                "variant": variant,
                "seed": seed,
                "overall_auc": 0.60,
                "overall_ndcg_at_3": 0.53,
                "low_signal_ndcg_at_3": 0.44,
                "not_low_signal_ndcg_at_3": 0.55,
                "fallback_event_share": 0.05,
                "unseen_cohort_share": 0.0,
                "heldout_community_share": (
                    1.0 if split_strategy == "community_holdout" else 0.0
                ),
                "num_test_events": 100,
                "num_test_households": 20,
                "num_test_communities": 4,
                "aggregate_reference_scope": (
                    "train_households_only"
                    if variant == "privacy_safe_aggregates"
                    else "not_applicable"
                ),
            }
            for scenario in benchmark_validation.EXPECTED_AGGREGATE_NOISE_SCENARIOS
            for split_strategy in benchmark_validation.EXPECTED_COMMUNITY_HOLDOUT_SPLITS
            for variant in benchmark_validation.EXPECTED_COMMUNITY_HOLDOUT_VARIANTS
            for seed in benchmark_validation.EXPECTED_SEEDS
        ]
    )


def _heldout_context_shift_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "shift_level": shift_level,
                "shift_strength": shift_strength,
                "variant": variant,
                "seed": seed,
                "overall_auc": 0.60,
                "overall_ndcg_at_3": 0.53,
                "low_signal_ndcg_at_3": 0.44,
                "not_low_signal_ndcg_at_3": 0.55,
                "fallback_event_share": 0.05,
                "bucket_migration_share": shift_strength * 0.80,
                "num_test_events": 100,
                "num_test_households": 20,
                "aggregate_reference_scope": (
                    "train_households_only"
                    if variant == "privacy_safe_aggregates"
                    else "not_applicable"
                ),
            }
            for scenario in benchmark_validation.EXPECTED_CONTEXT_SHIFT_SCENARIOS
            for shift_level, shift_strength in (
                benchmark_validation.EXPECTED_CONTEXT_SHIFT_LEVELS.items()
            )
            for variant in benchmark_validation.EXPECTED_CONTEXT_SHIFT_VARIANTS
            for seed in benchmark_validation.EXPECTED_CONTEXT_SHIFT_SEEDS
        ]
    )


def _multiseed_recovery_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"experiment": experiment, "seed": seed}
            for experiment in benchmark_validation.EXPECTED_RECOVERY_EXPERIMENTS
            for seed in benchmark_validation.EXPECTED_SEEDS
        ]
    )


def _multiseed_capacity_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": "example",
                "capacity_rate": 0.15,
                "low_signal_floor_fraction": 0.50,
                "seed": seed,
            }
            for seed in benchmark_validation.EXPECTED_SEEDS
        ]
    )


def _checks() -> pd.DataFrame:
    return benchmark_validation.build_validation_checks(
        signal_loss=_signal_loss(),
        recovery=_recovery(),
        capacity=_capacity(),
        score_calibration=_score_calibration(),
        public_reference=_public_reference(),
        aggregate_noise_sensitivity_raw=_aggregate_noise_sensitivity_raw(),
        cohort_threshold_sensitivity=_cohort_threshold_sensitivity(),
        recovery_feature_ablation_raw=_recovery_feature_ablation_raw(),
        aggregate_alignment_negative_control_raw=(
            _aggregate_alignment_negative_control_raw()
        ),
        missingness_mechanism_sensitivity_raw=(
            _missingness_mechanism_sensitivity_raw()
        ),
        model_sensitivity_raw=_model_sensitivity_raw(),
        pairwise_ranking_sensitivity_raw=_pairwise_ranking_sensitivity_raw(),
        underserved_recovery_profile_raw=_underserved_recovery_profile_raw(),
        community_holdout_robustness_raw=_community_holdout_robustness_raw(),
        heldout_context_shift_raw=_heldout_context_shift_raw(),
        multiseed_recovery_raw=_multiseed_recovery_raw(),
        multiseed_capacity_raw=_multiseed_capacity_raw(),
    )


def test_benchmark_validation_checks_pass_for_consistent_metrics() -> None:
    checks = _checks()

    benchmark_validation.raise_for_failed_required_checks(checks)

    assert (checks[checks["required"]]["status"] == "PASS").all()


def test_benchmark_validation_rejects_signal_loss_drift() -> None:
    checks = _checks()
    checks.loc[
        checks["check"] == "signal-loss availability endpoints are correct",
        "status",
    ] = "FAIL"

    with pytest.raises(RuntimeError, match="availability endpoints"):
        benchmark_validation.raise_for_failed_required_checks(checks)


def test_benchmark_validation_rejects_aggregate_reference_scope_drift() -> None:
    checks = _checks()
    checks.loc[
        checks["check"] == "aggregate preprocessing is train-fitted",
        "status",
    ] = "FAIL"

    with pytest.raises(RuntimeError, match="aggregate preprocessing"):
        benchmark_validation.raise_for_failed_required_checks(checks)


def test_benchmark_validation_rejects_community_holdout_drift() -> None:
    checks = _checks()
    checks.loc[
        checks["check"] == "community-held-out robustness coverage is complete",
        "status",
    ] = "FAIL"

    with pytest.raises(RuntimeError, match="community-held-out"):
        benchmark_validation.raise_for_failed_required_checks(checks)


def test_benchmark_validation_rejects_ranking_objective_drift() -> None:
    checks = _checks()
    checks.loc[
        checks["check"] == "ranking-objective sensitivity coverage is complete",
        "status",
    ] = "FAIL"

    with pytest.raises(RuntimeError, match="ranking-objective sensitivity"):
        benchmark_validation.raise_for_failed_required_checks(checks)


def test_benchmark_validation_rejects_alignment_control_drift() -> None:
    checks = _checks()
    checks.loc[
        checks["check"]
        == "aggregate-alignment negative-control coverage is complete",
        "status",
    ] = "FAIL"

    with pytest.raises(RuntimeError, match="aggregate-alignment negative-control"):
        benchmark_validation.raise_for_failed_required_checks(checks)


def test_benchmark_validation_rejects_missingness_mechanism_drift() -> None:
    checks = _checks()
    checks.loc[
        checks["check"]
        == "missingness-mechanism sensitivity coverage is complete",
        "status",
    ] = "FAIL"

    with pytest.raises(RuntimeError, match="missingness-mechanism sensitivity"):
        benchmark_validation.raise_for_failed_required_checks(checks)


def test_benchmark_validation_rejects_heldout_context_shift_drift() -> None:
    checks = _checks()
    checks.loc[
        checks["check"] == "heldout context-shift coverage is complete",
        "status",
    ] = "FAIL"

    with pytest.raises(RuntimeError, match="heldout context-shift"):
        benchmark_validation.raise_for_failed_required_checks(checks)


def test_benchmark_validation_report_renders(tmp_path: Path) -> None:
    out_path = tmp_path / "validation_report.md"

    benchmark_validation.write_markdown_report(_checks(), out_path)

    assert "**Overall required-check status:** PASS" in out_path.read_text()
