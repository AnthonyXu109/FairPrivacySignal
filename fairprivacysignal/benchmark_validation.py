from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


EXPECTED_SIGNAL_SCENARIOS = [
    "full_signal",
    "consent_restricted",
    "policy_restricted",
    "severe_signal_loss",
]


EXPECTED_RECOVERY_EXPERIMENTS = [
    "full_signal_raw_baseline",
    "severe_signal_loss_baseline",
    "severe_signal_loss_with_privacy_safe_aggregates",
    "severe_signal_loss_with_privacy_safe_fairness_aware",
    "policy_restricted_baseline",
    "policy_restricted_with_privacy_safe_aggregates",
    "policy_restricted_with_privacy_safe_fairness_aware",
]
EXPECTED_TRAIN_FITTED_RECOVERY_EXPERIMENTS = {
    "severe_signal_loss_with_privacy_safe_aggregates",
    "severe_signal_loss_with_privacy_safe_fairness_aware",
    "policy_restricted_with_privacy_safe_aggregates",
    "policy_restricted_with_privacy_safe_fairness_aware",
}


EXPECTED_SEEDS = {7, 11, 23, 42, 101}
EXPECTED_PUBLIC_REFERENCE_METRICS = {
    "median_household_income",
    "broadband_subscription_share",
}
EXPECTED_AGGREGATE_NOISE_SCENARIOS = {
    "severe_signal_loss",
    "policy_restricted",
}
EXPECTED_AGGREGATE_NOISE_SCALES = {0.0, 0.5, 1.0, 2.0, 4.0}
EXPECTED_AGGREGATE_NOISE_SEEDS = {7, 42, 101}
EXPECTED_COHORT_THRESHOLDS = {25, 50, 100, 200, 400, 800}
EXPECTED_ABLATION_VARIANTS = {
    "no_aggregate_substitutes",
    "engagement_aggregate_only",
    "cohort_context_aggregates_only",
    "combined_privacy_safe_aggregates",
}
EXPECTED_MODEL_SENSITIVITY_MODELS = {
    "logistic_regression",
    "hist_gradient_boosting",
}
EXPECTED_MODEL_SENSITIVITY_EXPERIMENTS = {
    "full_signal_raw_baseline",
    "severe_signal_loss_baseline",
    "severe_signal_loss_with_privacy_safe_aggregates",
    "policy_restricted_baseline",
    "policy_restricted_with_privacy_safe_aggregates",
}
EXPECTED_MODEL_SENSITIVITY_SEEDS = {7, 42, 101}
EXPECTED_PAIRWISE_RANKING_OBJECTIVES = {
    "pointwise_logistic",
    "linear_pairwise",
}
EXPECTED_PAIRWISE_RANKING_EXPERIMENTS = {
    "full_signal_raw_baseline",
    "severe_signal_loss_baseline",
    "severe_signal_loss_with_privacy_safe_aggregates",
    "policy_restricted_baseline",
    "policy_restricted_with_privacy_safe_aggregates",
}
EXPECTED_PAIRWISE_RANKING_SEEDS = {7, 42, 101}
EXPECTED_UNDERSERVED_PROFILE_VARIANTS = {
    "signal_loss_baseline",
    "privacy_safe_aggregates",
}
EXPECTED_UNDERSERVED_QUARTILES = {
    "q1_lower",
    "q2",
    "q3",
    "q4_higher",
}
EXPECTED_COMMUNITY_HOLDOUT_SPLITS = {
    "household_holdout",
    "community_holdout",
}
EXPECTED_COMMUNITY_HOLDOUT_VARIANTS = {
    "signal_loss_baseline",
    "privacy_safe_aggregates",
}


def _check(
    name: str,
    category: str,
    passed: bool,
    details: str,
    required: bool = True,
) -> dict:
    return {
        "check": name,
        "category": category,
        "required": required,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


def _columns_are_bounded(
    frame: pd.DataFrame,
    columns: List[str],
) -> bool:
    values = frame[columns].to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all())


def build_validation_checks(
    signal_loss: pd.DataFrame,
    recovery: pd.DataFrame,
    capacity: pd.DataFrame,
    score_calibration: pd.DataFrame,
    public_reference: pd.DataFrame,
    aggregate_noise_sensitivity_raw: pd.DataFrame,
    cohort_threshold_sensitivity: pd.DataFrame,
    recovery_feature_ablation_raw: pd.DataFrame,
    model_sensitivity_raw: pd.DataFrame,
    pairwise_ranking_sensitivity_raw: pd.DataFrame,
    underserved_recovery_profile_raw: pd.DataFrame,
    community_holdout_robustness_raw: pd.DataFrame,
    multiseed_recovery_raw: pd.DataFrame,
    multiseed_capacity_raw: pd.DataFrame,
) -> pd.DataFrame:
    checks = []

    signal_scenarios = set(signal_loss["scenario"])
    missing_signal_scenarios = sorted(
        set(EXPECTED_SIGNAL_SCENARIOS) - signal_scenarios
    )
    checks.append(
        _check(
            "signal-loss scenarios are complete",
            "signal loss",
            not missing_signal_scenarios,
            (
                "all four expected scenarios are present"
                if not missing_signal_scenarios
                else f"missing scenarios: {missing_signal_scenarios}"
            ),
        )
    )

    indexed_signal = signal_loss.set_index("scenario")
    if not missing_signal_scenarios:
        availability = indexed_signal.loc[
            EXPECTED_SIGNAL_SCENARIOS,
            "behavioral_available_share",
        ]
        exposure = indexed_signal.loc[
            EXPECTED_SIGNAL_SCENARIOS,
            "avg_privacy_exposure_score",
        ]
        event_counts = indexed_signal.loc[EXPECTED_SIGNAL_SCENARIOS, "num_events"]

        checks.append(
            _check(
                "signal-loss availability endpoints are correct",
                "signal loss",
                np.isclose(availability.iloc[0], 1.0)
                and np.isclose(availability.iloc[-1], 0.0),
                f"full={availability.iloc[0]:.3f}; severe={availability.iloc[-1]:.3f}",
            )
        )
        checks.append(
            _check(
                "signal-loss availability is monotonic",
                "signal loss",
                availability.is_monotonic_decreasing,
                " >= ".join(f"{value:.3f}" for value in availability),
            )
        )
        checks.append(
            _check(
                "privacy exposure is monotonic",
                "signal loss",
                exposure.is_monotonic_decreasing,
                " >= ".join(f"{value:.3f}" for value in exposure),
            )
        )
        checks.append(
            _check(
                "signal-loss scenarios use the same event count",
                "signal loss",
                event_counts.nunique() == 1 and event_counts.iloc[0] > 0,
                f"events per scenario={int(event_counts.iloc[0])}",
            )
        )

    recovery_experiments = set(recovery["experiment"])
    missing_recovery_experiments = sorted(
        set(EXPECTED_RECOVERY_EXPERIMENTS) - recovery_experiments
    )
    checks.append(
        _check(
            "privacy-recovery experiments are complete",
            "ranking",
            not missing_recovery_experiments,
            (
                "all seven expected recovery experiments are present"
                if not missing_recovery_experiments
                else f"missing experiments: {missing_recovery_experiments}"
            ),
        )
    )
    checks.append(
        _check(
            "privacy-recovery utility metrics are bounded",
            "ranking",
            _columns_are_bounded(
                recovery,
                [
                    "overall_auc",
                    "overall_ndcg_at_3",
                    "low_signal_ndcg_at_3",
                    "not_low_signal_ndcg_at_3",
                ],
            ),
            "AUC and NDCG metrics are finite values in [0, 1]",
        )
    )
    recovery_reference_scope = recovery.set_index("experiment")[
        "aggregate_reference_scope"
    ]
    expected_reference_scope = {
        experiment: (
            "train_households_only"
            if experiment in EXPECTED_TRAIN_FITTED_RECOVERY_EXPERIMENTS
            else "not_applicable"
        )
        for experiment in EXPECTED_RECOVERY_EXPERIMENTS
    }
    checks.append(
        _check(
            "aggregate preprocessing is train-fitted",
            "reproducibility",
            recovery_reference_scope.to_dict() == expected_reference_scope,
            "aggregate features use training-household reference statistics before holdout scoring",
        )
    )
    checks.append(
        _check(
            "capacity allocation metrics are bounded",
            "allocation",
            _columns_are_bounded(
                capacity,
                [
                    "allocated_relevance_rate",
                    "low_signal_selection_rate",
                    "not_low_signal_selection_rate",
                    "allocated_low_signal_share",
                ],
            )
            and (capacity["num_allocated"] > 0).all()
            and (capacity["num_allocated"] <= capacity["num_candidate_events"]).all(),
            "allocation rates are in [0, 1] and allocated counts are valid",
        )
    )
    checks.append(
        _check(
            "score-matched calibration has eligible bins",
            "calibration",
            (score_calibration["num_matched_bins"] > 0).all()
            and _columns_are_bounded(
                score_calibration,
                [
                    "low_signal_ece",
                    "not_low_signal_ece",
                    "mean_absolute_matched_relevance_gap",
                ],
            ),
            "every reported scenario has eligible bins and bounded diagnostics",
        )
    )

    public_reference_metrics = set(public_reference["metric"])
    public_reference_values = public_reference[
        [
            "reference_value",
            "synthetic_value",
            "synthetic_minus_reference",
            "relative_gap_vs_reference",
            "synthetic_as_share_of_reference",
        ]
    ].to_numpy(dtype=float)
    checks.append(
        _check(
            "public-reference calibration targets are documented",
            "calibration",
            public_reference_metrics == EXPECTED_PUBLIC_REFERENCE_METRICS
            and np.isfinite(public_reference_values).all()
            and (public_reference["reference_value"] > 0).all()
            and public_reference["source_url"]
            .str.startswith("https://www.census.gov/")
            .all(),
            "tracked Census QuickFacts targets are present with finite comparison metrics",
        )
    )

    aggregate_noise_seed_counts = aggregate_noise_sensitivity_raw.groupby(
        ["scenario", "noise_scale"]
    )["noise_seed"].nunique()
    aggregate_noise_metrics = aggregate_noise_sensitivity_raw[
        [
            "baseline_overall_ndcg_at_3",
            "aggregate_overall_ndcg_at_3",
            "baseline_low_signal_ndcg_at_3",
            "aggregate_low_signal_ndcg_at_3",
        ]
    ]
    checks.append(
        _check(
            "aggregate-noise sensitivity coverage is complete",
            "calibration",
            set(aggregate_noise_sensitivity_raw["scenario"])
            == EXPECTED_AGGREGATE_NOISE_SCENARIOS
            and set(aggregate_noise_sensitivity_raw["noise_scale"])
            == EXPECTED_AGGREGATE_NOISE_SCALES
            and set(aggregate_noise_sensitivity_raw["noise_seed"])
            == EXPECTED_AGGREGATE_NOISE_SEEDS
            and (aggregate_noise_seed_counts == len(EXPECTED_AGGREGATE_NOISE_SEEDS)).all()
            and _columns_are_bounded(
                aggregate_noise_metrics,
                list(aggregate_noise_metrics.columns),
            ),
            "two scenarios cover five stress scales and three aggregate-noise seeds",
        )
    )

    threshold_metrics = cohort_threshold_sensitivity[
        [
            "suppressed_event_share",
            "suppressed_unique_cohort_share",
            "baseline_overall_ndcg_at_3",
            "aggregate_overall_ndcg_at_3",
            "baseline_low_signal_ndcg_at_3",
            "aggregate_low_signal_ndcg_at_3",
        ]
    ]
    threshold_coverage = cohort_threshold_sensitivity.groupby("scenario")[
        "min_cohort_size"
    ].apply(set)
    threshold_monotonicity = (
        cohort_threshold_sensitivity.sort_values("min_cohort_size")
        .groupby("scenario")["suppressed_event_share"]
        .apply(lambda values: values.is_monotonic_increasing)
    )
    checks.append(
        _check(
            "cohort-threshold sensitivity coverage is complete",
            "calibration",
            set(cohort_threshold_sensitivity["scenario"])
            == EXPECTED_AGGREGATE_NOISE_SCENARIOS
            and (threshold_coverage == EXPECTED_COHORT_THRESHOLDS).all()
            and threshold_monotonicity.all()
            and _columns_are_bounded(
                threshold_metrics,
                list(threshold_metrics.columns),
            ),
            "two scenarios cover six k-thresholds with monotonic fallback coverage",
        )
    )

    ablation_seed_counts = recovery_feature_ablation_raw.groupby(
        ["scenario", "variant"]
    )["seed"].nunique()
    ablation_metrics = recovery_feature_ablation_raw[
        [
            "overall_ndcg_at_3",
            "low_signal_ndcg_at_3",
        ]
    ]
    checks.append(
        _check(
            "recovery feature-ablation coverage is complete",
            "ranking",
            set(recovery_feature_ablation_raw["scenario"])
            == EXPECTED_AGGREGATE_NOISE_SCENARIOS
            and set(recovery_feature_ablation_raw["variant"])
            == EXPECTED_ABLATION_VARIANTS
            and set(recovery_feature_ablation_raw["seed"]) == EXPECTED_SEEDS
            and (ablation_seed_counts == len(EXPECTED_SEEDS)).all()
            and _columns_are_bounded(
                ablation_metrics,
                list(ablation_metrics.columns),
            ),
            "two scenarios cover four feature sets and five paired synthetic-data seeds",
        )
    )

    model_sensitivity_seed_counts = model_sensitivity_raw.groupby(
        ["model", "experiment"]
    )["seed"].nunique()
    model_sensitivity_metrics = model_sensitivity_raw[
        [
            "overall_auc",
            "overall_ndcg_at_3",
            "low_signal_ndcg_at_3",
        ]
    ]
    checks.append(
        _check(
            "model-sensitivity coverage is complete",
            "ranking",
            set(model_sensitivity_raw["model"])
            == EXPECTED_MODEL_SENSITIVITY_MODELS
            and set(model_sensitivity_raw["experiment"])
            == EXPECTED_MODEL_SENSITIVITY_EXPERIMENTS
            and set(model_sensitivity_raw["seed"])
            == EXPECTED_MODEL_SENSITIVITY_SEEDS
            and (
                model_sensitivity_seed_counts
                == len(EXPECTED_MODEL_SENSITIVITY_SEEDS)
            ).all()
            and _columns_are_bounded(
                model_sensitivity_metrics,
                list(model_sensitivity_metrics.columns),
            ),
            "two models cover five scenarios and three paired synthetic-data seeds",
        )
    )

    pairwise_ranking_seed_counts = pairwise_ranking_sensitivity_raw.groupby(
        ["objective", "experiment"]
    )["seed"].nunique()
    pairwise_ranking_metrics = pairwise_ranking_sensitivity_raw[
        [
            "overall_auc",
            "overall_ndcg_at_3",
            "low_signal_ndcg_at_3",
            "not_low_signal_ndcg_at_3",
        ]
    ]
    pairwise_rows = pairwise_ranking_sensitivity_raw[
        pairwise_ranking_sensitivity_raw["objective"] == "linear_pairwise"
    ]
    pointwise_rows = pairwise_ranking_sensitivity_raw[
        pairwise_ranking_sensitivity_raw["objective"] == "pointwise_logistic"
    ]
    aggregate_pairwise_rows = pairwise_ranking_sensitivity_raw[
        pairwise_ranking_sensitivity_raw["experiment"].isin(
            EXPECTED_TRAIN_FITTED_RECOVERY_EXPERIMENTS
        )
    ]
    checks.append(
        _check(
            "pairwise ranking-objective coverage is complete",
            "ranking",
            set(pairwise_ranking_sensitivity_raw["objective"])
            == EXPECTED_PAIRWISE_RANKING_OBJECTIVES
            and set(pairwise_ranking_sensitivity_raw["experiment"])
            == EXPECTED_PAIRWISE_RANKING_EXPERIMENTS
            and set(pairwise_ranking_sensitivity_raw["seed"])
            == EXPECTED_PAIRWISE_RANKING_SEEDS
            and (
                pairwise_ranking_seed_counts
                == len(EXPECTED_PAIRWISE_RANKING_SEEDS)
            ).all()
            and _columns_are_bounded(
                pairwise_ranking_metrics,
                list(pairwise_ranking_metrics.columns),
            )
            and (pairwise_rows["num_training_pairs"] > 0).all()
            and (pointwise_rows["num_training_pairs"] == 0).all()
            and (
                aggregate_pairwise_rows["aggregate_reference_scope"]
                == "train_households_only"
            ).all(),
            "two training objectives cover five scenarios and three paired seeds; pairwise samples are present",
        )
    )

    profile_seed_counts = underserved_recovery_profile_raw.groupby(
        ["scenario", "variant", "underserved_quartile"]
    )["seed"].nunique()
    profile_metrics = underserved_recovery_profile_raw[
        [
            "overall_ndcg_at_3",
            "low_signal_ndcg_at_3",
            "low_signal_share",
        ]
    ]
    checks.append(
        _check(
            "underserved-quartile recovery coverage is complete",
            "fairness",
            set(underserved_recovery_profile_raw["scenario"])
            == EXPECTED_AGGREGATE_NOISE_SCENARIOS
            and set(underserved_recovery_profile_raw["variant"])
            == EXPECTED_UNDERSERVED_PROFILE_VARIANTS
            and set(underserved_recovery_profile_raw["underserved_quartile"])
            == EXPECTED_UNDERSERVED_QUARTILES
            and set(underserved_recovery_profile_raw["seed"]) == EXPECTED_SEEDS
            and (profile_seed_counts == len(EXPECTED_SEEDS)).all()
            and _columns_are_bounded(
                profile_metrics,
                list(profile_metrics.columns),
            )
            and (underserved_recovery_profile_raw["num_test_events"] > 0).all()
            and (underserved_recovery_profile_raw["num_low_signal_events"] > 0).all()
            and (underserved_recovery_profile_raw["num_communities"] > 0).all(),
            "two scenarios cover two paired variants, four quartiles, and five synthetic-data seeds",
        )
    )

    community_holdout_seed_counts = community_holdout_robustness_raw.groupby(
        ["scenario", "split_strategy", "variant"]
    )["seed"].nunique()
    community_holdout_metrics = community_holdout_robustness_raw[
        [
            "overall_auc",
            "overall_ndcg_at_3",
            "low_signal_ndcg_at_3",
            "not_low_signal_ndcg_at_3",
            "fallback_event_share",
            "unseen_cohort_share",
            "heldout_community_share",
        ]
    ]
    community_holdout_rows = community_holdout_robustness_raw[
        community_holdout_robustness_raw["split_strategy"] == "community_holdout"
    ]
    aggregate_community_holdout_rows = community_holdout_robustness_raw[
        community_holdout_robustness_raw["variant"]
        == "privacy_safe_aggregates"
    ]
    checks.append(
        _check(
            "community-held-out robustness coverage is complete",
            "reproducibility",
            set(community_holdout_robustness_raw["scenario"])
            == EXPECTED_AGGREGATE_NOISE_SCENARIOS
            and set(community_holdout_robustness_raw["split_strategy"])
            == EXPECTED_COMMUNITY_HOLDOUT_SPLITS
            and set(community_holdout_robustness_raw["variant"])
            == EXPECTED_COMMUNITY_HOLDOUT_VARIANTS
            and set(community_holdout_robustness_raw["seed"]) == EXPECTED_SEEDS
            and (community_holdout_seed_counts == len(EXPECTED_SEEDS)).all()
            and _columns_are_bounded(
                community_holdout_metrics,
                list(community_holdout_metrics.columns),
            )
            and np.isclose(
                community_holdout_rows["heldout_community_share"],
                1.0,
            ).all()
            and (
                aggregate_community_holdout_rows["aggregate_reference_scope"]
                == "train_households_only"
            ).all()
            and (community_holdout_robustness_raw["num_test_events"] > 0).all()
            and (community_holdout_robustness_raw["num_test_households"] > 0).all()
            and (community_holdout_robustness_raw["num_test_communities"] > 0).all(),
            "two scenarios cover two split strategies, two paired variants, and five seeds; community holdout keeps evaluation communities disjoint",
        )
    )

    recovery_seed_counts = multiseed_recovery_raw.groupby("experiment")["seed"].nunique()
    checks.append(
        _check(
            "privacy-recovery multi-seed coverage is complete",
            "reproducibility",
            set(multiseed_recovery_raw["seed"]) == EXPECTED_SEEDS
            and (recovery_seed_counts == len(EXPECTED_SEEDS)).all(),
            f"seeds={sorted(multiseed_recovery_raw['seed'].unique())}",
        )
    )

    capacity_seed_counts = multiseed_capacity_raw.groupby(
        ["experiment", "capacity_rate", "low_signal_floor_fraction"]
    )["seed"].nunique()
    checks.append(
        _check(
            "allocation-sensitivity multi-seed coverage is complete",
            "reproducibility",
            set(multiseed_capacity_raw["seed"]) == EXPECTED_SEEDS
            and (capacity_seed_counts == len(EXPECTED_SEEDS)).all(),
            f"seeds={sorted(multiseed_capacity_raw['seed'].unique())}",
        )
    )

    recovery_index = recovery.set_index("experiment")
    severe_baseline = recovery_index.loc[
        "severe_signal_loss_baseline",
        "overall_ndcg_at_3",
    ]
    severe_aggregates = recovery_index.loc[
        "severe_signal_loss_with_privacy_safe_aggregates",
        "overall_ndcg_at_3",
    ]
    checks.append(
        _check(
            "privacy-safe aggregates recover severe-loss utility",
            "result diagnostic",
            severe_aggregates > severe_baseline,
            f"severe baseline={severe_baseline:.3f}; aggregates={severe_aggregates:.3f}",
            required=False,
        )
    )

    return pd.DataFrame(checks)


def write_markdown_report(checks: pd.DataFrame, out_path: Path) -> None:
    required = checks[checks["required"]]
    overall_status = "PASS" if (required["status"] == "PASS").all() else "FAIL"
    rows = checks.copy()
    rows["Required"] = rows["required"].map({True: "yes", False: "no"})
    rows["Status"] = rows["status"]
    rows = rows.rename(
        columns={
            "check": "Check",
            "category": "Category",
            "details": "Details",
        }
    )[["Check", "Category", "Required", "Status", "Details"]]

    out_path.write_text(
        "# Benchmark Validation Report\n\n"
        "This report is generated by the reproducible benchmark pipeline. Required "
        "checks enforce stable methodological invariants; informational checks expose "
        "current result behavior without blocking future experimentation.\n\n"
        f"**Overall required-check status:** {overall_status}\n\n"
        + rows.to_markdown(index=False)
        + "\n"
    )


def raise_for_failed_required_checks(checks: pd.DataFrame) -> None:
    failures = checks[(checks["required"]) & (checks["status"] != "PASS")]

    if not failures.empty:
        failed_names = ", ".join(failures["check"])
        raise RuntimeError(f"required benchmark validations failed: {failed_names}")


def main() -> None:
    tables_dir = Path("outputs/tables")
    docs_dir = Path("docs")

    checks = build_validation_checks(
        signal_loss=pd.read_csv(tables_dir / "signal_loss_summary.csv"),
        recovery=pd.read_csv(tables_dir / "privacy_recovery_metrics.csv"),
        capacity=pd.read_csv(tables_dir / "capacity_allocation_metrics.csv"),
        score_calibration=pd.read_csv(
            tables_dir / "score_matched_calibration_summary.csv"
        ),
        public_reference=pd.read_csv(
            tables_dir / "public_reference_calibration.csv"
        ),
        aggregate_noise_sensitivity_raw=pd.read_csv(
            tables_dir / "aggregate_noise_sensitivity_raw.csv"
        ),
        cohort_threshold_sensitivity=pd.read_csv(
            tables_dir / "cohort_threshold_sensitivity.csv"
        ),
        recovery_feature_ablation_raw=pd.read_csv(
            tables_dir / "recovery_feature_ablation_raw.csv"
        ),
        model_sensitivity_raw=pd.read_csv(
            tables_dir / "model_sensitivity_raw.csv"
        ),
        pairwise_ranking_sensitivity_raw=pd.read_csv(
            tables_dir / "pairwise_ranking_sensitivity_raw.csv"
        ),
        underserved_recovery_profile_raw=pd.read_csv(
            tables_dir / "underserved_recovery_profile_raw.csv"
        ),
        community_holdout_robustness_raw=pd.read_csv(
            tables_dir / "community_holdout_robustness_raw.csv"
        ),
        multiseed_recovery_raw=pd.read_csv(
            tables_dir / "multiseed_privacy_recovery_raw.csv"
        ),
        multiseed_capacity_raw=pd.read_csv(
            tables_dir / "multiseed_capacity_sensitivity_raw.csv"
        ),
    )

    csv_path = tables_dir / "benchmark_validation_checks.csv"
    markdown_path = docs_dir / "validation_report.md"

    checks.to_csv(csv_path, index=False)
    write_markdown_report(checks, markdown_path)

    print("Benchmark validation checks:")
    print(checks.to_string(index=False))
    print("\nWrote:")
    print(f"- {csv_path}")
    print(f"- {markdown_path}")

    raise_for_failed_required_checks(checks)


if __name__ == "__main__":
    main()
