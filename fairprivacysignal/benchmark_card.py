from __future__ import annotations

from pathlib import Path

import pandas as pd


RECOVERY_ORDER = [
    "full_signal_raw_baseline",
    "severe_signal_loss_baseline",
    "severe_signal_loss_with_privacy_safe_aggregates",
    "policy_restricted_baseline",
    "policy_restricted_with_privacy_safe_aggregates",
]

RECOVERY_DISPLAY_NAMES = {
    "full_signal_raw_baseline": "Full signal raw baseline",
    "severe_signal_loss_baseline": "Severe signal loss",
    "severe_signal_loss_with_privacy_safe_aggregates": (
        "Severe loss + privacy-safe aggregates"
    ),
    "policy_restricted_baseline": "Policy restricted",
    "policy_restricted_with_privacy_safe_aggregates": (
        "Policy restricted + privacy-safe aggregates"
    ),
}


def _mean_std(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"


def _recovery_table(recovery_summary: pd.DataFrame) -> str:
    indexed = recovery_summary.set_index("experiment")
    rows = []

    for experiment in RECOVERY_ORDER:
        row = indexed.loc[experiment]
        rows.append(
            {
                "Scenario": RECOVERY_DISPLAY_NAMES[experiment],
                "Privacy exposure": _mean_std(row, "avg_privacy_exposure_score"),
                "NDCG@3": _mean_std(row, "overall_ndcg_at_3"),
                "Low-signal NDCG@3": _mean_std(row, "low_signal_ndcg_at_3"),
                "Low-signal gap": _mean_std(row, "ndcg_gap_not_low_minus_low"),
            }
        )

    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)


def _noise_checkpoint_table(noise_summary: pd.DataFrame) -> str:
    default_rows = noise_summary[noise_summary["noise_scale"] == 1.0].copy()
    rows = []

    for _, row in default_rows.sort_values("scenario").iterrows():
        rows.append(
            {
                "Scenario": row["display_name"],
                "Stress scale": f"{row['noise_scale']:.1f}",
                "Overall recovery": _mean_std(row, "overall_utility_recovery"),
                "Low-signal recovery": _mean_std(
                    row,
                    "low_signal_utility_recovery",
                ),
            }
        )

    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)


def _threshold_checkpoint_table(threshold_sensitivity: pd.DataFrame) -> str:
    selected = threshold_sensitivity[
        threshold_sensitivity["min_cohort_size"].isin([50, 200])
    ].copy()
    rows = []

    for _, row in selected.sort_values(["scenario", "min_cohort_size"]).iterrows():
        rows.append(
            {
                "Scenario": row["display_name"],
                "Minimum cohort size": int(row["min_cohort_size"]),
                "Fallback event share": f"{row['suppressed_event_share']:.1%}",
                "Overall recovery": f"{row['overall_utility_recovery']:+.3f}",
                "Low-signal recovery": f"{row['low_signal_utility_recovery']:+.3f}",
            }
        )

    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)


def build_benchmark_card(
    communities: pd.DataFrame,
    households: pd.DataFrame,
    events: pd.DataFrame,
    signal_loss: pd.DataFrame,
    multiseed_recovery_raw: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    multiseed_capacity_raw: pd.DataFrame,
    aggregate_noise_raw: pd.DataFrame,
    aggregate_noise_summary: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    recovery_feature_ablation_raw: pd.DataFrame,
    model_sensitivity_raw: pd.DataFrame,
    underserved_recovery_profile_raw: pd.DataFrame,
    validation_checks: pd.DataFrame,
) -> str:
    required_checks = validation_checks[validation_checks["required"].astype(bool)]
    informational_checks = validation_checks[
        ~validation_checks["required"].astype(bool)
    ]
    overall_status = (
        "PASS" if (required_checks["status"] == "PASS").all() else "FAIL"
    )

    scale_rows = [
        {"Item": "Synthetic communities", "Count": f"{len(communities):,}"},
        {"Item": "Synthetic households", "Count": f"{len(households):,}"},
        {"Item": "Synthetic household-service events", "Count": f"{len(events):,}"},
        {"Item": "Signal-loss scenarios", "Count": f"{signal_loss['scenario'].nunique():,}"},
        {
            "Item": "Privacy-recovery experiments",
            "Count": f"{recovery_summary['experiment'].nunique():,}",
        },
    ]
    coverage_rows = [
        {
            "Diagnostic": "Privacy-recovery robustness",
            "Coverage": (
                f"{multiseed_recovery_raw['seed'].nunique()} synthetic-data seeds"
            ),
        },
        {
            "Diagnostic": "Allocation sensitivity",
            "Coverage": (
                f"{multiseed_capacity_raw['seed'].nunique()} seeds; "
                f"{multiseed_capacity_raw['capacity_rate'].nunique()} capacity levels; "
                f"{multiseed_capacity_raw['low_signal_floor_fraction'].nunique()} "
                "allocation-floor strengths"
            ),
        },
        {
            "Diagnostic": "Aggregate-noise sensitivity",
            "Coverage": (
                f"{aggregate_noise_raw['scenario'].nunique()} scenarios; "
                f"{aggregate_noise_raw['noise_scale'].nunique()} stress scales; "
                f"{aggregate_noise_raw['noise_seed'].nunique()} noise seeds"
            ),
        },
        {
            "Diagnostic": "Cohort-threshold sensitivity",
            "Coverage": (
                f"{threshold_sensitivity['scenario'].nunique()} scenarios; "
                f"{threshold_sensitivity['min_cohort_size'].nunique()} k-thresholds"
            ),
        },
        {
            "Diagnostic": "Recovery feature ablation",
            "Coverage": (
                f"{recovery_feature_ablation_raw['scenario'].nunique()} scenarios; "
                f"{recovery_feature_ablation_raw['variant'].nunique()} feature sets; "
                f"{recovery_feature_ablation_raw['seed'].nunique()} paired seeds"
            ),
        },
        {
            "Diagnostic": "Model sensitivity",
            "Coverage": (
                f"{model_sensitivity_raw['model'].nunique()} models; "
                f"{model_sensitivity_raw['experiment'].nunique()} scenarios; "
                f"{model_sensitivity_raw['seed'].nunique()} paired seeds"
            ),
        },
        {
            "Diagnostic": "Underserved quartile recovery",
            "Coverage": (
                f"{underserved_recovery_profile_raw['scenario'].nunique()} scenarios; "
                f"{underserved_recovery_profile_raw['underserved_quartile'].nunique()} "
                "community-context quartiles; "
                f"{underserved_recovery_profile_raw['seed'].nunique()} paired seeds"
            ),
        },
        {
            "Diagnostic": "Aggregate preprocessing scope",
            "Coverage": "training households only before holdout scoring",
        },
        {
            "Diagnostic": "Validation gate",
            "Coverage": (
                f"{len(required_checks)} required checks; "
                f"{len(informational_checks)} informational check"
            ),
        },
    ]

    return (
        "# FairPrivacySignal Benchmark Card\n\n"
        "This reviewer-facing card is generated by the reproducible benchmark "
        "pipeline from auditable synthetic-data outputs. It summarizes the current "
        "experimental surface without replacing the detailed methodology documents.\n\n"
        f"**Required-check status:** {overall_status}\n\n"
        "## Benchmark Scale\n\n"
        + pd.DataFrame(scale_rows).to_markdown(index=False)
        + "\n\n"
        "## Experimental Coverage\n\n"
        + pd.DataFrame(coverage_rows).to_markdown(index=False)
        + "\n\n"
        "## Multi-Seed Recovery Results\n\n"
        "Mean +/- standard deviation across synthetic-data and aggregate-noise "
        "seeds.\n\n"
        + _recovery_table(recovery_summary)
        + "\n\n"
        "## Aggregate-Noise Checkpoint\n\n"
        "Default DP-style aggregate-noise stress scale: `1.0`. This is a stress "
        "parameter, not a formal privacy budget.\n\n"
        + _noise_checkpoint_table(aggregate_noise_summary)
        + "\n\n"
        "## Cohort-Threshold Checkpoints\n\n"
        "The default threshold is `k=50`; `k=200` shows a more restrictive fallback "
        "regime.\n\n"
        + _threshold_checkpoint_table(threshold_sensitivity)
        + "\n\n"
        "## Evidence Map\n\n"
        "- [Benchmark design](benchmark_design.md)\n"
        "- [Experiment matrix](experiment_matrix.md)\n"
        "- [Validation report](validation_report.md)\n"
        "- [Public-reference calibration](public_reference_calibration.md)\n"
        "- [Aggregate-noise sensitivity](aggregate_noise_sensitivity.md)\n"
        "- [Cohort-threshold sensitivity](cohort_threshold_sensitivity.md)\n"
        "- [Recovery feature ablation](recovery_feature_ablation.md)\n"
        "- [Model sensitivity diagnostic](model_sensitivity.md)\n"
        "- [Underserved quartile recovery profile](underserved_recovery_profile.md)\n"
        "- [Capacity-constrained allocation](capacity_allocation.md)\n"
        "- [Fairness metrics](fairness_metrics.md)\n"
        "- [Limitations](limitations.md)\n"
        "- [Reproducibility guide](reproducibility.md)\n\n"
        "## Interpretation Limits\n\n"
        "FairPrivacySignal is a synthetic engineering benchmark. It does not model "
        "a real community, provide a production privacy guarantee, prove that "
        "fairness gaps are solved, or establish that a particular threshold is "
        "appropriate for deployment. See [limitations.md](limitations.md) for the "
        "full scope statement.\n"
    )


def main() -> None:
    data_dir = Path("data/synthetic")
    tables_dir = Path("outputs/tables")
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    card = build_benchmark_card(
        communities=pd.read_csv(data_dir / "synthetic_communities.csv"),
        households=pd.read_csv(data_dir / "synthetic_households.csv"),
        events=pd.read_csv(data_dir / "synthetic_outreach_events.csv"),
        signal_loss=pd.read_csv(tables_dir / "signal_loss_summary.csv"),
        multiseed_recovery_raw=pd.read_csv(
            tables_dir / "multiseed_privacy_recovery_raw.csv"
        ),
        recovery_summary=pd.read_csv(
            tables_dir / "multiseed_privacy_recovery_summary.csv"
        ),
        multiseed_capacity_raw=pd.read_csv(
            tables_dir / "multiseed_capacity_sensitivity_raw.csv"
        ),
        aggregate_noise_raw=pd.read_csv(
            tables_dir / "aggregate_noise_sensitivity_raw.csv"
        ),
        aggregate_noise_summary=pd.read_csv(
            tables_dir / "aggregate_noise_sensitivity_summary.csv"
        ),
        threshold_sensitivity=pd.read_csv(
            tables_dir / "cohort_threshold_sensitivity.csv"
        ),
        recovery_feature_ablation_raw=pd.read_csv(
            tables_dir / "recovery_feature_ablation_raw.csv"
        ),
        model_sensitivity_raw=pd.read_csv(
            tables_dir / "model_sensitivity_raw.csv"
        ),
        underserved_recovery_profile_raw=pd.read_csv(
            tables_dir / "underserved_recovery_profile_raw.csv"
        ),
        validation_checks=pd.read_csv(
            tables_dir / "benchmark_validation_checks.csv"
        ),
    )

    out_path = docs_dir / "benchmark_card.md"
    out_path.write_text(card)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
