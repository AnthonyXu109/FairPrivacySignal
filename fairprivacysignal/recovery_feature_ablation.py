from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    evaluate_model,
)
from fairprivacysignal.privacy_transforms import add_privacy_safe_features
from fairprivacysignal.signal_loss import apply_signal_loss


SEEDS = [7, 11, 23, 42, 101]
BASELINE_VARIANT = "no_aggregate_substitutes"

SCENARIOS = {
    "severe_signal_loss": {
        "display_name": "Severe signal loss",
        "color": "#ea580c",
    },
    "policy_restricted": {
        "display_name": "Policy restricted",
        "color": "#0f766e",
    },
}

COHORT_CONTEXT_FEATURES = [
    "privacy_safe_cohort_avg_underserved",
    "privacy_safe_cohort_avg_food_risk",
    "privacy_safe_cohort_avg_health_need",
    "privacy_safe_cohort_avg_housing_pressure",
]

ABLATIONS = {
    BASELINE_VARIANT: {
        "display_name": "No aggregate substitutes",
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "engagement_aggregate_only": {
        "display_name": "Engagement aggregate only",
        "numeric_features": BASE_NUMERIC_FEATURES
        + ["privacy_safe_engagement_signal"],
    },
    "cohort_context_aggregates_only": {
        "display_name": "Cohort context aggregates only",
        "numeric_features": BASE_NUMERIC_FEATURES + COHORT_CONTEXT_FEATURES,
    },
    "combined_privacy_safe_aggregates": {
        "display_name": "Combined privacy-safe aggregates",
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
    },
}

PLOT_VARIANTS = [
    "engagement_aggregate_only",
    "cohort_context_aggregates_only",
    "combined_privacy_safe_aggregates",
]


def run_feature_ablation(
    events: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rows = []

    for scenario, scenario_metadata in SCENARIOS.items():
        signal_limited = apply_signal_loss(events, scenario)
        privacy_safe = add_privacy_safe_features(signal_limited, seed=seed)

        for variant, variant_metadata in ABLATIONS.items():
            frame = (
                signal_limited
                if variant == BASELINE_VARIANT
                else privacy_safe
            )
            metrics = evaluate_model(
                frame,
                f"{scenario}_{variant}",
                variant_metadata["numeric_features"],
            )
            rows.append(
                {
                    "seed": int(seed),
                    "scenario": scenario,
                    "scenario_display_name": scenario_metadata["display_name"],
                    "variant": variant,
                    "variant_display_name": variant_metadata["display_name"],
                    "overall_ndcg_at_3": metrics["overall_ndcg_at_3"],
                    "low_signal_ndcg_at_3": metrics["low_signal_ndcg_at_3"],
                    "ndcg_gap_not_low_minus_low": metrics[
                        "ndcg_gap_not_low_minus_low"
                    ],
                }
            )

    return pd.DataFrame(rows)


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    return run_feature_ablation(events, seed=seed)


def build_feature_ablation_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "ndcg_gap_not_low_minus_low",
    ]
    baseline = raw[raw["variant"] == BASELINE_VARIANT][
        ["scenario", "seed"] + metric_columns
    ].rename(
        columns={
            metric: f"baseline_{metric}"
            for metric in metric_columns
        }
    )
    paired = raw.merge(
        baseline,
        on=["scenario", "seed"],
        how="left",
        validate="many_to_one",
    )
    paired["overall_recovery_vs_no_aggregates"] = (
        paired["overall_ndcg_at_3"] - paired["baseline_overall_ndcg_at_3"]
    )
    paired["low_signal_recovery_vs_no_aggregates"] = (
        paired["low_signal_ndcg_at_3"]
        - paired["baseline_low_signal_ndcg_at_3"]
    )
    paired["gap_change_vs_no_aggregates"] = (
        paired["ndcg_gap_not_low_minus_low"]
        - paired["baseline_ndcg_gap_not_low_minus_low"]
    )

    summary_metrics = metric_columns + [
        "overall_recovery_vs_no_aggregates",
        "low_signal_recovery_vs_no_aggregates",
        "gap_change_vs_no_aggregates",
    ]
    summary = (
        paired.groupby(
            [
                "scenario",
                "scenario_display_name",
                "variant",
                "variant_display_name",
            ]
        )[summary_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary.fillna(0.0)


def _plot_recovery_metric(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    y_positions = np.arange(len(PLOT_VARIANTS), dtype=float)
    offsets = [-0.10, 0.10]

    for offset, (scenario, metadata) in zip(offsets, SCENARIOS.items()):
        scenario_rows = (
            summary[summary["scenario"] == scenario]
            .set_index("variant")
            .loc[PLOT_VARIANTS]
        )
        mean = scenario_rows[f"{metric}_mean"].to_numpy(dtype=float)
        std = scenario_rows[f"{metric}_std"].to_numpy(dtype=float)

        axis.errorbar(
            mean,
            y_positions + offset,
            xerr=std,
            fmt="o",
            color=metadata["color"],
            ecolor=metadata["color"],
            elinewidth=1.8,
            capsize=4,
            markersize=7,
            label=metadata["display_name"],
        )

    axis.axvline(0.0, color="#334155", linewidth=1.4, linestyle="--")
    axis.set_yticks(
        y_positions,
        [ABLATIONS[variant]["display_name"] for variant in PLOT_VARIANTS],
    )
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#0f172a")
    axis.set_xlabel("Paired NDCG@3 recovery vs no aggregates, mean +/- std")
    axis.grid(axis="x", alpha=0.25)
    axis.grid(axis="y", visible=False)
    axis.spines[["top", "right", "left"]].set_visible(False)


def plot_feature_ablation(summary: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    fig.patch.set_facecolor("#f8fafc")

    for axis in axes:
        axis.set_facecolor("#f8fafc")

    _plot_recovery_metric(
        axes[0],
        summary,
        metric="overall_recovery_vs_no_aggregates",
        title="Overall ranking recovery",
    )
    _plot_recovery_metric(
        axes[1],
        summary,
        metric="low_signal_recovery_vs_no_aggregates",
        title="Low-signal ranking recovery",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.925),
        ncol=2,
    )
    fig.suptitle(
        "Feature ablation: which privacy-safe substitutes recover ranking utility?",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.055,
        0.91,
        "Paired differences isolate the incremental value of engagement and cohort-context aggregates.",
        ha="left",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(
        0.055,
        0.035,
        "Mean +/- standard deviation across synthetic-data seeds 7, 11, 23, 42, and 101. "
        "Positive values indicate recovery relative to the same-seed no-aggregate baseline.",
        ha="left",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=(0.03, 0.11, 0.99, 0.84))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []

    for scenario in SCENARIOS:
        indexed = summary[summary["scenario"] == scenario].set_index("variant")
        for variant in ABLATIONS:
            row = indexed.loc[variant]
            rows.append(
                {
                    "Scenario": row["scenario_display_name"],
                    "Feature set": row["variant_display_name"],
                    "Overall NDCG@3": (
                        f"{row['overall_ndcg_at_3_mean']:.3f} +/- "
                        f"{row['overall_ndcg_at_3_std']:.3f}"
                    ),
                    "Overall recovery": (
                        f"{row['overall_recovery_vs_no_aggregates_mean']:+.3f} +/- "
                        f"{row['overall_recovery_vs_no_aggregates_std']:.3f}"
                    ),
                    "Low-signal recovery": (
                        f"{row['low_signal_recovery_vs_no_aggregates_mean']:+.3f} +/- "
                        f"{row['low_signal_recovery_vs_no_aggregates_std']:.3f}"
                    ),
                }
            )

    out_path.write_text(
        "# Recovery Feature Ablation\n\n"
        "This diagnostic separates the privacy-safe recovery layer into inspectable "
        "feature groups. It compares no aggregate substitutes, an engagement aggregate "
        "only, cohort-context aggregates only, and their combined use.\n\n"
        "Each recovery value is a paired difference against the no-aggregate baseline "
        "for the same signal-loss scenario and synthetic-data seed. The table reports "
        "mean +/- standard deviation across five seeds.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Interpretation limits\n\n"
        "The ablation isolates feature-group contributions within this synthetic "
        "benchmark. It does not prove that the same contributions will transfer to a "
        "real deployment, and it should not be interpreted as formal privacy "
        "accounting.\n"
    )


def main() -> None:
    frames = []

    for seed in SEEDS:
        print(f"Running seed={seed}")
        frames.append(evaluate_seed(seed))

    raw = pd.concat(frames, ignore_index=True)
    summary = build_feature_ablation_summary(raw)

    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    docs_dir = Path("docs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    raw_path = tables_dir / "recovery_feature_ablation_raw.csv"
    summary_path = tables_dir / "recovery_feature_ablation_summary.csv"
    figure_path = assets_dir / "recovery_feature_ablation.png"
    markdown_path = docs_dir / "recovery_feature_ablation.md"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_feature_ablation(summary, figure_path)
    write_markdown_summary(summary, markdown_path)

    print("\nRecovery feature-ablation summary:")
    print(summary.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    print(f"- {figure_path}")
    print(f"- {markdown_path}")


if __name__ == "__main__":
    main()
