from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    evaluate_model,
)
from fairprivacysignal.privacy_transforms import add_privacy_safe_features
from fairprivacysignal.signal_loss import apply_signal_loss


COHORT_THRESHOLDS = [25, 50, 100, 200, 400, 800]
DEFAULT_COHORT_THRESHOLD = 50
ANNOTATED_THRESHOLDS = {50, 200, 800}
PRIVACY_NOISE_SEED = 42
PRIVACY_NOISE_SCALE = 1.0

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


def run_cohort_threshold_sensitivity(
    events: pd.DataFrame,
    cohort_thresholds: Iterable[int] = COHORT_THRESHOLDS,
) -> pd.DataFrame:
    rows = []

    for scenario, metadata in SCENARIOS.items():
        signal_limited = apply_signal_loss(events, scenario)
        baseline = evaluate_model(
            signal_limited,
            f"{scenario}_baseline",
            BASE_NUMERIC_FEATURES,
        )

        for min_cohort_size in cohort_thresholds:
            privacy_safe = add_privacy_safe_features(
                signal_limited,
                min_cohort_size=min_cohort_size,
                dp_noise_scale=PRIVACY_NOISE_SCALE,
                seed=PRIVACY_NOISE_SEED,
            )
            metrics = evaluate_model(
                privacy_safe,
                f"{scenario}_with_privacy_safe_aggregates",
                PRIVACY_SAFE_NUMERIC_FEATURES,
            )
            cohort_suppression = privacy_safe.groupby(
                [
                    "service_category",
                    "urbanicity",
                    "income_band",
                    "age_group",
                ],
                observed=False,
            )["cohort_suppressed"].first()

            rows.append(
                {
                    "scenario": scenario,
                    "display_name": metadata["display_name"],
                    "min_cohort_size": int(min_cohort_size),
                    "privacy_noise_scale": PRIVACY_NOISE_SCALE,
                    "privacy_noise_seed": PRIVACY_NOISE_SEED,
                    "suppressed_event_share": privacy_safe["cohort_suppressed"].mean(),
                    "suppressed_unique_cohort_share": cohort_suppression.mean(),
                    "baseline_overall_ndcg_at_3": baseline["overall_ndcg_at_3"],
                    "aggregate_overall_ndcg_at_3": metrics["overall_ndcg_at_3"],
                    "overall_utility_recovery": (
                        metrics["overall_ndcg_at_3"]
                        - baseline["overall_ndcg_at_3"]
                    ),
                    "baseline_low_signal_ndcg_at_3": baseline[
                        "low_signal_ndcg_at_3"
                    ],
                    "aggregate_low_signal_ndcg_at_3": metrics[
                        "low_signal_ndcg_at_3"
                    ],
                    "low_signal_utility_recovery": (
                        metrics["low_signal_ndcg_at_3"]
                        - baseline["low_signal_ndcg_at_3"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def _plot_recovery_tradeoff(
    axis: plt.Axes,
    results: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    for scenario, metadata in SCENARIOS.items():
        scenario_rows = results[results["scenario"] == scenario].sort_values(
            "min_cohort_size"
        )
        x = 100 * scenario_rows["suppressed_event_share"].to_numpy(dtype=float)
        y = scenario_rows[metric].to_numpy(dtype=float)

        axis.plot(
            x,
            y,
            color=metadata["color"],
            marker="o",
            markersize=6,
            linewidth=2.6,
            label=metadata["display_name"],
        )
        for suppressed_share, recovery, threshold in zip(
            x,
            y,
            scenario_rows["min_cohort_size"],
        ):
            if threshold not in ANNOTATED_THRESHOLDS:
                continue
            axis.annotate(
                f"k={threshold}",
                (suppressed_share, recovery),
                xytext=(5, 8 if scenario == "severe_signal_loss" else -13),
                textcoords="offset points",
                fontsize=7.5,
                color=metadata["color"],
            )

    axis.axhline(0, color="#334155", linestyle="--", linewidth=1.2, alpha=0.8)
    axis.set_xlabel("Events using service-level fallback (%)")
    axis.set_ylabel("NDCG@3 recovery versus no-aggregate baseline")
    axis.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color="#0f172a")
    axis.spines[["top", "right"]].set_visible(False)


def plot_cohort_threshold_sensitivity(
    results: pd.DataFrame,
    out_path: Path,
) -> None:
    coverage = (
        results[results["scenario"] == "severe_signal_loss"]
        .sort_values("min_cohort_size")
        .copy()
    )
    thresholds = coverage["min_cohort_size"].to_numpy(dtype=int)
    suppressed_event_share = 100 * coverage["suppressed_event_share"].to_numpy(
        dtype=float
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.8),
        gridspec_kw={"width_ratios": [0.9, 1.15, 1.15]},
    )
    fig.patch.set_facecolor("#f8fafc")
    for axis in axes:
        axis.set_facecolor("#f8fafc")

    axes[0].plot(
        thresholds,
        suppressed_event_share,
        color="#334155",
        marker="o",
        markersize=7,
        linewidth=2.8,
    )
    axes[0].fill_between(
        thresholds,
        suppressed_event_share,
        color="#94a3b8",
        alpha=0.20,
    )
    axes[0].axvline(
        DEFAULT_COHORT_THRESHOLD,
        color="#0f766e",
        linestyle=":",
        linewidth=1.5,
    )
    axes[0].text(
        DEFAULT_COHORT_THRESHOLD * 1.06,
        0.96,
        "default k=50",
        transform=axes[0].get_xaxis_transform(),
        color="#0f766e",
        fontsize=8.5,
        rotation=90,
        va="top",
    )
    for threshold, value in zip(thresholds, suppressed_event_share):
        axes[0].annotate(
            f"{value:.1f}%",
            (threshold, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#334155",
            fontweight="bold",
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(thresholds, labels=[str(value) for value in thresholds])
    axes[0].set_xlabel("Minimum cohort size (k)")
    axes[0].set_ylabel("Events using service-level fallback (%)")
    axes[0].set_title(
        "Fallback coverage",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )
    axes[0].spines[["top", "right"]].set_visible(False)

    _plot_recovery_tradeoff(
        axes[1],
        results,
        metric="overall_utility_recovery",
        title="Overall ranking recovery",
    )
    _plot_recovery_tradeoff(
        axes[2],
        results,
        metric="low_signal_utility_recovery",
        title="Low-signal ranking recovery",
    )
    axes[1].legend(frameon=False, fontsize=8.5, loc="best")

    fig.suptitle(
        "Cohort-threshold sensitivity of privacy-safe recovery",
        x=0.05,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.05,
        0.92,
        "Higher k-thresholds suppress more cohort aggregates and route more events "
        "to broad service-level fallback signals.",
        ha="left",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(
        0.05,
        0.035,
        "Synthetic dataset seed 42; DP-style aggregate-noise scale 1.0 and seed 42. "
        "Threshold sensitivity is a diagnostic, not a privacy guarantee.",
        ha="left",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=(0.03, 0.10, 0.99, 0.86))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    events = pd.read_csv("data/synthetic/synthetic_outreach_events.csv")
    results = run_cohort_threshold_sensitivity(events)

    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "cohort_threshold_sensitivity.csv"
    figure_path = assets_dir / "cohort_threshold_sensitivity.png"
    results.to_csv(csv_path, index=False)
    plot_cohort_threshold_sensitivity(results, figure_path)

    print("Cohort-threshold sensitivity:")
    print(results.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {csv_path}")
    print(f"- {figure_path}")


if __name__ == "__main__":
    main()
