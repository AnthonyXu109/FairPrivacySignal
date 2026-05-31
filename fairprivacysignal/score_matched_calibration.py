from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairprivacysignal.capacity_allocation import (
    DISPLAY_NAME,
    EXPERIMENTS,
    score_experiment,
)


FULL_SIGNAL = "full_signal_raw_baseline"
SEVERE_AGGREGATES = "severe_signal_loss_with_privacy_safe_aggregates"
POLICY_AGGREGATES = "policy_restricted_with_privacy_safe_aggregates"


GAP_SCENARIOS = [
    FULL_SIGNAL,
    POLICY_AGGREGATES,
    SEVERE_AGGREGATES,
]


COLORS = {
    FULL_SIGNAL: "#264653",
    POLICY_AGGREGATES: "#7C3AED",
    SEVERE_AGGREGATES: "#2A9D8F",
}


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)

    if weights[valid].sum() == 0:
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def compute_score_matched_bins(
    scored_events: pd.DataFrame,
    n_bins: int = 8,
    min_group_count: int = 30,
) -> pd.DataFrame:
    """
    Compare observed outcomes for low-signal and not-low-signal events in shared
    predicted-score bins.

    This is a lightweight score-matched calibration diagnostic. It is not a
    replacement for a formal ranking-fairness analysis.
    """
    required_columns = {
        "experiment",
        "predicted_relevance",
        "relevant",
        "low_signal",
    }
    missing = sorted(required_columns - set(scored_events.columns))

    if missing:
        raise ValueError(f"scored events are missing required columns: {missing}")

    working = scored_events.copy()
    unique_scores = working["predicted_relevance"].nunique()

    if unique_scores < 2:
        raise ValueError("score-matched calibration requires at least two scores")

    working["score_bin"] = pd.qcut(
        working["predicted_relevance"],
        q=min(n_bins, unique_scores),
        duplicates="drop",
    )

    rows = []
    experiment = working["experiment"].iloc[0]

    for order, (score_bin, group) in enumerate(
        working.groupby("score_bin", observed=True)
    ):
        low = group[group["low_signal"].astype(bool)]
        not_low = group[~group["low_signal"].astype(bool)]

        low_observed = low["relevant"].mean()
        not_low_observed = not_low["relevant"].mean()

        rows.append(
            {
                "experiment": experiment,
                "score_bin_order": order,
                "score_bin": str(score_bin),
                "score_bin_left": float(score_bin.left),
                "score_bin_right": float(score_bin.right),
                "score_bin_midpoint": float(score_bin.mid),
                "low_signal_count": int(len(low)),
                "not_low_signal_count": int(len(not_low)),
                "low_signal_mean_score": low["predicted_relevance"].mean(),
                "not_low_signal_mean_score": not_low["predicted_relevance"].mean(),
                "low_signal_observed_relevance": low_observed,
                "not_low_signal_observed_relevance": not_low_observed,
                "matched_relevance_gap_not_low_minus_low": (
                    not_low_observed - low_observed
                ),
                "matched_weight": int(min(len(low), len(not_low))),
                "eligible_for_matched_comparison": (
                    len(low) >= min_group_count and len(not_low) >= min_group_count
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_score_matched_bins(bins: pd.DataFrame) -> dict:
    eligible = bins[bins["eligible_for_matched_comparison"]].copy()

    low_ece = _weighted_average(
        (
            bins["low_signal_observed_relevance"]
            - bins["low_signal_mean_score"]
        ).abs(),
        bins["low_signal_count"],
    )
    not_low_ece = _weighted_average(
        (
            bins["not_low_signal_observed_relevance"]
            - bins["not_low_signal_mean_score"]
        ).abs(),
        bins["not_low_signal_count"],
    )
    mean_abs_gap = _weighted_average(
        eligible["matched_relevance_gap_not_low_minus_low"].abs(),
        eligible["matched_weight"],
    )
    signed_gap = _weighted_average(
        eligible["matched_relevance_gap_not_low_minus_low"],
        eligible["matched_weight"],
    )

    return {
        "experiment": bins["experiment"].iloc[0],
        "low_signal_ece": low_ece,
        "not_low_signal_ece": not_low_ece,
        "mean_absolute_matched_relevance_gap": mean_abs_gap,
        "signed_matched_relevance_gap_not_low_minus_low": signed_gap,
        "num_matched_bins": int(len(eligible)),
        "num_scored_events": int(
            bins["low_signal_count"].sum() + bins["not_low_signal_count"].sum()
        ),
    }


def run_score_matched_calibration(
    events: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bin_frames = []
    summaries = []

    for experiment_name, signal_scenario, use_privacy_safe_features, numeric_features in EXPERIMENTS:
        scored = score_experiment(
            events,
            experiment_name,
            signal_scenario,
            use_privacy_safe_features,
            numeric_features,
        )
        bins = compute_score_matched_bins(scored)
        bin_frames.append(bins)
        summaries.append(summarize_score_matched_bins(bins))

    return pd.concat(bin_frames, ignore_index=True), pd.DataFrame(summaries)


def _annotate_heatmap(ax: plt.Axes, values: pd.DataFrame) -> None:
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap="Blues")
    midpoint = (np.nanmin(values.to_numpy()) + np.nanmax(values.to_numpy())) / 2

    for row in range(len(values.index)):
        for col in range(len(values.columns)):
            value = values.iloc[row, col]
            ax.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > midpoint else "#16324F",
                fontsize=8,
                fontweight="bold",
            )

    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Diagnostic magnitude", fontsize=9)


def plot_score_matched_calibration(
    bins: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    observed_experiments = set(bins["experiment"])
    missing = sorted(set(GAP_SCENARIOS) - observed_experiments)

    if missing:
        raise ValueError(f"calibration results are missing scenarios: {missing}")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.2),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.25]},
    )
    fig.patch.set_facecolor("#F7FAFC")

    for ax in axes:
        ax.set_facecolor("white")

    policy = bins[
        (bins["experiment"] == POLICY_AGGREGATES)
        & bins["eligible_for_matched_comparison"]
    ]
    axes[0].plot(
        policy["low_signal_mean_score"],
        policy["low_signal_observed_relevance"],
        marker="o",
        linewidth=2.2,
        color="#D97706",
        label="Low signal",
    )
    axes[0].plot(
        policy["not_low_signal_mean_score"],
        policy["not_low_signal_observed_relevance"],
        marker="o",
        linewidth=2.2,
        color="#2563EB",
        label="Not low signal",
    )
    score_min = min(
        policy["low_signal_mean_score"].min(),
        policy["not_low_signal_mean_score"].min(),
    )
    score_max = max(
        policy["low_signal_mean_score"].max(),
        policy["not_low_signal_mean_score"].max(),
    )
    axes[0].plot(
        [score_min, score_max],
        [score_min, score_max],
        linestyle="--",
        linewidth=1.4,
        color="#64748B",
        label="Perfect calibration",
    )
    axes[0].set_xlabel("Mean predicted relevance")
    axes[0].set_ylabel("Observed relevance rate")
    axes[0].set_title(
        "Policy + privacy-safe calibration",
        loc="left",
        fontweight="bold",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(color="#D8E1E8", linewidth=0.8, alpha=0.8)
    axes[0].set_axisbelow(True)

    for experiment in GAP_SCENARIOS:
        scenario = bins[
            (bins["experiment"] == experiment)
            & bins["eligible_for_matched_comparison"]
        ]
        axes[1].plot(
            scenario["score_bin_midpoint"],
            scenario["matched_relevance_gap_not_low_minus_low"],
            marker="o",
            linewidth=2.1,
            markersize=5,
            color=COLORS[experiment],
            label=DISPLAY_NAME[experiment],
        )

    axes[1].axhline(0, color="#64748B", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Predicted-score bin midpoint")
    axes[1].set_ylabel("Observed relevance gap")
    axes[1].set_title(
        "Gap within matched-score bins",
        loc="left",
        fontweight="bold",
    )
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(color="#D8E1E8", linewidth=0.8, alpha=0.8)
    axes[1].set_axisbelow(True)

    order = [experiment[0] for experiment in EXPERIMENTS]
    ordered = summary.set_index("experiment").loc[order]
    heatmap = ordered[
        [
            "low_signal_ece",
            "not_low_signal_ece",
            "mean_absolute_matched_relevance_gap",
        ]
    ]
    _annotate_heatmap(axes[2], heatmap)
    axes[2].set_xticks(
        np.arange(3),
        ["Low-signal\nECE", "Not-low-signal\nECE", "Mean absolute\nmatched gap"],
    )
    axes[2].set_yticks(
        np.arange(len(order)),
        [DISPLAY_NAME[experiment] for experiment in order],
    )
    axes[2].set_title(
        "Calibration diagnostics by scenario",
        loc="left",
        fontweight="bold",
    )

    fig.suptitle(
        "Score-matched calibration: aggregate ranking metrics can hide within-score subgroup differences",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#16324F",
    )
    fig.text(
        0.06,
        0.01,
        "Single-seed diagnostic. Bins compare observed relevance for low-signal and not-low-signal events "
        "with similar predicted scores; this is not a formal fairness guarantee.",
        fontsize=9,
        color="#5B7083",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def main() -> None:
    data_dir = Path("data/synthetic")
    out_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_dir / "synthetic_outreach_events.csv")
    bins, summary = run_score_matched_calibration(events)

    bins_path = out_dir / "score_matched_calibration_bins.csv"
    summary_path = out_dir / "score_matched_calibration_summary.csv"
    figure_path = assets_dir / "score_matched_calibration.png"

    bins.to_csv(bins_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_score_matched_calibration(bins, summary, figure_path)

    print("Score-matched calibration summary:")
    print(summary.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {bins_path}")
    print(f"- {summary_path}")
    print(f"- {figure_path}")


if __name__ == "__main__":
    main()
