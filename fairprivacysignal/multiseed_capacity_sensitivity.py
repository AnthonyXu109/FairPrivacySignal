from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairprivacysignal.capacity_allocation import DISPLAY_NAME, EXPERIMENTS
from fairprivacysignal.capacity_sensitivity import (
    CAPACITY_RATES,
    COLORS,
    FRONTIER_SCENARIOS,
    LOW_SIGNAL_FLOOR_FRACTIONS,
    POLICY_AGGREGATES,
    run_capacity_sensitivity,
)
from fairprivacysignal.data_generator import generate_all


SEEDS = [7, 11, 23, 42, 101]
MULTISEED_EXPERIMENTS = [
    experiment
    for experiment in EXPERIMENTS
    if experiment[0] in FRONTIER_SCENARIOS
]


SUMMARY_METRICS = [
    "allocated_relevance_rate",
    "allocated_low_signal_share",
    "selection_rate_gap_not_low_minus_low",
    "allocated_relevance_cost_vs_utility_only",
]


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    results = run_capacity_sensitivity(
        events,
        experiments=MULTISEED_EXPERIMENTS,
    )
    results["seed"] = seed
    return results


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "experiment",
        "capacity_rate",
        "low_signal_floor_fraction",
    ]
    summary = results.groupby(group_columns)[SUMMARY_METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _annotated_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    title: str,
    colorbar_label: str,
    color_map: str,
) -> None:
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=color_map)

    ax.set_xticks(
        np.arange(len(pivot.columns)),
        [f"{100 * value:.0f}%" for value in pivot.columns],
    )
    ax.set_yticks(
        np.arange(len(pivot.index)),
        [f"{100 * value:.0f}%" for value in pivot.index],
    )
    ax.set_xlabel("Low-signal allocation floor strength")
    ax.set_ylabel("Outreach capacity")
    ax.set_title(title, loc="left", fontweight="bold")

    midpoint = (np.nanmin(pivot.to_numpy()) + np.nanmax(pivot.to_numpy())) / 2

    for row in range(len(pivot.index)):
        for col in range(len(pivot.columns)):
            value = pivot.iloc[row, col]
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
    colorbar.set_label(colorbar_label, fontsize=9)


def plot_multiseed_capacity_sensitivity(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    observed_experiments = set(summary["experiment"])
    missing = sorted(set(FRONTIER_SCENARIOS) - observed_experiments)

    if missing:
        raise ValueError(f"multi-seed allocation results are missing scenarios: {missing}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.patch.set_facecolor("#F7FAFC")

    for ax in axes:
        ax.set_facecolor("white")

    frontier = summary[np.isclose(summary["capacity_rate"], 0.15)]

    for experiment in FRONTIER_SCENARIOS:
        scenario = frontier[frontier["experiment"] == experiment].sort_values(
            "low_signal_floor_fraction"
        )
        axes[0].errorbar(
            scenario["allocated_low_signal_share_mean"],
            scenario["allocated_relevance_rate_mean"],
            xerr=scenario["allocated_low_signal_share_std"].fillna(0),
            yerr=scenario["allocated_relevance_rate_std"].fillna(0),
            marker="o",
            linewidth=2.3,
            markersize=6,
            capsize=3,
            color=COLORS[experiment],
            label=DISPLAY_NAME[experiment],
        )

        if experiment == POLICY_AGGREGATES:
            for _, row in scenario.iterrows():
                axes[0].annotate(
                    f"{100 * row['low_signal_floor_fraction']:.0f}%",
                    (
                        row["allocated_low_signal_share_mean"],
                        row["allocated_relevance_rate_mean"],
                    ),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=COLORS[experiment],
                )

    axes[0].set_xlabel("Allocated low-signal share, mean")
    axes[0].set_ylabel("Allocated relevance rate, mean")
    axes[0].set_title(
        "Multi-seed frontier at 15% capacity",
        loc="left",
        fontweight="bold",
    )
    axes[0].text(
        0.02,
        0.04,
        "Whiskers show ±1 std; labels show policy floor strength",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color="#5B7083",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].grid(color="#D8E1E8", linewidth=0.8, alpha=0.8)
    axes[0].set_axisbelow(True)

    policy = summary[summary["experiment"] == POLICY_AGGREGATES]
    gap = policy.pivot(
        index="capacity_rate",
        columns="low_signal_floor_fraction",
        values="selection_rate_gap_not_low_minus_low_mean",
    )
    relevance_cost = policy.pivot(
        index="capacity_rate",
        columns="low_signal_floor_fraction",
        values="allocated_relevance_cost_vs_utility_only_mean",
    )

    _annotated_heatmap(
        axes[1],
        gap,
        "Mean policy + privacy-safe selection gap",
        "Mean selection-rate gap",
        "Blues",
    )
    _annotated_heatmap(
        axes[2],
        relevance_cost,
        "Mean policy + privacy-safe relevance cost",
        "Mean allocated relevance cost",
        "Oranges",
    )

    fig.suptitle(
        "Multi-seed capacity sensitivity: allocation tradeoffs persist across synthetic draws",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#16324F",
    )
    fig.text(
        0.06,
        0.01,
        "Mean across five synthetic seeds. Lower selection-rate gaps are more balanced; "
        "relevance cost is measured against the matching utility-only allocation.",
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
    out_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for seed in SEEDS:
        print(f"Running allocation sensitivity seed={seed}")
        frames.append(evaluate_seed(seed))

    results = pd.concat(frames, ignore_index=True)
    summary = build_summary(results)

    raw_path = out_dir / "multiseed_capacity_sensitivity_raw.csv"
    summary_path = out_dir / "multiseed_capacity_sensitivity_summary.csv"
    figure_path = assets_dir / "multiseed_capacity_sensitivity.png"

    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_multiseed_capacity_sensitivity(summary, figure_path)

    print("\nMulti-seed allocation sensitivity summary:")
    print(summary.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    print(f"- {figure_path}")


if __name__ == "__main__":
    main()
