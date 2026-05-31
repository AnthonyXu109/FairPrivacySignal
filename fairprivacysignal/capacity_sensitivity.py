from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairprivacysignal.capacity_allocation import (
    DISPLAY_NAME,
    EXPERIMENTS,
    allocate_with_capacity,
    score_experiment,
    summarize_allocation,
)


CAPACITY_RATES = [0.05, 0.10, 0.15, 0.20, 0.30]
LOW_SIGNAL_FLOOR_FRACTIONS = [0.00, 0.25, 0.50, 0.75, 1.00]


FULL_SIGNAL = "full_signal_raw_baseline"
SEVERE_AGGREGATES = "severe_signal_loss_with_privacy_safe_aggregates"
POLICY_AGGREGATES = "policy_restricted_with_privacy_safe_aggregates"


FRONTIER_SCENARIOS = [
    FULL_SIGNAL,
    POLICY_AGGREGATES,
    SEVERE_AGGREGATES,
]


COLORS = {
    FULL_SIGNAL: "#264653",
    POLICY_AGGREGATES: "#7C3AED",
    SEVERE_AGGREGATES: "#2A9D8F",
}


def run_capacity_sensitivity(
    events: pd.DataFrame,
    experiments=None,
) -> pd.DataFrame:
    summaries = []
    selected_experiments = EXPERIMENTS if experiments is None else experiments

    for experiment_name, signal_scenario, use_privacy_safe_features, numeric_features in selected_experiments:
        scored = score_experiment(
            events,
            experiment_name,
            signal_scenario,
            use_privacy_safe_features,
            numeric_features,
        )

        for capacity_rate in CAPACITY_RATES:
            for floor_fraction in LOW_SIGNAL_FLOOR_FRACTIONS:
                allocated = allocate_with_capacity(
                    scored,
                    capacity_rate=capacity_rate,
                    allocation_policy="fairness_constrained",
                    low_signal_floor_fraction=floor_fraction,
                )
                summary = summarize_allocation(allocated)
                summary["capacity_rate"] = capacity_rate
                summary["low_signal_floor_fraction"] = floor_fraction
                summaries.append(summary)

    results = pd.DataFrame(summaries)
    baseline = (
        results[results["low_signal_floor_fraction"] == 0.0][
            ["experiment", "capacity_rate", "allocated_relevance_rate"]
        ]
        .rename(
            columns={
                "allocated_relevance_rate": "utility_only_allocated_relevance_rate"
            }
        )
    )
    results = results.merge(
        baseline,
        on=["experiment", "capacity_rate"],
        how="left",
    )
    results["allocated_relevance_cost_vs_utility_only"] = (
        results["utility_only_allocated_relevance_rate"]
        - results["allocated_relevance_rate"]
    )
    return results.sort_values(
        ["experiment", "capacity_rate", "low_signal_floor_fraction"]
    ).reset_index(drop=True)


def _annotated_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    title: str,
    colorbar_label: str,
    value_format: str,
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
                value_format.format(value),
                ha="center",
                va="center",
                color="white" if value > midpoint else "#16324F",
                fontsize=8,
                fontweight="bold",
            )

    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=9)


def plot_capacity_sensitivity(
    results: pd.DataFrame,
    out_path: Path,
) -> None:
    required_experiments = set(FRONTIER_SCENARIOS)
    observed_experiments = set(results["experiment"])
    missing = sorted(required_experiments - observed_experiments)

    if missing:
        raise ValueError(f"capacity sensitivity results are missing scenarios: {missing}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.patch.set_facecolor("#F7FAFC")

    for ax in axes:
        ax.set_facecolor("white")

    frontier = results[np.isclose(results["capacity_rate"], 0.15)]

    for experiment in FRONTIER_SCENARIOS:
        scenario = frontier[frontier["experiment"] == experiment].sort_values(
            "low_signal_floor_fraction"
        )
        axes[0].plot(
            scenario["allocated_low_signal_share"],
            scenario["allocated_relevance_rate"],
            marker="o",
            linewidth=2.3,
            markersize=6,
            color=COLORS[experiment],
            label=DISPLAY_NAME[experiment],
        )

        if experiment == POLICY_AGGREGATES:
            for _, row in scenario.iterrows():
                axes[0].annotate(
                    f"{100 * row['low_signal_floor_fraction']:.0f}%",
                    (
                        row["allocated_low_signal_share"],
                        row["allocated_relevance_rate"],
                    ),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=COLORS[experiment],
                )

    axes[0].set_xlabel("Allocated low-signal share")
    axes[0].set_ylabel("Allocated relevance rate")
    axes[0].set_title(
        "Allocation frontier at 15% outreach capacity",
        loc="left",
        fontweight="bold",
    )
    axes[0].text(
        0.02,
        0.04,
        "Labels show policy + privacy-safe floor strength",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color="#5B7083",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].grid(color="#D8E1E8", linewidth=0.8, alpha=0.8)
    axes[0].set_axisbelow(True)

    policy = results[results["experiment"] == POLICY_AGGREGATES]
    gap = policy.pivot(
        index="capacity_rate",
        columns="low_signal_floor_fraction",
        values="selection_rate_gap_not_low_minus_low",
    )
    relevance_cost = policy.pivot(
        index="capacity_rate",
        columns="low_signal_floor_fraction",
        values="allocated_relevance_cost_vs_utility_only",
    )

    _annotated_heatmap(
        axes[1],
        gap,
        "Policy + privacy-safe selection gap",
        "Selection-rate gap",
        "{:.3f}",
        "Blues",
    )
    _annotated_heatmap(
        axes[2],
        relevance_cost,
        "Policy + privacy-safe relevance cost",
        "Allocated relevance cost",
        "{:.3f}",
        "Oranges",
    )

    fig.suptitle(
        "Capacity sensitivity: stronger low-signal allocation floors reveal the utility-fairness frontier",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#16324F",
    )
    fig.text(
        0.06,
        0.01,
        "Single-seed sensitivity analysis. Lower selection-rate gaps are more balanced; "
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
    data_dir = Path("data/synthetic")
    out_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_dir / "synthetic_outreach_events.csv")
    results = run_capacity_sensitivity(events)

    metrics_path = out_dir / "capacity_sensitivity_metrics.csv"
    figure_path = assets_dir / "capacity_sensitivity_frontier.png"

    results.to_csv(metrics_path, index=False)
    plot_capacity_sensitivity(results, figure_path)

    print("Capacity sensitivity metrics:")
    print(results.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {metrics_path}")
    print(f"- {figure_path}")


if __name__ == "__main__":
    main()
