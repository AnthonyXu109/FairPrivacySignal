from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


MULTISEED_SUMMARY_PATH = Path("outputs/tables/multiseed_privacy_recovery_summary.csv")
CAPACITY_METRICS_PATH = Path("outputs/tables/capacity_allocation_metrics.csv")
OVERVIEW_ASSET_PATH = Path("docs/assets/benchmark_overview.png")


FULL_SIGNAL = "full_signal_raw_baseline"
SEVERE_BASELINE = "severe_signal_loss_baseline"
SEVERE_AGGREGATES = "severe_signal_loss_with_privacy_safe_aggregates"
SEVERE_FAIRNESS = "severe_signal_loss_with_privacy_safe_fairness_aware"
POLICY_BASELINE = "policy_restricted_baseline"
POLICY_AGGREGATES = "policy_restricted_with_privacy_safe_aggregates"
POLICY_FAIRNESS = "policy_restricted_with_privacy_safe_fairness_aware"


MULTISEED_ORDER = [
    FULL_SIGNAL,
    SEVERE_BASELINE,
    SEVERE_AGGREGATES,
    SEVERE_FAIRNESS,
    POLICY_BASELINE,
    POLICY_AGGREGATES,
    POLICY_FAIRNESS,
]


CAPACITY_SCENARIOS = [
    FULL_SIGNAL,
    POLICY_AGGREGATES,
    SEVERE_AGGREGATES,
]


COLORS = {
    "ink": "#16324F",
    "muted": "#5B7083",
    "grid": "#D8E1E8",
    "full": "#264653",
    "baseline": "#9CA3AF",
    "aggregate": "#2A9D8F",
    "fairness": "#E76F51",
    "policy": "#457B9D",
    "capacity_full": "#264653",
    "capacity_policy": "#7C3AED",
    "capacity_severe": "#2A9D8F",
    "background": "#F7FAFC",
}


def _require_rows(
    df: pd.DataFrame,
    column: str,
    expected: list[str],
    table_name: str,
) -> None:
    observed = set(df[column])
    missing = [value for value in expected if value not in observed]

    if missing:
        raise ValueError(f"{table_name} is missing required rows: {missing}")


def load_metric_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    multiseed = pd.read_csv(MULTISEED_SUMMARY_PATH)
    capacity = pd.read_csv(CAPACITY_METRICS_PATH)
    return multiseed, capacity


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)


def _plot_privacy_utility(ax: plt.Axes, multiseed: pd.DataFrame) -> None:
    ordered = multiseed.set_index("experiment").loc[MULTISEED_ORDER]
    point_colors = [
        COLORS["full"],
        COLORS["baseline"],
        COLORS["aggregate"],
        COLORS["fairness"],
        COLORS["policy"],
        COLORS["aggregate"],
        COLORS["fairness"],
    ]

    ax.errorbar(
        ordered["avg_privacy_exposure_score_mean"],
        ordered["overall_ndcg_at_3_mean"],
        yerr=ordered["overall_ndcg_at_3_std"],
        fmt="none",
        ecolor=COLORS["muted"],
        elinewidth=1,
        capsize=3,
        alpha=0.8,
        zorder=2,
    )
    ax.scatter(
        ordered["avg_privacy_exposure_score_mean"],
        ordered["overall_ndcg_at_3_mean"],
        s=80,
        c=point_colors,
        edgecolor="white",
        linewidth=1,
        zorder=3,
    )

    for start, end in [
        (SEVERE_BASELINE, SEVERE_AGGREGATES),
        (SEVERE_AGGREGATES, SEVERE_FAIRNESS),
        (POLICY_BASELINE, POLICY_AGGREGATES),
        (POLICY_AGGREGATES, POLICY_FAIRNESS),
    ]:
        ax.annotate(
            "",
            xy=(
                ordered.loc[end, "avg_privacy_exposure_score_mean"],
                ordered.loc[end, "overall_ndcg_at_3_mean"],
            ),
            xytext=(
                ordered.loc[start, "avg_privacy_exposure_score_mean"],
                ordered.loc[start, "overall_ndcg_at_3_mean"],
            ),
            arrowprops={
                "arrowstyle": "->",
                "color": COLORS["ink"],
                "linewidth": 1.1,
                "shrinkA": 6,
                "shrinkB": 6,
            },
            zorder=2,
        )

    labels = {
        FULL_SIGNAL: ("Full signal", (-10, -16), "right"),
        SEVERE_BASELINE: ("Severe loss", (10, -12), "left"),
        SEVERE_AGGREGATES: ("+ aggregates", (10, 0), "left"),
        SEVERE_FAIRNESS: ("+ fairness-aware", (10, 13), "left"),
        POLICY_BASELINE: ("Policy restricted", (10, -12), "left"),
        POLICY_AGGREGATES: ("+ aggregates", (10, 0), "left"),
        POLICY_FAIRNESS: ("+ fairness-aware", (10, 14), "left"),
    }

    for experiment, (label, offset, alignment) in labels.items():
        ax.annotate(
            label,
            (
                ordered.loc[experiment, "avg_privacy_exposure_score_mean"],
                ordered.loc[experiment, "overall_ndcg_at_3_mean"],
            ),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            color=COLORS["ink"],
            fontsize=8.5,
        )

    ax.set_xlim(0.43, 0.97)
    ax.set_ylim(0.495, 0.572)
    ax.set_xlabel("Average privacy exposure score", color=COLORS["ink"], fontsize=10)
    ax.set_ylabel("Overall NDCG@3", color=COLORS["ink"], fontsize=10)
    ax.set_title(
        "Privacy exposure vs ranking utility",
        loc="left",
        color=COLORS["ink"],
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.96,
        "Multi-seed mean with standard deviation bars",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=8.5,
        va="top",
    )
    _style_axis(ax)
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.8, alpha=0.75)


def _plot_recovery_gain(ax: plt.Axes, multiseed: pd.DataFrame) -> None:
    indexed = multiseed.set_index("experiment")
    labels = [
        "Severe loss\n+ aggregates",
        "Severe loss\n+ fairness-aware",
        "Policy restricted\n+ aggregates",
        "Policy restricted\n+ fairness-aware",
    ]
    gains = [
        indexed.loc[SEVERE_AGGREGATES, "overall_ndcg_at_3_mean"]
        - indexed.loc[SEVERE_BASELINE, "overall_ndcg_at_3_mean"],
        indexed.loc[SEVERE_FAIRNESS, "overall_ndcg_at_3_mean"]
        - indexed.loc[SEVERE_BASELINE, "overall_ndcg_at_3_mean"],
        indexed.loc[POLICY_AGGREGATES, "overall_ndcg_at_3_mean"]
        - indexed.loc[POLICY_BASELINE, "overall_ndcg_at_3_mean"],
        indexed.loc[POLICY_FAIRNESS, "overall_ndcg_at_3_mean"]
        - indexed.loc[POLICY_BASELINE, "overall_ndcg_at_3_mean"],
    ]
    colors = [
        COLORS["aggregate"],
        COLORS["fairness"],
        COLORS["aggregate"],
        COLORS["fairness"],
    ]
    y = np.arange(len(labels))

    bars = ax.barh(y, gains, color=colors, height=0.62)

    for bar, value in zip(bars, gains):
        ax.text(
            value + 0.00035,
            bar.get_y() + bar.get_height() / 2,
            f"+{value:.3f}",
            va="center",
            color=COLORS["ink"],
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(gains) * 1.35)
    ax.set_xlabel("NDCG@3 gain vs matching baseline", color=COLORS["ink"], fontsize=10)
    ax.set_title(
        "Utility recovered after signal loss",
        loc="left",
        color=COLORS["ink"],
        fontsize=13,
        fontweight="bold",
    )
    _style_axis(ax)


def _plot_low_signal_gap(ax: plt.Axes, multiseed: pd.DataFrame) -> None:
    indexed = multiseed.set_index("experiment")
    families = [
        ("Severe signal loss", [SEVERE_BASELINE, SEVERE_AGGREGATES, SEVERE_FAIRNESS]),
        ("Policy restricted", [POLICY_BASELINE, POLICY_AGGREGATES, POLICY_FAIRNESS]),
    ]
    series = [
        ("Baseline", COLORS["baseline"]),
        ("Privacy-safe aggregates", COLORS["aggregate"]),
        ("Fairness-aware recovery", COLORS["fairness"]),
    ]

    for y, (_, experiments) in enumerate(families):
        values = [
            indexed.loc[experiment, "ndcg_gap_not_low_minus_low_mean"]
            for experiment in experiments
        ]
        ax.plot(values, [y] * len(values), color=COLORS["grid"], linewidth=3, zorder=1)

        label_offsets = (
            [(-16, -15), (0, -28), (16, -15)]
            if y == 0
            else [(-16, -15), (0, 12), (16, -15)]
        )
        for value, (label, color), label_offset in zip(
            values,
            series,
            label_offsets,
        ):
            ax.scatter(
                value,
                y,
                s=85,
                color=color,
                edgecolor="white",
                linewidth=1,
                zorder=3,
                label=label if y == 0 else None,
            )
            ax.annotate(
                f"{value:.3f}",
                (value, y),
                xytext=label_offset,
                textcoords="offset points",
                ha="center",
                color=COLORS["ink"],
                fontsize=8.5,
            )

    ax.set_yticks(np.arange(len(families)), [name for name, _ in families])
    ax.invert_yaxis()
    ax.set_xlim(0.100, 0.119)
    ax.set_xlabel("NDCG@3 gap: not-low-signal minus low-signal", color=COLORS["ink"], fontsize=10)
    ax.set_title(
        "Low-signal ranking gap diagnostic",
        loc="left",
        color=COLORS["ink"],
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.99,
        0.05,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        color=COLORS["muted"],
        fontsize=8.5,
        style="italic",
    )
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.58, 0.50),
        ncol=1,
        frameon=False,
        fontsize=8,
    )
    _style_axis(ax)


def _plot_capacity_tradeoff(ax: plt.Axes, capacity: pd.DataFrame) -> None:
    indexed = capacity.set_index(["experiment", "allocation_policy"])
    scenarios = [
        ("Full signal", FULL_SIGNAL, COLORS["capacity_full"], (7, -17)),
        ("Policy + privacy-safe", POLICY_AGGREGATES, COLORS["capacity_policy"], (7, 7)),
        ("Severe loss + privacy-safe", SEVERE_AGGREGATES, COLORS["capacity_severe"], (7, -10)),
    ]

    for label, experiment, color, label_offset in scenarios:
        utility = indexed.loc[(experiment, "utility_only")]
        constrained = indexed.loc[(experiment, "fairness_constrained")]
        start = (
            utility["allocated_relevance_rate"],
            utility["allocated_low_signal_share"],
        )
        end = (
            constrained["allocated_relevance_rate"],
            constrained["allocated_low_signal_share"],
        )

        if not np.allclose(start, end):
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 2,
                    "shrinkA": 5,
                    "shrinkB": 5,
                },
                zorder=2,
            )

        ax.scatter(
            *start,
            marker="o",
            s=75,
            color=color,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        ax.scatter(
            *end,
            marker="^",
            s=85,
            color=color,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )

        text_point = end if not np.allclose(start, end) else start
        ax.annotate(
            label,
            text_point,
            xytext=label_offset,
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            fontweight="bold",
        )

    policy_start = indexed.loc[(POLICY_AGGREGATES, "utility_only")]
    policy_end = indexed.loc[(POLICY_AGGREGATES, "fairness_constrained")]
    ax.text(
        0.97,
        0.95,
        "Policy + privacy-safe example:\n"
        f"{policy_start['allocated_low_signal_share']:.1%} → "
        f"{policy_end['allocated_low_signal_share']:.1%} low-signal share\n"
        f"{policy_start['allocated_relevance_rate']:.1%} → "
        f"{policy_end['allocated_relevance_rate']:.1%} allocated relevance",
        transform=ax.transAxes,
        va="top",
        ha="right",
        color=COLORS["ink"],
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F8FAFC",
            "edgecolor": COLORS["grid"],
        },
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS["ink"],
                markeredgecolor="white",
                markersize=7,
                label="Utility-only",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor=COLORS["ink"],
                markeredgecolor="white",
                markersize=8,
                label="Fairness-constrained",
            ),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8,
    )
    ax.set_xlim(0.38, 0.54)
    ax.set_ylim(-0.02, 0.42)
    ax.set_xlabel("Allocated relevance rate", color=COLORS["ink"], fontsize=10)
    ax.set_ylabel("Allocated low-signal share", color=COLORS["ink"], fontsize=10)
    ax.set_title(
        "Capacity allocation tradeoff",
        loc="left",
        color=COLORS["ink"],
        fontsize=13,
        fontweight="bold",
    )
    _style_axis(ax)
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.8, alpha=0.75)


def plot_benchmark_overview(
    multiseed: pd.DataFrame,
    capacity: pd.DataFrame,
    out_path: Path,
) -> None:
    _require_rows(multiseed, "experiment", MULTISEED_ORDER, "multi-seed summary")
    _require_rows(capacity, "experiment", CAPACITY_SCENARIOS, "capacity metrics")
    _require_rows(
        capacity,
        "allocation_policy",
        ["utility_only", "fairness_constrained"],
        "capacity metrics",
    )

    fig = plt.figure(figsize=(15, 10), facecolor=COLORS["background"])
    grid = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.97,
        bottom=0.10,
        top=0.84,
        wspace=0.25,
        hspace=0.42,
        width_ratios=[1.12, 1],
    )

    fig.text(
        0.07,
        0.965,
        "FairPrivacySignal benchmark overview",
        color=COLORS["ink"],
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.932,
        "Synthetic public-service ranking under privacy-driven signal loss",
        color=COLORS["muted"],
        fontsize=12,
    )

    badges = [
        ("5", "reproducible seeds"),
        ("7", "recovery scenarios"),
        ("15%", "outreach capacity"),
    ]
    for index, (value, label) in enumerate(badges):
        fig.text(
            0.59 + index * 0.13,
            0.945,
            f"{value}  {label}",
            color=COLORS["ink"],
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "edgecolor": COLORS["grid"],
            },
        )

    _plot_privacy_utility(fig.add_subplot(grid[0, 0]), multiseed)
    _plot_recovery_gain(fig.add_subplot(grid[0, 1]), multiseed)
    _plot_low_signal_gap(fig.add_subplot(grid[1, 0]), multiseed)
    _plot_capacity_tradeoff(fig.add_subplot(grid[1, 1]), capacity)

    fig.text(
        0.07,
        0.035,
        "All results use synthetic data. Privacy exposure is a comparative diagnostic proxy, "
        "not a formal privacy guarantee. Fairness gaps remain explicitly reported.",
        color=COLORS["muted"],
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    multiseed, capacity = load_metric_tables()
    plot_benchmark_overview(multiseed, capacity, OVERVIEW_ASSET_PATH)
    print(f"Wrote {OVERVIEW_ASSET_PATH}")


if __name__ == "__main__":
    main()
