from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairprivacysignal.capacity_allocation import score_experiment
from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
)
from fairprivacysignal.ranking import average_ndcg_at_k


SEEDS = [7, 11, 23, 42, 101]
BASELINE_VARIANT = "signal_loss_baseline"
AGGREGATE_VARIANT = "privacy_safe_aggregates"

SCENARIOS = {
    "severe_signal_loss": {
        "display_name": "Severe signal loss",
        "color": "#ea580c",
        "marker": "o",
    },
    "policy_restricted": {
        "display_name": "Policy restricted",
        "color": "#0f766e",
        "marker": "s",
    },
}

VARIANTS = {
    BASELINE_VARIANT: {
        "display_name": "Signal-loss baseline",
        "use_privacy_safe_features": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    AGGREGATE_VARIANT: {
        "display_name": "Privacy-safe aggregates",
        "use_privacy_safe_features": True,
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
    },
}

QUARTILES = {
    "q1_lower": "Q1 lower",
    "q2": "Q2",
    "q3": "Q3",
    "q4_higher": "Q4 higher",
}


def assign_underserved_quartiles(events: pd.DataFrame) -> pd.DataFrame:
    communities = (
        events[["community_id", "underserved_score"]]
        .drop_duplicates()
        .sort_values("community_id")
        .reset_index(drop=True)
    )
    if communities["community_id"].duplicated().any():
        raise ValueError("community_id must map to one underserved_score")
    if len(communities) < len(QUARTILES):
        raise ValueError("at least four communities are required for quartile analysis")

    # Ranking keeps quartile assignment stable if rounded scores contain ties.
    communities["underserved_quartile"] = pd.qcut(
        communities["underserved_score"].rank(method="first"),
        q=len(QUARTILES),
        labels=list(QUARTILES),
    )
    return communities


def summarize_scored_quartiles(
    scored: pd.DataFrame,
    quartiles: pd.DataFrame,
    seed: int,
    scenario: str,
    variant: str,
) -> pd.DataFrame:
    enriched = scored.merge(
        quartiles[["community_id", "underserved_quartile"]],
        on="community_id",
        how="left",
        validate="many_to_one",
    )
    if enriched["underserved_quartile"].isna().any():
        raise ValueError("every scored event must map to an underserved quartile")

    rows = []
    for quartile, quartile_display_name in QUARTILES.items():
        group = enriched[enriched["underserved_quartile"] == quartile]
        low_signal = group[group["low_signal"].astype(bool)]
        rows.append(
            {
                "seed": int(seed),
                "scenario": scenario,
                "scenario_display_name": SCENARIOS[scenario]["display_name"],
                "variant": variant,
                "variant_display_name": VARIANTS[variant]["display_name"],
                "underserved_quartile": quartile,
                "quartile_display_name": quartile_display_name,
                "overall_ndcg_at_3": average_ndcg_at_k(group, k=3),
                "low_signal_ndcg_at_3": average_ndcg_at_k(low_signal, k=3),
                "low_signal_share": group["low_signal"].mean(),
                "num_test_events": int(len(group)),
                "num_low_signal_events": int(len(low_signal)),
                "num_communities": int(group["community_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def run_underserved_recovery_profile(
    events: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    quartiles = assign_underserved_quartiles(events)
    frames = []

    for scenario in SCENARIOS:
        for variant, metadata in VARIANTS.items():
            scored = score_experiment(
                events,
                experiment_name=f"{scenario}_{variant}",
                signal_scenario=scenario,
                use_privacy_safe_features=metadata["use_privacy_safe_features"],
                numeric_features=metadata["numeric_features"],
                privacy_noise_seed=seed,
            )
            frames.append(
                summarize_scored_quartiles(
                    scored,
                    quartiles=quartiles,
                    seed=seed,
                    scenario=scenario,
                    variant=variant,
                )
            )

    return pd.concat(frames, ignore_index=True)


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    return run_underserved_recovery_profile(events, seed=seed)


def build_paired_recovery(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "seed",
        "scenario",
        "scenario_display_name",
        "underserved_quartile",
        "quartile_display_name",
    ]
    metric_columns = [
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "low_signal_share",
        "num_test_events",
        "num_low_signal_events",
        "num_communities",
    ]
    baseline = raw[raw["variant"] == BASELINE_VARIANT][
        keys + metric_columns
    ].rename(
        columns={column: f"baseline_{column}" for column in metric_columns}
    )
    aggregates = raw[raw["variant"] == AGGREGATE_VARIANT][
        keys + metric_columns
    ].rename(
        columns={column: f"aggregate_{column}" for column in metric_columns}
    )
    paired = baseline.merge(
        aggregates,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    paired["overall_recovery"] = (
        paired["aggregate_overall_ndcg_at_3"]
        - paired["baseline_overall_ndcg_at_3"]
    )
    paired["low_signal_recovery"] = (
        paired["aggregate_low_signal_ndcg_at_3"]
        - paired["baseline_low_signal_ndcg_at_3"]
    )
    return paired


def build_profile_summary(paired: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "baseline_overall_ndcg_at_3",
        "aggregate_overall_ndcg_at_3",
        "baseline_low_signal_ndcg_at_3",
        "aggregate_low_signal_ndcg_at_3",
        "baseline_low_signal_share",
        "overall_recovery",
        "low_signal_recovery",
    ]
    summary = (
        paired.groupby(
            [
                "scenario",
                "scenario_display_name",
                "underserved_quartile",
                "quartile_display_name",
            ],
            observed=False,
        )[metrics]
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
    x = np.arange(len(QUARTILES), dtype=float)
    offsets = [-0.07, 0.07]

    for offset, (scenario, metadata) in zip(offsets, SCENARIOS.items()):
        scenario_rows = (
            summary[summary["scenario"] == scenario]
            .set_index("underserved_quartile")
            .loc[list(QUARTILES)]
        )
        mean = scenario_rows[f"{metric}_mean"].to_numpy(dtype=float)
        std = scenario_rows[f"{metric}_std"].to_numpy(dtype=float)
        axis.errorbar(
            x + offset,
            mean,
            yerr=std,
            color=metadata["color"],
            marker=metadata["marker"],
            linewidth=2.3,
            elinewidth=1.5,
            capsize=4,
            markersize=7,
            label=metadata["display_name"],
        )

    axis.axhline(0.0, color="#334155", linewidth=1.4, linestyle="--")
    axis.set_xticks(x, QUARTILES.values())
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#0f172a")
    axis.set_ylabel("Paired NDCG@3 recovery, mean +/- std")
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)


def plot_underserved_recovery_profile(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.9), sharey=True)
    fig.patch.set_facecolor("#f8fafc")
    for axis in axes:
        axis.set_facecolor("#f8fafc")

    _plot_recovery_metric(
        axes[0],
        summary,
        metric="overall_recovery",
        title="Overall ranking recovery",
    )
    _plot_recovery_metric(
        axes[1],
        summary,
        metric="low_signal_recovery",
        title="Low-signal ranking recovery",
    )
    lower, upper = axes[0].get_ylim()
    margin = (upper - lower) * 0.08
    axes[0].set_ylim(lower - margin, upper + margin)
    for axis in axes:
        axis.axhspan(
            lower - margin,
            0.0,
            color="#fef2f2",
            alpha=0.78,
            zorder=0,
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
        "Recovery heterogeneity: pooled gains can hide quartile-specific regressions",
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
        "Synthetic communities are grouped by underserved score before comparing privacy-safe aggregates with the same-seed baseline.",
        ha="left",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(
        0.055,
        0.035,
        "Mean +/- standard deviation across synthetic-data seeds 7, 11, 23, 42, and 101. "
        "Negative values expose quartile-specific regressions that pooled averages can obscure.",
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
        indexed = summary[summary["scenario"] == scenario].set_index(
            "underserved_quartile"
        )
        for quartile in QUARTILES:
            row = indexed.loc[quartile]
            rows.append(
                {
                    "Scenario": row["scenario_display_name"],
                    "Underserved quartile": row["quartile_display_name"],
                    "Low-signal event share": (
                        f"{row['baseline_low_signal_share_mean']:.1%}"
                    ),
                    "Overall recovery": (
                        f"{row['overall_recovery_mean']:+.3f} +/- "
                        f"{row['overall_recovery_std']:.3f}"
                    ),
                    "Low-signal recovery": (
                        f"{row['low_signal_recovery_mean']:+.3f} +/- "
                        f"{row['low_signal_recovery_std']:.3f}"
                    ),
                }
            )

    out_path.write_text(
        "# Underserved Quartile Recovery Profile\n\n"
        "This diagnostic checks whether pooled ranking-recovery results hide uneven "
        "effects across synthetic community contexts. Communities are assigned to "
        "quartiles using their synthetic underserved score before the logistic "
        "primary baseline is evaluated.\n\n"
        "Each recovery value is a paired aggregate-minus-baseline NDCG@3 difference "
        "for the same signal-loss scenario and synthetic-data seed. Quartiles are "
        "formed from distinct synthetic communities, not weighted by event volume. "
        "The table reports mean +/- standard deviation across five seeds.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Interpretation limits\n\n"
        "This profile is a synthetic heterogeneity diagnostic. Positive pooled "
        "recovery should not be read as uniform benefit: a quartile can show a "
        "negative low-signal recovery delta in the same configuration. The quartiles "
        "are benchmark constructs, not real-world demographic groups, and the "
        "diagnostic does not establish domain-specific fairness.\n"
    )


def main(seeds: Iterable[int] = SEEDS) -> None:
    frames = []
    for seed in seeds:
        print(f"Running seed={seed}")
        frames.append(evaluate_seed(seed))

    raw = pd.concat(frames, ignore_index=True)
    paired = build_paired_recovery(raw)
    summary = build_profile_summary(paired)

    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    docs_dir = Path("docs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    raw_path = tables_dir / "underserved_recovery_profile_raw.csv"
    paired_path = tables_dir / "underserved_recovery_profile_paired.csv"
    summary_path = tables_dir / "underserved_recovery_profile_summary.csv"
    figure_path = assets_dir / "underserved_recovery_profile.png"
    markdown_path = docs_dir / "underserved_recovery_profile.md"

    raw.to_csv(raw_path, index=False)
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_underserved_recovery_profile(summary, figure_path)
    write_markdown_summary(summary, markdown_path)

    print("\nUnderserved quartile recovery summary:")
    print(summary.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {paired_path}")
    print(f"- {summary_path}")
    print(f"- {figure_path}")
    print(f"- {markdown_path}")


if __name__ == "__main__":
    main()
