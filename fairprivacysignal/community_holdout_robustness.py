from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import GroupShuffleSplit

from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    apply_train_fitted_privacy_safe_features,
    build_model,
    split_household_events,
)
from fairprivacysignal.privacy_transforms import COHORT_COLUMNS
from fairprivacysignal.ranking import average_ndcg_at_k, safe_auc
from fairprivacysignal.signal_loss import apply_signal_loss


SEEDS = [7, 11, 23, 42, 101]
BASELINE_VARIANT = "signal_loss_baseline"
AGGREGATE_VARIANT = "privacy_safe_aggregates"

SCENARIOS = {
    "severe_signal_loss": "Severe signal loss",
    "policy_restricted": "Policy restricted",
}

SPLIT_STRATEGIES = {
    "household_holdout": "Household holdout",
    "community_holdout": "Community holdout",
}

VARIANTS = {
    BASELINE_VARIANT: {
        "display_name": "Signal-loss baseline",
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    AGGREGATE_VARIANT: {
        "display_name": "Privacy-safe aggregates",
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
    },
}

COLORS = {
    "household_holdout": "#0f766e",
    "community_holdout": "#ea580c",
    "zero": "#334155",
    "ink": "#0f172a",
    "muted": "#64748b",
    "background": "#f8fafc",
}


def split_community_events(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_index, test_index = next(
        splitter.split(events, groups=events["community_id"])
    )
    train = events.iloc[train_index].copy()
    test = events.iloc[test_index].copy()
    return train, test


def calculate_unseen_cohort_share(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> float:
    reference_cohorts = train[COHORT_COLUMNS].drop_duplicates()
    matched = test.merge(
        reference_cohorts.assign(reference_cohort_seen=True),
        on=COHORT_COLUMNS,
        how="left",
    )
    return float(matched["reference_cohort_seen"].isna().mean())


def calculate_heldout_community_share(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> float:
    train_communities = set(train["community_id"])
    return float((~test["community_id"].isin(train_communities)).mean())


def score_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    scenario: str,
    split_strategy: str,
    variant: str,
    fallback_event_share: float,
    unseen_cohort_share: float,
    heldout_community_share: float,
) -> dict:
    numeric_features = VARIANTS[variant]["numeric_features"]
    features = numeric_features + [
        "service_category",
        "age_group",
        "income_band",
        "urbanicity",
    ]
    model = build_model(numeric_features)
    model.fit(train[features], train["relevant"])

    scored = test.copy()
    scored["predicted_relevance"] = model.predict_proba(scored[features])[:, 1]
    low_signal = scored[scored["low_signal"].astype(bool)]
    not_low_signal = scored[~scored["low_signal"].astype(bool)]

    return {
        "seed": int(seed),
        "scenario": scenario,
        "scenario_display_name": SCENARIOS[scenario],
        "split_strategy": split_strategy,
        "split_display_name": SPLIT_STRATEGIES[split_strategy],
        "variant": variant,
        "variant_display_name": VARIANTS[variant]["display_name"],
        "overall_auc": safe_auc(scored["relevant"], scored["predicted_relevance"]),
        "overall_ndcg_at_3": average_ndcg_at_k(scored, k=3),
        "low_signal_ndcg_at_3": average_ndcg_at_k(low_signal, k=3),
        "not_low_signal_ndcg_at_3": average_ndcg_at_k(not_low_signal, k=3),
        "fallback_event_share": fallback_event_share,
        "unseen_cohort_share": unseen_cohort_share,
        "heldout_community_share": heldout_community_share,
        "num_test_events": int(len(scored)),
        "num_test_households": int(scored["household_id"].nunique()),
        "num_test_communities": int(scored["community_id"].nunique()),
        "aggregate_reference_scope": (
            "train_households_only"
            if variant == AGGREGATE_VARIANT
            else "not_applicable"
        ),
    }


def run_community_holdout_robustness(
    events: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    frames = []
    splitters: dict[str, Callable[[pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]] = {
        "household_holdout": split_household_events,
        "community_holdout": split_community_events,
    }

    for scenario in SCENARIOS:
        signal_limited = apply_signal_loss(events, scenario)
        for split_strategy, splitter in splitters.items():
            train, test = splitter(signal_limited)
            unseen_cohort_share = calculate_unseen_cohort_share(train, test)
            heldout_community_share = calculate_heldout_community_share(train, test)
            safe_train, safe_test = apply_train_fitted_privacy_safe_features(
                train,
                test,
                privacy_safe_feature_options={"seed": seed},
            )
            fallback_event_share = float(safe_test["cohort_suppressed"].mean())

            frames.append(
                score_variant(
                    train,
                    test,
                    seed=seed,
                    scenario=scenario,
                    split_strategy=split_strategy,
                    variant=BASELINE_VARIANT,
                    fallback_event_share=fallback_event_share,
                    unseen_cohort_share=unseen_cohort_share,
                    heldout_community_share=heldout_community_share,
                )
            )
            frames.append(
                score_variant(
                    safe_train,
                    safe_test,
                    seed=seed,
                    scenario=scenario,
                    split_strategy=split_strategy,
                    variant=AGGREGATE_VARIANT,
                    fallback_event_share=fallback_event_share,
                    unseen_cohort_share=unseen_cohort_share,
                    heldout_community_share=heldout_community_share,
                )
            )

    return pd.DataFrame(frames)


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    return run_community_holdout_robustness(events, seed=seed)


def build_paired_recovery(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "seed",
        "scenario",
        "scenario_display_name",
        "split_strategy",
        "split_display_name",
    ]
    metrics = [
        "overall_auc",
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "not_low_signal_ndcg_at_3",
        "fallback_event_share",
        "unseen_cohort_share",
        "heldout_community_share",
        "num_test_events",
        "num_test_households",
        "num_test_communities",
    ]
    baseline = raw[raw["variant"] == BASELINE_VARIANT][keys + metrics].rename(
        columns={column: f"baseline_{column}" for column in metrics}
    )
    aggregates = raw[raw["variant"] == AGGREGATE_VARIANT][keys + metrics].rename(
        columns={column: f"aggregate_{column}" for column in metrics}
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


def build_summary(paired: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "baseline_overall_ndcg_at_3",
        "aggregate_overall_ndcg_at_3",
        "baseline_low_signal_ndcg_at_3",
        "aggregate_low_signal_ndcg_at_3",
        "overall_recovery",
        "low_signal_recovery",
        "aggregate_fallback_event_share",
        "aggregate_unseen_cohort_share",
        "aggregate_heldout_community_share",
    ]
    summary = (
        paired.groupby(
            [
                "scenario",
                "scenario_display_name",
                "split_strategy",
                "split_display_name",
            ]
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


def _plot_metric(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    percent: bool = False,
) -> None:
    x = np.arange(len(SCENARIOS), dtype=float)
    width = 0.34
    offsets = [-width / 2, width / 2]

    for offset, split_strategy in zip(offsets, SPLIT_STRATEGIES):
        indexed = (
            summary[summary["split_strategy"] == split_strategy]
            .set_index("scenario")
            .loc[list(SCENARIOS)]
        )
        mean = indexed[f"{metric}_mean"].to_numpy(dtype=float)
        std = indexed[f"{metric}_std"].to_numpy(dtype=float)
        bars = axis.bar(
            x + offset,
            mean,
            width=width,
            yerr=std,
            capsize=4,
            color=COLORS[split_strategy],
            alpha=0.90,
            label=SPLIT_STRATEGIES[split_strategy],
        )
        labels = [f"{value:.1%}" if percent else f"{value:+.3f}" for value in mean]
        axis.bar_label(bars, labels=labels, padding=4, fontsize=8.5)

    if not percent:
        axis.axhline(0.0, color=COLORS["zero"], linewidth=1.3, linestyle="--")
    else:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xticks(x, SCENARIOS.values())
    axis.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=COLORS["ink"])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.22)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)


def plot_community_holdout_robustness(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.7))
    fig.patch.set_facecolor(COLORS["background"])
    for axis in axes:
        axis.set_facecolor(COLORS["background"])

    _plot_metric(
        axes[0],
        summary,
        metric="overall_recovery",
        title="Overall ranking recovery",
        ylabel="Paired NDCG@3 recovery",
    )
    _plot_metric(
        axes[1],
        summary,
        metric="low_signal_recovery",
        title="Low-signal ranking recovery",
        ylabel="Paired low-signal NDCG@3 recovery",
    )
    _plot_metric(
        axes[2],
        summary,
        metric="aggregate_fallback_event_share",
        title="Broad fallback coverage",
        ylabel="Holdout events using service fallback",
        percent=True,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.91),
        frameon=False,
        ncol=2,
        fontsize=9.5,
    )
    fig.suptitle(
        "Community-held-out stress test: recovery under unseen contexts",
        x=0.04,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.04,
        0.905,
        "Training and evaluation communities are disjoint in the community-holdout path; the household-holdout baseline retains the primary protocol.",
        ha="left",
        fontsize=10,
        color=COLORS["muted"],
    )
    fig.text(
        0.04,
        0.025,
        "Mean +/- standard deviation across synthetic-data seeds 7, 11, 23, 42, and 101. "
        "This is a synthetic robustness diagnostic, not a geographic validation claim.",
        ha="left",
        fontsize=9,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.025, 0.10, 0.99, 0.84))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for scenario in SCENARIOS:
        indexed = summary[summary["scenario"] == scenario].set_index(
            "split_strategy"
        )
        for split_strategy in SPLIT_STRATEGIES:
            row = indexed.loc[split_strategy]
            rows.append(
                {
                    "Scenario": row["scenario_display_name"],
                    "Evaluation split": row["split_display_name"],
                    "Overall recovery": (
                        f"{row['overall_recovery_mean']:+.3f} +/- "
                        f"{row['overall_recovery_std']:.3f}"
                    ),
                    "Low-signal recovery": (
                        f"{row['low_signal_recovery_mean']:+.3f} +/- "
                        f"{row['low_signal_recovery_std']:.3f}"
                    ),
                    "Fallback event share": (
                        f"{row['aggregate_fallback_event_share_mean']:.1%}"
                    ),
                    "Unseen cohort share": (
                        f"{row['aggregate_unseen_cohort_share_mean']:.1%}"
                    ),
                    "Held-out community share": (
                        f"{row['aggregate_heldout_community_share_mean']:.1%}"
                    ),
                }
            )

    out_path.write_text(
        "# Community-Held-Out Robustness Diagnostic\n\n"
        "This diagnostic compares the primary household-level holdout with a stricter "
        "synthetic community-held-out stress test. In the stricter path, training and "
        "evaluation communities are disjoint. Privacy-safe aggregates remain fitted "
        "from training households only before scoring holdout events.\n\n"
        "Each recovery value is a paired aggregate-minus-baseline NDCG@3 difference "
        "for the same signal-loss scenario, split strategy, and synthetic-data seed. "
        "The table reports mean +/- standard deviation across five seeds.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Interpretation limits\n\n"
        "This is a synthetic grouped-holdout diagnostic. It checks whether the "
        "benchmark claim survives a stricter separation of generated community "
        "contexts, but it is not a real geographic, temporal, or deployment "
        "validation study. The household-level holdout remains the primary benchmark "
        "protocol.\n"
    )


def main(seeds: Iterable[int] = SEEDS) -> None:
    frames = []
    for seed in seeds:
        print(f"Running seed={seed}")
        frames.append(evaluate_seed(seed))

    raw = pd.concat(frames, ignore_index=True)
    paired = build_paired_recovery(raw)
    summary = build_summary(paired)

    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    docs_dir = Path("docs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    raw_path = tables_dir / "community_holdout_robustness_raw.csv"
    paired_path = tables_dir / "community_holdout_robustness_paired.csv"
    summary_path = tables_dir / "community_holdout_robustness_summary.csv"
    figure_path = assets_dir / "community_holdout_robustness.png"
    markdown_path = docs_dir / "community_holdout_robustness.md"

    raw.to_csv(raw_path, index=False)
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_community_holdout_robustness(summary, figure_path)
    write_markdown_summary(summary, markdown_path)

    print("\nCommunity-held-out robustness summary:")
    print(summary.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {paired_path}")
    print(f"- {summary_path}")
    print(f"- {figure_path}")
    print(f"- {markdown_path}")


if __name__ == "__main__":
    main()
