from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    build_model,
    split_household_events,
)
from fairprivacysignal.privacy_transforms import add_privacy_safe_features
from fairprivacysignal.ranking import average_ndcg_at_k, safe_auc
from fairprivacysignal.signal_loss import apply_signal_loss


SEEDS = [7, 42, 101]
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

SHIFT_LEVELS = {
    "no_shift": {
        "display_name": "Reference",
        "strength": 0.0,
    },
    "moderate_shift": {
        "display_name": "Moderate",
        "strength": 0.5,
    },
    "pronounced_shift": {
        "display_name": "Pronounced",
        "strength": 1.0,
    },
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

CONTEXT_SHIFT_RULES = {
    "median_income": {
        "operation": "scale",
        "amount": -0.12,
        "lower": 1.0,
        "upper": None,
    },
    "unemployment_rate": {
        "operation": "add",
        "amount": 0.05,
        "lower": 0.01,
        "upper": 0.35,
    },
    "broadband_access": {
        "operation": "add",
        "amount": -0.10,
        "lower": 0.35,
        "upper": 0.99,
    },
    "food_access_risk": {
        "operation": "add",
        "amount": 0.08,
        "lower": 0.02,
        "upper": 0.95,
    },
    "health_need_score": {
        "operation": "add",
        "amount": 0.07,
        "lower": 0.02,
        "upper": 0.98,
    },
    "housing_pressure": {
        "operation": "add",
        "amount": 0.08,
        "lower": 0.02,
        "upper": 0.95,
    },
    "underserved_score": {
        "operation": "add",
        "amount": 0.10,
        "lower": 0.0,
        "upper": 1.0,
    },
}

CONTEXT_BUCKET_REMAP = {
    "income_band": {
        "low": "middle",
        "middle": "high",
        "high": "low",
    },
    "urbanicity": {
        "urban": "suburban",
        "suburban": "rural",
        "rural": "urban",
    },
}
MAX_BUCKET_MIGRATION_SHARE = 0.80

COLORS = {
    "zero": "#334155",
    "muted": "#64748b",
    "ink": "#0f172a",
    "background": "#f8fafc",
}


def apply_holdout_context_shift(
    events: pd.DataFrame,
    strength: float,
) -> pd.DataFrame:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    missing = sorted(set(CONTEXT_SHIFT_RULES) - set(events.columns))
    if missing:
        raise ValueError(f"missing context-shift columns: {missing}")

    shifted = events.copy()
    for column, rule in CONTEXT_SHIFT_RULES.items():
        if rule["operation"] == "scale":
            shifted[column] = shifted[column] * (1.0 + rule["amount"] * strength)
        elif rule["operation"] == "add":
            shifted[column] = shifted[column] + rule["amount"] * strength
        else:
            raise ValueError(f"unknown shift operation: {rule['operation']}")
        shifted[column] = shifted[column].clip(
            lower=rule["lower"],
            upper=rule["upper"],
        )

    household_hash = pd.util.hash_pandas_object(
        shifted["household_id"].astype(str),
        index=False,
    ).to_numpy(dtype=np.uint64)
    migration_mask = (
        (household_hash % 10000) / 10000.0
        < MAX_BUCKET_MIGRATION_SHARE * strength
    )
    for column, remap in CONTEXT_BUCKET_REMAP.items():
        original = shifted.loc[migration_mask, column]
        shifted.loc[migration_mask, column] = (
            original.map(remap).fillna(original)
        )

    shifted["context_shift_strength"] = float(strength)
    shifted["context_bucket_migrated"] = migration_mask
    return shifted


def fit_variant_model(train: pd.DataFrame, variant: str):
    numeric_features = VARIANTS[variant]["numeric_features"]
    features = numeric_features + CATEGORICAL_FEATURES
    model = build_model(numeric_features)
    model.fit(train[features], train["relevant"])
    return model


def score_shifted_holdout(
    model,
    test: pd.DataFrame,
    seed: int,
    scenario: str,
    shift_level: str,
    variant: str,
    fallback_event_share: float,
    bucket_migration_share: float,
) -> dict:
    numeric_features = VARIANTS[variant]["numeric_features"]
    features = numeric_features + CATEGORICAL_FEATURES
    scored = test.copy()
    scored["predicted_relevance"] = model.predict_proba(scored[features])[:, 1]
    low_signal = scored[scored["low_signal"].astype(bool)]
    not_low_signal = scored[~scored["low_signal"].astype(bool)]

    return {
        "seed": int(seed),
        "scenario": scenario,
        "scenario_display_name": SCENARIOS[scenario]["display_name"],
        "shift_level": shift_level,
        "shift_display_name": SHIFT_LEVELS[shift_level]["display_name"],
        "shift_strength": SHIFT_LEVELS[shift_level]["strength"],
        "variant": variant,
        "variant_display_name": VARIANTS[variant]["display_name"],
        "overall_auc": safe_auc(scored["relevant"], scored["predicted_relevance"]),
        "overall_ndcg_at_3": average_ndcg_at_k(scored, k=3),
        "low_signal_ndcg_at_3": average_ndcg_at_k(low_signal, k=3),
        "not_low_signal_ndcg_at_3": average_ndcg_at_k(not_low_signal, k=3),
        "fallback_event_share": fallback_event_share,
        "bucket_migration_share": bucket_migration_share,
        "num_test_events": int(len(scored)),
        "num_test_households": int(scored["household_id"].nunique()),
        "aggregate_reference_scope": (
            "train_households_only"
            if variant == AGGREGATE_VARIANT
            else "not_applicable"
        ),
    }


def run_heldout_context_shift(
    events: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rows = []

    for scenario in SCENARIOS:
        signal_limited = apply_signal_loss(events, scenario)
        train, test = split_household_events(signal_limited)

        # Aggregate statistics are fitted once on unshifted training households.
        safe_train = add_privacy_safe_features(
            train,
            reference_events=train,
            seed=seed,
        )
        safe_train["aggregate_reference_scope"] = "train_households_only"
        models = {
            BASELINE_VARIANT: fit_variant_model(train, BASELINE_VARIANT),
            AGGREGATE_VARIANT: fit_variant_model(safe_train, AGGREGATE_VARIANT),
        }

        for shift_level, shift_metadata in SHIFT_LEVELS.items():
            shifted_test = apply_holdout_context_shift(
                test,
                strength=shift_metadata["strength"],
            )
            safe_shifted_test = add_privacy_safe_features(
                shifted_test,
                reference_events=train,
                seed=seed,
            )
            safe_shifted_test["aggregate_reference_scope"] = (
                "train_households_only"
            )
            fallback_event_share = float(
                safe_shifted_test["cohort_suppressed"].mean()
            )
            bucket_migration_share = float(
                safe_shifted_test["context_bucket_migrated"].mean()
            )

            rows.append(
                score_shifted_holdout(
                    models[BASELINE_VARIANT],
                    shifted_test,
                    seed=seed,
                    scenario=scenario,
                    shift_level=shift_level,
                    variant=BASELINE_VARIANT,
                    fallback_event_share=fallback_event_share,
                    bucket_migration_share=bucket_migration_share,
                )
            )
            rows.append(
                score_shifted_holdout(
                    models[AGGREGATE_VARIANT],
                    safe_shifted_test,
                    seed=seed,
                    scenario=scenario,
                    shift_level=shift_level,
                    variant=AGGREGATE_VARIANT,
                    fallback_event_share=fallback_event_share,
                    bucket_migration_share=bucket_migration_share,
                )
            )

    return pd.DataFrame(rows)


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    return run_heldout_context_shift(events, seed=seed)


def build_paired_recovery(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "seed",
        "scenario",
        "scenario_display_name",
        "shift_level",
        "shift_display_name",
        "shift_strength",
    ]
    metrics = [
        "overall_auc",
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "not_low_signal_ndcg_at_3",
        "fallback_event_share",
        "bucket_migration_share",
        "num_test_events",
        "num_test_households",
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
        "aggregate_bucket_migration_share",
    ]
    summary = (
        paired.groupby(
            [
                "scenario",
                "scenario_display_name",
                "shift_level",
                "shift_display_name",
                "shift_strength",
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


def _plot_recovery_metric(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    for scenario, metadata in SCENARIOS.items():
        rows = (
            summary[summary["scenario"] == scenario]
            .set_index("shift_level")
            .loc[list(SHIFT_LEVELS)]
        )
        axis.errorbar(
            rows["shift_strength"],
            rows[f"{metric}_mean"],
            yerr=rows[f"{metric}_std"],
            color=metadata["color"],
            marker=metadata["marker"],
            linewidth=2.2,
            markersize=7,
            capsize=4,
            label=metadata["display_name"],
        )
    axis.axhline(0.0, color=COLORS["zero"], linewidth=1.3, linestyle="--")
    axis.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
    axis.set_ylabel(ylabel)


def _plot_absolute_utility(axis: plt.Axes, summary: pd.DataFrame) -> None:
    for scenario, metadata in SCENARIOS.items():
        rows = (
            summary[summary["scenario"] == scenario]
            .set_index("shift_level")
            .loc[list(SHIFT_LEVELS)]
        )
        axis.plot(
            rows["shift_strength"],
            rows["baseline_overall_ndcg_at_3_mean"],
            color=metadata["color"],
            marker=metadata["marker"],
            linewidth=1.7,
            linestyle="--",
            alpha=0.75,
            label=f"{metadata['display_name']} baseline",
        )
        axis.plot(
            rows["shift_strength"],
            rows["aggregate_overall_ndcg_at_3_mean"],
            color=metadata["color"],
            marker=metadata["marker"],
            linewidth=2.6,
            label=f"{metadata['display_name']} + aggregates",
        )
    axis.set_title("Absolute holdout utility", loc="left", fontsize=12.5, fontweight="bold")
    axis.set_ylabel("Overall NDCG@3, mean")
    axis.text(
        0.02,
        0.05,
        "solid = privacy-safe aggregates\ndashed = signal-loss baseline",
        transform=axis.transAxes,
        fontsize=8.5,
        color=COLORS["muted"],
        va="bottom",
    )


def plot_heldout_context_shift(summary: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.9))
    fig.patch.set_facecolor(COLORS["background"])
    for axis in axes:
        axis.set_facecolor(COLORS["background"])
        axis.set_xticks(
            [metadata["strength"] for metadata in SHIFT_LEVELS.values()],
            [metadata["display_name"] for metadata in SHIFT_LEVELS.values()],
        )
        axis.set_xlabel("Heldout context-shift strength")
        axis.grid(axis="y", alpha=0.22)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)

    _plot_recovery_metric(
        axes[0],
        summary,
        metric="overall_recovery",
        title="Overall ranking recovery",
        ylabel="Paired NDCG@3 recovery, mean +/- std",
    )
    _plot_recovery_metric(
        axes[1],
        summary,
        metric="low_signal_recovery",
        title="Low-signal ranking recovery",
        ylabel="Paired low-signal NDCG@3 recovery, mean +/- std",
    )
    _plot_absolute_utility(axes[2], summary)

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
        "Heldout context-shift stress test: recovery under controlled drift",
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
        "Training households and labels stay fixed; evaluation-side context covariates move and cohort buckets are deterministically remapped.",
        ha="left",
        fontsize=10,
        color=COLORS["muted"],
    )
    fig.text(
        0.04,
        0.025,
        "Mean +/- standard deviation across synthetic-data seeds 7, 42, and 101. "
        "This is a controlled covariate-drift proxy, not a temporal validation study.",
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
        indexed = summary[summary["scenario"] == scenario].set_index("shift_level")
        for shift_level in SHIFT_LEVELS:
            row = indexed.loc[shift_level]
            rows.append(
                {
                    "Scenario": row["scenario_display_name"],
                    "Heldout context shift": row["shift_display_name"],
                    "Baseline NDCG@3": (
                        f"{row['baseline_overall_ndcg_at_3_mean']:.3f} +/- "
                        f"{row['baseline_overall_ndcg_at_3_std']:.3f}"
                    ),
                    "Aggregate NDCG@3": (
                        f"{row['aggregate_overall_ndcg_at_3_mean']:.3f} +/- "
                        f"{row['aggregate_overall_ndcg_at_3_std']:.3f}"
                    ),
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
                    "Remapped context-bucket share": (
                        f"{row['aggregate_bucket_migration_share_mean']:.1%}"
                    ),
                }
            )

    out_path.write_text(
        "# Heldout Context-Shift Stress Test\n\n"
        "This paired diagnostic tests aggregate recovery when synthetic context "
        "covariates drift after the household-level train/test split. Training "
        "households, labels, and service candidates remain fixed. Only holdout-side "
        "income, unemployment, broadband access, food-access risk, health need, "
        "housing pressure, and underserved-score context features move across three "
        "controlled stress levels. A deterministic share of holdout household "
        "income-band and urbanicity buckets is also remapped so the aggregate layer "
        "must use different training-fitted cohort lookups.\n\n"
        "Privacy-safe aggregates remain fitted from unshifted training households "
        "only. Each recovery value is an aggregate-minus-baseline NDCG@3 difference "
        "for the same scenario, shift level, and synthetic-data seed. The table "
        "reports mean +/- standard deviation across three paired seeds.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Interpretation limits\n\n"
        "This is a synthetic covariate-drift proxy with fixed labels. It is not a "
        "temporal validation study, does not estimate real-world distribution shift, "
        "and does not establish deployment robustness. The diagnostic is intended to "
        "make one additional failure mode inspectable under controlled conditions.\n"
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

    raw_path = tables_dir / "heldout_context_shift_raw.csv"
    paired_path = tables_dir / "heldout_context_shift_paired.csv"
    summary_path = tables_dir / "heldout_context_shift_summary.csv"
    figure_path = assets_dir / "heldout_context_shift.png"
    markdown_path = docs_dir / "heldout_context_shift.md"

    raw.to_csv(raw_path, index=False)
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_heldout_context_shift(summary, figure_path)
    write_markdown_summary(summary, markdown_path)

    print("\nHeldout context-shift summary:")
    print(summary.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {paired_path}")
    print(f"- {summary_path}")
    print(f"- {figure_path}")
    print(f"- {markdown_path}")


if __name__ == "__main__":
    main()
