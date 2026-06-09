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
from fairprivacysignal.ranking import average_ndcg_at_k
from fairprivacysignal.signal_loss import apply_signal_loss


SEEDS = [7, 42, 101]
BASELINE_VARIANT = "no_aggregate_substitutes"
ALIGNED_VARIANT = "aligned_privacy_safe_aggregates"
PERMUTED_VARIANT = "service_permuted_aggregates"

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

VARIANTS = {
    BASELINE_VARIANT: {
        "display_name": "No aggregate substitutes",
        "numeric_features": BASE_NUMERIC_FEATURES,
        "alignment": "not_applicable",
    },
    ALIGNED_VARIANT: {
        "display_name": "Aligned aggregates",
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
        "alignment": "aligned",
    },
    PERMUTED_VARIANT: {
        "display_name": "Service-permuted aggregates",
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
        "alignment": "permuted",
    },
}

PLOT_VARIANTS = [ALIGNED_VARIANT, PERMUTED_VARIANT]


def permute_reference_service_categories(reference: pd.DataFrame) -> pd.DataFrame:
    categories = sorted(reference["service_category"].dropna().unique())
    if len(categories) < 2:
        raise ValueError("at least two service categories are required")

    rotated = categories[1:] + categories[:1]
    mapping = dict(zip(categories, rotated))
    permuted = reference.copy()
    permuted["service_category"] = permuted["service_category"].map(mapping)
    return permuted


def _score_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    scenario: str,
    variant: str,
) -> dict:
    metadata = VARIANTS[variant]
    numeric_features = metadata["numeric_features"]
    features = numeric_features + CATEGORICAL_FEATURES
    model = build_model(numeric_features)
    model.fit(train[features], train["relevant"])

    scored = test.copy()
    scored["predicted_relevance"] = model.predict_proba(scored[features])[:, 1]
    low_signal = scored[scored["low_signal"].astype(bool)]
    return {
        "seed": int(seed),
        "scenario": scenario,
        "scenario_display_name": SCENARIOS[scenario]["display_name"],
        "variant": variant,
        "variant_display_name": metadata["display_name"],
        "service_alignment": metadata["alignment"],
        "overall_ndcg_at_3": average_ndcg_at_k(scored, k=3),
        "low_signal_ndcg_at_3": average_ndcg_at_k(low_signal, k=3),
        "num_test_events": int(len(scored)),
        "aggregate_reference_scope": (
            "not_applicable"
            if variant == BASELINE_VARIANT
            else "train_households_only"
        ),
    }


def run_negative_control(events: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        signal_limited = apply_signal_loss(events, scenario)
        train, test = split_household_events(signal_limited)

        rows.append(
            _score_variant(
                train,
                test,
                seed=seed,
                scenario=scenario,
                variant=BASELINE_VARIANT,
            )
        )

        aligned_train = add_privacy_safe_features(
            train,
            reference_events=train,
            seed=seed,
        )
        aligned_test = add_privacy_safe_features(
            test,
            reference_events=train,
            seed=seed,
        )
        rows.append(
            _score_variant(
                aligned_train,
                aligned_test,
                seed=seed,
                scenario=scenario,
                variant=ALIGNED_VARIANT,
            )
        )

        permuted_reference = permute_reference_service_categories(train)
        permuted_train = add_privacy_safe_features(
            train,
            reference_events=permuted_reference,
            seed=seed,
        )
        permuted_test = add_privacy_safe_features(
            test,
            reference_events=permuted_reference,
            seed=seed,
        )
        rows.append(
            _score_variant(
                permuted_train,
                permuted_test,
                seed=seed,
                scenario=scenario,
                variant=PERMUTED_VARIANT,
            )
        )

    return pd.DataFrame(rows)


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    return run_negative_control(events, seed=seed)


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["overall_ndcg_at_3", "low_signal_ndcg_at_3"]
    baseline = raw[raw["variant"] == BASELINE_VARIANT][
        ["scenario", "seed"] + metrics
    ].rename(columns={metric: f"baseline_{metric}" for metric in metrics})
    paired = raw.merge(
        baseline,
        on=["scenario", "seed"],
        how="left",
        validate="many_to_one",
    )
    paired["overall_recovery"] = (
        paired["overall_ndcg_at_3"] - paired["baseline_overall_ndcg_at_3"]
    )
    paired["low_signal_recovery"] = (
        paired["low_signal_ndcg_at_3"]
        - paired["baseline_low_signal_ndcg_at_3"]
    )
    summary = (
        paired.groupby(
            [
                "scenario",
                "scenario_display_name",
                "variant",
                "variant_display_name",
            ]
        )[
            [
                "overall_ndcg_at_3",
                "low_signal_ndcg_at_3",
                "overall_recovery",
                "low_signal_recovery",
            ]
        ]
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
) -> None:
    x = np.arange(len(PLOT_VARIANTS), dtype=float)
    offsets = [-0.08, 0.08]
    for offset, (scenario, metadata) in zip(offsets, SCENARIOS.items()):
        rows = (
            summary[summary["scenario"] == scenario]
            .set_index("variant")
            .loc[PLOT_VARIANTS]
        )
        axis.errorbar(
            x + offset,
            rows[f"{metric}_mean"],
            yerr=rows[f"{metric}_std"],
            color=metadata["color"],
            marker="o",
            linewidth=2.1,
            markersize=7,
            capsize=4,
            label=metadata["display_name"],
        )
    axis.axhline(0.0, color="#334155", linewidth=1.3, linestyle="--")
    axis.set_xticks(
        x,
        [VARIANTS[variant]["display_name"] for variant in PLOT_VARIANTS],
    )
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold")
    axis.set_ylabel("Paired NDCG@3 recovery, mean +/- std")
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)


def plot_negative_control(summary: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    fig.patch.set_facecolor("#f8fafc")
    for axis in axes:
        axis.set_facecolor("#f8fafc")

    _plot_metric(axes[0], summary, "overall_recovery", "Overall recovery")
    _plot_metric(
        axes[1],
        summary,
        "low_signal_recovery",
        "Low-signal recovery",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.92),
        frameon=False,
        ncol=2,
        fontsize=9.5,
    )
    fig.suptitle(
        "Aggregate-alignment negative control",
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
        "Service permutation preserves train-only aggregate construction while breaking service-to-signal semantics.",
        ha="left",
        fontsize=10.2,
        color="#475569",
    )
    fig.text(
        0.055,
        0.035,
        "Mean +/- standard deviation across synthetic-data seeds 7, 42, and 101. "
        "The control is diagnostic and does not identify a causal mechanism.",
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
        for variant in VARIANTS:
            row = indexed.loc[variant]
            rows.append(
                {
                    "Scenario": row["scenario_display_name"],
                    "Variant": row["variant_display_name"],
                    "Overall NDCG@3": (
                        f"{row['overall_ndcg_at_3_mean']:.3f} +/- "
                        f"{row['overall_ndcg_at_3_std']:.3f}"
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
    severe = summary[summary["scenario"] == "severe_signal_loss"].set_index(
        "variant"
    )
    policy = summary[summary["scenario"] == "policy_restricted"].set_index("variant")
    out_path.write_text(
        "# Aggregate-Alignment Negative Control\n\n"
        "This diagnostic tests whether observed recovery depends on service-aligned "
        "aggregate structure. The negative-control path cyclically permutes service "
        "categories in the training reference before privacy-safe aggregates are "
        "constructed. Aggregate distributions and train-only fitting are retained, "
        "but service-to-signal semantics are deliberately broken.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Current result\n\n"
        "Under severe signal loss, mean overall recovery changes from "
        f"`{severe.loc[ALIGNED_VARIANT, 'overall_recovery_mean']:+.4f}` with "
        "aligned aggregates to "
        f"`{severe.loc[PERMUTED_VARIANT, 'overall_recovery_mean']:+.4f}` after "
        "service permutation. Under the policy-restricted scenario, mean overall "
        "recovery changes from "
        f"`{policy.loc[ALIGNED_VARIANT, 'overall_recovery_mean']:+.4f}` to "
        f"`{policy.loc[PERMUTED_VARIANT, 'overall_recovery_mean']:+.4f}`. The "
        "policy-restricted low-signal comparison is less separated, so the "
        "negative control should be read metric by metric rather than as a uniform "
        "effect.\n\n"
        "## Interpretation limits\n\n"
        "This is a synthetic structural negative control. A weaker permuted result "
        "supports the interpretation that service alignment matters in this benchmark, "
        "but it does not establish causality or transfer to a real deployment.\n"
    )


def main(seeds: Iterable[int] = SEEDS) -> None:
    raw = pd.concat([evaluate_seed(seed) for seed in seeds], ignore_index=True)
    summary = build_summary(raw)
    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    docs_dir = Path("docs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(tables_dir / "aggregate_alignment_negative_control_raw.csv", index=False)
    summary.to_csv(
        tables_dir / "aggregate_alignment_negative_control_summary.csv",
        index=False,
    )
    plot_negative_control(
        summary,
        assets_dir / "aggregate_alignment_negative_control.png",
    )
    write_markdown_summary(
        summary,
        docs_dir / "aggregate_alignment_negative_control.md",
    )
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
