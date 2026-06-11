from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.policy_rules import privacy_exposure_score
from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    evaluate_model,
    split_household_events,
)


SEEDS = [7, 42, 101]
TARGET_AVAILABILITY = 0.56
BASELINE_VARIANT = "signal_loss_baseline"
AGGREGATE_VARIANT = "privacy_safe_aggregates"

MECHANISMS = {
    "uniform_random": {
        "display_name": "Uniform random",
        "short_name": "MCAR-like",
        "color": "#2563eb",
    },
    "observed_context": {
        "display_name": "Observed-context conditioned",
        "short_name": "MAR-like",
        "color": "#0f766e",
    },
    "signal_dependent": {
        "display_name": "Signal-dependent",
        "short_name": "MNAR-like",
        "color": "#ea580c",
    },
}

VARIANTS = {
    BASELINE_VARIANT: "Signal-loss baseline",
    AGGREGATE_VARIANT: "Privacy-safe aggregates",
}


def _standardize(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    scale = array.std()
    if np.isclose(scale, 0.0):
        return np.zeros(len(array), dtype=float)
    return (array - array.mean()) / scale


def build_matched_availability_mask(
    events: pd.DataFrame,
    mechanism: str,
    target_share: float = TARGET_AVAILABILITY,
    seed: int = 42,
) -> np.ndarray:
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown missingness mechanism: {mechanism}")
    if not 0.0 < target_share < 1.0:
        raise ValueError("target_share must be between 0 and 1")

    rng = np.random.default_rng(seed)
    if mechanism == "uniform_random":
        priority = rng.random(len(events))
    elif mechanism == "observed_context":
        priority = (
            0.45 * _standardize(events["broadband_access"])
            - 0.45 * _standardize(events["underserved_score"])
            + 0.20 * events["consent_behavioral"].astype(float).to_numpy()
            + rng.gumbel(size=len(events))
        )
    else:
        priority = (
            0.70
            * _standardize(
                np.log1p(events["historical_service_engagement_count"])
            )
            + rng.gumbel(size=len(events))
        )

    mask = np.zeros(len(events), dtype=bool)
    if "household_id" in events.columns:
        _, test = split_household_events(events)
        test_positions = events.index.isin(test.index)
        partitions = [~test_positions, test_positions]
    else:
        partitions = [np.ones(len(events), dtype=bool)]

    for partition in partitions:
        positions = np.flatnonzero(partition)
        available_count = int(round(target_share * len(positions)))
        local_order = np.argsort(priority[positions], kind="stable")
        mask[positions[local_order[-available_count:]]] = True
    return mask


def apply_matched_signal_loss(
    events: pd.DataFrame,
    mechanism: str,
    target_share: float = TARGET_AVAILABILITY,
    seed: int = 42,
) -> pd.DataFrame:
    frame = events.copy()
    available = build_matched_availability_mask(
        frame,
        mechanism=mechanism,
        target_share=target_share,
        seed=seed,
    )
    frame["scenario"] = f"matched_{mechanism}"
    frame["behavioral_available"] = available
    frame["available_historical_service_engagement_count"] = np.where(
        available,
        frame["historical_service_engagement_count"],
        0,
    )
    frame["available_historical_engagement_count"] = np.where(
        available,
        frame["historical_engagement_count"],
        0,
    )
    frame["privacy_exposure_score"] = privacy_exposure_score(frame, available)
    return frame


def _availability_summary(frame: pd.DataFrame) -> dict:
    _, test = split_household_events(frame)
    low = test["low_signal"].astype(bool)
    low_share = test.loc[low, "behavioral_available"].mean()
    not_low_share = test.loc[~low, "behavioral_available"].mean()
    return {
        "behavioral_available_share": test["behavioral_available"].mean(),
        "low_signal_available_share": low_share,
        "not_low_signal_available_share": not_low_share,
        "availability_gap_not_low_minus_low": not_low_share - low_share,
        "num_test_events": int(len(test)),
    }


def run_mechanism_sensitivity(events: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []

    for mechanism, metadata in MECHANISMS.items():
        signal_limited = apply_matched_signal_loss(
            events,
            mechanism=mechanism,
            seed=seed,
        )
        availability = _availability_summary(signal_limited)

        for variant, display_name in VARIANTS.items():
            metrics = evaluate_model(
                signal_limited,
                f"{mechanism}_{variant}",
                (
                    BASE_NUMERIC_FEATURES
                    if variant == BASELINE_VARIANT
                    else PRIVACY_SAFE_NUMERIC_FEATURES
                ),
                privacy_safe_feature_options=(
                    None if variant == BASELINE_VARIANT else {"seed": seed}
                ),
            )
            rows.append(
                {
                    "seed": int(seed),
                    "mechanism": mechanism,
                    "mechanism_display_name": metadata["display_name"],
                    "mechanism_short_name": metadata["short_name"],
                    "variant": variant,
                    "variant_display_name": display_name,
                    "overall_ndcg_at_3": metrics["overall_ndcg_at_3"],
                    "low_signal_ndcg_at_3": metrics["low_signal_ndcg_at_3"],
                    "aggregate_reference_scope": metrics[
                        "aggregate_reference_scope"
                    ],
                    **availability,
                }
            )

    return pd.DataFrame(rows)


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    return run_mechanism_sensitivity(events, seed=seed)


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    baseline = raw[raw["variant"] == BASELINE_VARIANT][
        [
            "seed",
            "mechanism",
            "overall_ndcg_at_3",
            "low_signal_ndcg_at_3",
        ]
    ].rename(
        columns={
            "overall_ndcg_at_3": "baseline_overall_ndcg_at_3",
            "low_signal_ndcg_at_3": "baseline_low_signal_ndcg_at_3",
        }
    )
    paired = raw.merge(
        baseline,
        on=["seed", "mechanism"],
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
    metrics = [
        "behavioral_available_share",
        "low_signal_available_share",
        "not_low_signal_available_share",
        "availability_gap_not_low_minus_low",
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "overall_recovery",
        "low_signal_recovery",
    ]
    summary = (
        paired.groupby(
            [
                "mechanism",
                "mechanism_display_name",
                "mechanism_short_name",
                "variant",
                "variant_display_name",
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


def plot_mechanism_sensitivity(summary: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7))
    fig.patch.set_facecolor("#f8fafc")
    x = np.arange(len(MECHANISMS), dtype=float)
    baseline_rows = (
        summary[summary["variant"] == BASELINE_VARIANT]
        .set_index("mechanism")
        .loc[list(MECHANISMS)]
    )
    aggregate_rows = (
        summary[summary["variant"] == AGGREGATE_VARIANT]
        .set_index("mechanism")
        .loc[list(MECHANISMS)]
    )
    labels = [
        f"{metadata['short_name']}\n{metadata['display_name']}"
        for metadata in MECHANISMS.values()
    ]

    axes[0].bar(
        x - 0.17,
        baseline_rows["low_signal_available_share_mean"],
        width=0.34,
        color="#f59e0b",
        label="Low-signal",
    )
    axes[0].bar(
        x + 0.17,
        baseline_rows["not_low_signal_available_share_mean"],
        width=0.34,
        color="#334155",
        label="Not low-signal",
    )
    axes[0].axhline(
        TARGET_AVAILABILITY,
        color="#64748b",
        linewidth=1.2,
        linestyle="--",
    )
    axes[0].set_title("Matched quantity, different incidence", loc="left")
    axes[0].set_ylabel("Behavioral signal available share")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    for rows, marker, label in [
        (baseline_rows, "o", "Signal-loss baseline"),
        (aggregate_rows, "s", "Privacy-safe aggregates"),
    ]:
        axes[1].errorbar(
            x,
            rows["overall_ndcg_at_3_mean"],
            yerr=rows["overall_ndcg_at_3_std"],
            marker=marker,
            linewidth=2,
            capsize=4,
            label=label,
        )
    axes[1].set_title("Overall ranking utility", loc="left")
    axes[1].set_ylabel("NDCG@3, mean +/- std")
    axes[1].legend(frameon=False)

    axes[2].errorbar(
        x - 0.08,
        aggregate_rows["overall_recovery_mean"],
        yerr=aggregate_rows["overall_recovery_std"],
        marker="o",
        linewidth=2,
        capsize=4,
        label="Overall recovery",
    )
    axes[2].errorbar(
        x + 0.08,
        aggregate_rows["low_signal_recovery_mean"],
        yerr=aggregate_rows["low_signal_recovery_std"],
        marker="s",
        linewidth=2,
        capsize=4,
        label="Low-signal recovery",
    )
    axes[2].axhline(0.0, color="#334155", linewidth=1.2, linestyle="--")
    axes[2].set_title("Aggregate recovery is mechanism-sensitive", loc="left")
    axes[2].set_ylabel("Paired NDCG@3 recovery")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.set_facecolor("#f8fafc")
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=8.5)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Matched-rate missingness mechanism sensitivity",
        x=0.045,
        y=0.99,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.045,
        0.92,
        "All mechanisms retain 56% of behavioral events; only the incidence rule changes.",
        color="#475569",
        fontsize=10.5,
    )
    fig.text(
        0.045,
        0.02,
        "Mean +/- standard deviation across synthetic-data seeds 7, 42, and 101. "
        "Mechanism labels are controlled analogues, not empirical missingness diagnoses.",
        color="#64748b",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.09, 0.99, 0.87))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for mechanism in MECHANISMS:
        indexed = summary[summary["mechanism"] == mechanism].set_index("variant")
        baseline = indexed.loc[BASELINE_VARIANT]
        aggregate = indexed.loc[AGGREGATE_VARIANT]
        rows.append(
            {
                "Mechanism": (
                    f"{baseline['mechanism_short_name']}: "
                    f"{baseline['mechanism_display_name']}"
                ),
                "Overall availability": (
                    f"{baseline['behavioral_available_share_mean']:.1%}"
                ),
                "Low-signal availability": (
                    f"{baseline['low_signal_available_share_mean']:.1%}"
                ),
                "Not-low availability": (
                    f"{baseline['not_low_signal_available_share_mean']:.1%}"
                ),
                "Baseline NDCG@3": (
                    f"{baseline['overall_ndcg_at_3_mean']:.3f} +/- "
                    f"{baseline['overall_ndcg_at_3_std']:.3f}"
                ),
                "Aggregate recovery": (
                    f"{aggregate['overall_recovery_mean']:+.3f} +/- "
                    f"{aggregate['overall_recovery_std']:.3f}"
                ),
                "Low-signal recovery": (
                    f"{aggregate['low_signal_recovery_mean']:+.3f} +/- "
                    f"{aggregate['low_signal_recovery_std']:.3f}"
                ),
            }
        )
    indexed = summary.set_index(["mechanism", "variant"])
    random_baseline = indexed.loc[("uniform_random", BASELINE_VARIANT)]
    signal_baseline = indexed.loc[("signal_dependent", BASELINE_VARIANT)]

    out_path.write_text(
        "# Missingness-Mechanism Sensitivity\n\n"
        "This diagnostic holds the overall behavioral-signal availability rate at "
        "56% while changing which events retain signal. It separates signal "
        "quantity from incidence using controlled uniform-random, observed-context, "
        "and signal-dependent mechanisms.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Current result\n\n"
        "The overall holdout availability rate is fixed at 56% for every mechanism, "
        "but low-signal availability changes from "
        f"{random_baseline['low_signal_available_share_mean']:.1%} under the "
        "uniform-random mechanism to "
        f"{signal_baseline['low_signal_available_share_mean']:.1%} under the "
        "signal-dependent mechanism. The signal-dependent baseline has higher "
        "overall NDCG@3 because high-engagement events are preferentially retained, "
        "showing why matched aggregate availability does not imply matched subgroup "
        "incidence or comparable ranking difficulty.\n\n"
        "## Interpretation limits\n\n"
        "The MCAR-like, MAR-like, and MNAR-like labels describe synthetic mechanism "
        "analogues. They are not empirical missingness diagnoses, causal estimates, "
        "or implementations of inverse-propensity learning. The signal-dependent "
        "path intentionally uses the value later suppressed to create an "
        "adversarial incidence pattern.\n"
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

    raw.to_csv(
        tables_dir / "missingness_mechanism_sensitivity_raw.csv",
        index=False,
    )
    summary.to_csv(
        tables_dir / "missingness_mechanism_sensitivity_summary.csv",
        index=False,
    )
    plot_mechanism_sensitivity(
        summary,
        assets_dir / "missingness_mechanism_sensitivity.png",
    )
    write_markdown_summary(
        summary,
        docs_dir / "missingness_mechanism_sensitivity.md",
    )
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
