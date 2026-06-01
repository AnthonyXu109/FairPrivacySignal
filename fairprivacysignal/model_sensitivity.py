from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    build_model as build_logistic_model,
    evaluate_model,
)
from fairprivacysignal.privacy_transforms import add_privacy_safe_features
from fairprivacysignal.signal_loss import apply_signal_loss


SEEDS = [7, 42, 101]

EXPERIMENTS = {
    "full_signal_raw_baseline": {
        "display_name": "Full signal",
        "scenario": "full_signal",
        "use_privacy_safe_features": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "severe_signal_loss_baseline": {
        "display_name": "Severe loss",
        "scenario": "severe_signal_loss",
        "use_privacy_safe_features": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "severe_signal_loss_with_privacy_safe_aggregates": {
        "display_name": "Severe loss\n+ aggregates",
        "scenario": "severe_signal_loss",
        "use_privacy_safe_features": True,
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
    },
    "policy_restricted_baseline": {
        "display_name": "Policy restricted",
        "scenario": "policy_restricted",
        "use_privacy_safe_features": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "policy_restricted_with_privacy_safe_aggregates": {
        "display_name": "Policy restricted\n+ aggregates",
        "scenario": "policy_restricted",
        "use_privacy_safe_features": True,
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
    },
}

RECOVERY_PAIRS = {
    "severe_signal_loss": {
        "display_name": "Severe signal loss",
        "baseline": "severe_signal_loss_baseline",
        "aggregates": "severe_signal_loss_with_privacy_safe_aggregates",
    },
    "policy_restricted": {
        "display_name": "Policy restricted",
        "baseline": "policy_restricted_baseline",
        "aggregates": "policy_restricted_with_privacy_safe_aggregates",
    },
}


def build_hist_gradient_boosting_model(numeric_features: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )
    categorical_mask = [True] * len(CATEGORICAL_FEATURES) + [False] * len(
        numeric_features
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=42,
                    max_iter=100,
                    max_leaf_nodes=15,
                    class_weight="balanced",
                    categorical_features=categorical_mask,
                ),
            ),
        ]
    )


MODELS: Dict[str, Dict[str, object]] = {
    "logistic_regression": {
        "display_name": "Logistic baseline",
        "builder": build_logistic_model,
        "color": "#0f766e",
        "marker": "o",
    },
    "hist_gradient_boosting": {
        "display_name": "Histogram gradient boosting",
        "builder": build_hist_gradient_boosting_model,
        "color": "#ea580c",
        "marker": "s",
    },
}


def run_model_sensitivity(
    events: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    scenario_frames = {
        scenario: apply_signal_loss(events, scenario)
        for scenario in ["full_signal", "severe_signal_loss", "policy_restricted"]
    }
    privacy_safe_frames = {
        scenario: add_privacy_safe_features(frame, seed=seed)
        for scenario, frame in scenario_frames.items()
        if scenario != "full_signal"
    }
    rows = []

    for model_name, model_metadata in MODELS.items():
        builder = model_metadata["builder"]

        for experiment, experiment_metadata in EXPERIMENTS.items():
            scenario = experiment_metadata["scenario"]
            frame = (
                privacy_safe_frames[scenario]
                if experiment_metadata["use_privacy_safe_features"]
                else scenario_frames[scenario]
            )
            metrics = evaluate_model(
                frame,
                experiment,
                experiment_metadata["numeric_features"],
                model_builder=builder,
            )
            rows.append(
                {
                    "seed": int(seed),
                    "model": model_name,
                    "model_display_name": model_metadata["display_name"],
                    "experiment": experiment,
                    "experiment_display_name": experiment_metadata["display_name"],
                    "overall_auc": metrics["overall_auc"],
                    "overall_ndcg_at_3": metrics["overall_ndcg_at_3"],
                    "low_signal_ndcg_at_3": metrics["low_signal_ndcg_at_3"],
                    "ndcg_gap_not_low_minus_low": metrics[
                        "ndcg_gap_not_low_minus_low"
                    ],
                }
            )

    return pd.DataFrame(rows)


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    return run_model_sensitivity(events, seed=seed)


def build_model_sensitivity_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_auc",
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "ndcg_gap_not_low_minus_low",
    ]
    summary = (
        raw.groupby(
            [
                "model",
                "model_display_name",
                "experiment",
                "experiment_display_name",
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


def build_paired_recovery_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scenario, metadata in RECOVERY_PAIRS.items():
        baseline = raw[raw["experiment"] == metadata["baseline"]][
            [
                "seed",
                "model",
                "model_display_name",
                "overall_ndcg_at_3",
                "low_signal_ndcg_at_3",
            ]
        ].rename(
            columns={
                "overall_ndcg_at_3": "baseline_overall_ndcg_at_3",
                "low_signal_ndcg_at_3": "baseline_low_signal_ndcg_at_3",
            }
        )
        aggregates = raw[raw["experiment"] == metadata["aggregates"]][
            [
                "seed",
                "model",
                "overall_ndcg_at_3",
                "low_signal_ndcg_at_3",
            ]
        ].rename(
            columns={
                "overall_ndcg_at_3": "aggregate_overall_ndcg_at_3",
                "low_signal_ndcg_at_3": "aggregate_low_signal_ndcg_at_3",
            }
        )
        paired = baseline.merge(
            aggregates,
            on=["seed", "model"],
            how="inner",
            validate="one_to_one",
        )
        paired["scenario"] = scenario
        paired["scenario_display_name"] = metadata["display_name"]
        paired["overall_recovery"] = (
            paired["aggregate_overall_ndcg_at_3"]
            - paired["baseline_overall_ndcg_at_3"]
        )
        paired["low_signal_recovery"] = (
            paired["aggregate_low_signal_ndcg_at_3"]
            - paired["baseline_low_signal_ndcg_at_3"]
        )
        rows.append(paired)

    paired_raw = pd.concat(rows, ignore_index=True)
    summary = (
        paired_raw.groupby(
            [
                "model",
                "model_display_name",
                "scenario",
                "scenario_display_name",
            ]
        )[["overall_recovery", "low_signal_recovery"]]
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
    paired_recovery: pd.DataFrame,
    metric: str,
    recovery_metric: str,
    title: str,
) -> None:
    x = np.arange(len(EXPERIMENTS), dtype=float)

    for model_name, metadata in MODELS.items():
        model_rows = (
            summary[summary["model"] == model_name]
            .set_index("experiment")
            .loc[list(EXPERIMENTS)]
        )
        mean = model_rows[f"{metric}_mean"].to_numpy(dtype=float)
        std = model_rows[f"{metric}_std"].to_numpy(dtype=float)

        axis.errorbar(
            x,
            mean,
            yerr=std,
            color=metadata["color"],
            marker=metadata["marker"],
            linewidth=2.4,
            elinewidth=1.4,
            capsize=4,
            markersize=7,
            label=metadata["display_name"],
        )

        recovery_rows = paired_recovery[
            paired_recovery["model"] == model_name
        ].set_index("scenario")
        for scenario, x_position in [
            ("severe_signal_loss", 2),
            ("policy_restricted", 4),
        ]:
            delta = recovery_rows.loc[scenario, f"{recovery_metric}_mean"]
            vertical_offset = -18 if model_name == "logistic_regression" else 12
            horizontal_offset = -10 if model_name == "logistic_regression" else 10
            axis.annotate(
                f"{delta:+.3f}",
                xy=(x_position, mean[x_position]),
                xytext=(horizontal_offset, vertical_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=metadata["color"],
                fontweight="bold",
            )

    axis.set_xticks(
        x,
        [metadata["display_name"] for metadata in EXPERIMENTS.values()],
    )
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#0f172a")
    axis.set_ylabel("NDCG@3, mean +/- std")
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)


def plot_model_sensitivity(
    summary: pd.DataFrame,
    paired_recovery: pd.DataFrame,
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
    fig.patch.set_facecolor("#f8fafc")

    for axis in axes:
        axis.set_facecolor("#f8fafc")

    _plot_metric(
        axes[0],
        summary,
        paired_recovery,
        metric="overall_ndcg_at_3",
        recovery_metric="overall_recovery",
        title="Overall ranking utility",
    )
    _plot_metric(
        axes[1],
        summary,
        paired_recovery,
        metric="low_signal_ndcg_at_3",
        recovery_metric="low_signal_recovery",
        title="Low-signal ranking utility",
    )
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Model sensitivity: aggregate recovery is not model-invariant",
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
        "Logistic regression remains the interpretable primary baseline; "
        "histogram gradient boosting is a lightweight boundary check.",
        ha="left",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(
        0.055,
        0.035,
        "Labels at aggregate points report paired NDCG@3 recovery versus the same-model, "
        "same-seed signal-loss baseline. Mean +/- standard deviation across seeds 7, 42, and 101.",
        ha="left",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=(0.03, 0.11, 0.99, 0.86))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    summary: pd.DataFrame,
    paired_recovery: pd.DataFrame,
    out_path: Path,
) -> None:
    recovery_rows = []

    for model_name in MODELS:
        indexed = paired_recovery[
            paired_recovery["model"] == model_name
        ].set_index("scenario")
        for scenario in RECOVERY_PAIRS:
            row = indexed.loc[scenario]
            recovery_rows.append(
                {
                    "Model": row["model_display_name"],
                    "Scenario": row["scenario_display_name"],
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

    score_rows = []

    for model_name in MODELS:
        indexed = summary[summary["model"] == model_name].set_index("experiment")
        for experiment in EXPERIMENTS:
            row = indexed.loc[experiment]
            score_rows.append(
                {
                    "Model": row["model_display_name"],
                    "Scenario": row["experiment_display_name"].replace("\n", " "),
                    "Overall NDCG@3": (
                        f"{row['overall_ndcg_at_3_mean']:.3f} +/- "
                        f"{row['overall_ndcg_at_3_std']:.3f}"
                    ),
                    "Low-signal NDCG@3": (
                        f"{row['low_signal_ndcg_at_3_mean']:.3f} +/- "
                        f"{row['low_signal_ndcg_at_3_std']:.3f}"
                    ),
                }
            )

    out_path.write_text(
        "# Model Sensitivity Diagnostic\n\n"
        "This diagnostic compares the interpretable logistic primary baseline with "
        "histogram gradient boosting. Both models receive the same synthetic-data "
        "draws, household-level train/test split, signal-loss scenarios, and "
        "privacy-safe aggregate features.\n\n"
        "## Paired aggregate-recovery deltas\n\n"
        + pd.DataFrame(recovery_rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Scenario scores\n\n"
        + pd.DataFrame(score_rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Interpretation limits\n\n"
        "Histogram gradient boosting is a lightweight model-class sensitivity check, "
        "not a ranking-specific objective and not a replacement for the interpretable "
        "primary baseline. Differences across models show that aggregate-recovery "
        "results should be reported with model context rather than treated as "
        "model-invariant.\n"
    )


def main(seeds: Iterable[int] = SEEDS) -> None:
    frames = []

    for seed in seeds:
        print(f"Running seed={seed}")
        frames.append(evaluate_seed(seed))

    raw = pd.concat(frames, ignore_index=True)
    summary = build_model_sensitivity_summary(raw)
    paired_recovery = build_paired_recovery_summary(raw)

    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    docs_dir = Path("docs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    raw_path = tables_dir / "model_sensitivity_raw.csv"
    summary_path = tables_dir / "model_sensitivity_summary.csv"
    paired_path = tables_dir / "model_sensitivity_paired_recovery.csv"
    figure_path = assets_dir / "model_sensitivity.png"
    markdown_path = docs_dir / "model_sensitivity.md"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired_recovery.to_csv(paired_path, index=False)
    plot_model_sensitivity(summary, paired_recovery, figure_path)
    write_markdown_summary(summary, paired_recovery, markdown_path)

    print("\nModel-sensitivity paired recovery:")
    print(paired_recovery.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    print(f"- {paired_path}")
    print(f"- {figure_path}")
    print(f"- {markdown_path}")


if __name__ == "__main__":
    main()
