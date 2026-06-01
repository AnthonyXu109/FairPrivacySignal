from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    evaluate_model,
)
from fairprivacysignal.privacy_transforms import add_privacy_safe_features
from fairprivacysignal.signal_loss import apply_signal_loss


NOISE_SCALES = [0.0, 0.5, 1.0, 2.0, 4.0]
NOISE_SEEDS = [7, 42, 101]
DEFAULT_NOISE_SCALE = 1.0

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


def run_noise_sensitivity(
    events: pd.DataFrame,
    noise_scales: Iterable[float] = NOISE_SCALES,
    noise_seeds: Iterable[int] = NOISE_SEEDS,
) -> pd.DataFrame:
    rows = []

    for scenario, metadata in SCENARIOS.items():
        signal_limited = apply_signal_loss(events, scenario)
        baseline = evaluate_model(
            signal_limited,
            f"{scenario}_baseline",
            BASE_NUMERIC_FEATURES,
        )

        for noise_scale in noise_scales:
            for noise_seed in noise_seeds:
                privacy_safe = add_privacy_safe_features(
                    signal_limited,
                    dp_noise_scale=noise_scale,
                    seed=noise_seed,
                )
                metrics = evaluate_model(
                    privacy_safe,
                    f"{scenario}_with_privacy_safe_aggregates",
                    PRIVACY_SAFE_NUMERIC_FEATURES,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "display_name": metadata["display_name"],
                        "noise_scale": float(noise_scale),
                        "noise_seed": int(noise_seed),
                        "baseline_overall_ndcg_at_3": baseline["overall_ndcg_at_3"],
                        "aggregate_overall_ndcg_at_3": metrics["overall_ndcg_at_3"],
                        "overall_utility_recovery": (
                            metrics["overall_ndcg_at_3"]
                            - baseline["overall_ndcg_at_3"]
                        ),
                        "baseline_low_signal_ndcg_at_3": baseline[
                            "low_signal_ndcg_at_3"
                        ],
                        "aggregate_low_signal_ndcg_at_3": metrics[
                            "low_signal_ndcg_at_3"
                        ],
                        "low_signal_utility_recovery": (
                            metrics["low_signal_ndcg_at_3"]
                            - baseline["low_signal_ndcg_at_3"]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_noise_sensitivity_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "baseline_overall_ndcg_at_3",
        "aggregate_overall_ndcg_at_3",
        "overall_utility_recovery",
        "baseline_low_signal_ndcg_at_3",
        "aggregate_low_signal_ndcg_at_3",
        "low_signal_utility_recovery",
    ]
    summary = (
        raw.groupby(["scenario", "display_name", "noise_scale"])[metric_columns]
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
    baseline_metric: str,
    title: str,
) -> None:
    for scenario, metadata in SCENARIOS.items():
        scenario_rows = summary[summary["scenario"] == scenario].sort_values(
            "noise_scale"
        )
        x = scenario_rows["noise_scale"].to_numpy(dtype=float)
        mean = scenario_rows[f"{metric}_mean"].to_numpy(dtype=float)
        std = scenario_rows[f"{metric}_std"].to_numpy(dtype=float)
        baseline = scenario_rows[f"{baseline_metric}_mean"].iloc[0]

        axis.plot(
            x,
            mean,
            color=metadata["color"],
            linewidth=2.8,
            marker="o",
            markersize=7,
            label=f"{metadata['display_name']} + aggregates",
        )
        axis.fill_between(
            x,
            mean - std,
            mean + std,
            color=metadata["color"],
            alpha=0.14,
        )
        axis.axhline(
            baseline,
            color=metadata["color"],
            linestyle="--",
            linewidth=1.3,
            alpha=0.7,
            label=f"{metadata['display_name']} baseline",
        )

    axis.axvline(
        DEFAULT_NOISE_SCALE,
        color="#334155",
        linestyle=":",
        linewidth=1.5,
        alpha=0.85,
    )
    axis.text(
        DEFAULT_NOISE_SCALE + 0.06,
        0.98,
        "default scale",
        transform=axis.get_xaxis_transform(),
        color="#475569",
        fontsize=9,
        rotation=90,
        va="top",
    )
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#0f172a")
    axis.set_xlabel("Aggregate-noise stress scale")
    axis.set_ylabel("NDCG@3, mean ± std")
    axis.spines[["top", "right"]].set_visible(False)


def plot_noise_sensitivity(summary: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    for axis in axes:
        axis.set_facecolor("#f8fafc")

    _plot_metric(
        axes[0],
        summary,
        metric="aggregate_overall_ndcg_at_3",
        baseline_metric="baseline_overall_ndcg_at_3",
        title="Overall ranking utility",
    )
    _plot_metric(
        axes[1],
        summary,
        metric="aggregate_low_signal_ndcg_at_3",
        baseline_metric="baseline_low_signal_ndcg_at_3",
        title="Low-signal ranking utility",
    )

    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle(
        "Aggregate-noise sensitivity of privacy-safe recovery",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.06,
        0.92,
        "Recovery remains visible across a controlled sweep of DP-style "
        "aggregate-noise stress scales.",
        ha="left",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(
        0.06,
        0.035,
        "Synthetic dataset seed 42; mean ± std across aggregate-noise seeds "
        "7, 42, and 101. Stress scale is not a formal privacy budget.",
        ha="left",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=(0.03, 0.10, 0.99, 0.86))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    events = pd.read_csv("data/synthetic/synthetic_outreach_events.csv")
    raw = run_noise_sensitivity(events)
    summary = build_noise_sensitivity_summary(raw)

    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    raw_path = tables_dir / "aggregate_noise_sensitivity_raw.csv"
    summary_path = tables_dir / "aggregate_noise_sensitivity_summary.csv"
    figure_path = assets_dir / "aggregate_noise_sensitivity.png"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_noise_sensitivity(summary, figure_path)

    print("Aggregate-noise sensitivity summary:")
    print(summary.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    print(f"- {figure_path}")


if __name__ == "__main__":
    main()
