from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUPPORTED_AGGREGATIONS = {"population_weighted_mean"}
REQUIRED_TARGET_COLUMNS = {
    "metric",
    "display_name",
    "synthetic_column",
    "aggregation",
    "reference_value",
    "unit",
    "source_name",
    "source_period",
    "source_url",
    "retrieved_on",
    "notes",
}


def population_weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)

    if values_array.shape != weights_array.shape:
        raise ValueError("values and weights must have the same shape")
    if not np.isfinite(values_array).all() or not np.isfinite(weights_array).all():
        raise ValueError("values and weights must be finite")
    if (weights_array < 0).any() or np.isclose(weights_array.sum(), 0):
        raise ValueError("weights must be non-negative and sum to a positive value")

    return float(np.average(values_array, weights=weights_array))


def load_public_reference_targets(path: Path) -> pd.DataFrame:
    targets = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_TARGET_COLUMNS - set(targets.columns))

    if missing_columns:
        raise ValueError(f"public reference targets missing columns: {missing_columns}")
    if targets.empty:
        raise ValueError("public reference targets must not be empty")
    if not targets["metric"].is_unique:
        raise ValueError("public reference target metric names must be unique")
    if not set(targets["aggregation"]).issubset(SUPPORTED_AGGREGATIONS):
        raise ValueError("public reference targets contain an unsupported aggregation")
    if not (targets["source_url"].str.startswith("https://www.census.gov/")).all():
        raise ValueError("public reference targets must use documented Census sources")

    reference_values = targets["reference_value"].to_numpy(dtype=float)
    if not np.isfinite(reference_values).all() or (reference_values <= 0).any():
        raise ValueError("public reference target values must be finite and positive")

    return targets


def build_public_reference_comparison(
    communities: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    if "population" not in communities:
        raise ValueError("synthetic communities missing population column")

    rows = []
    for target in targets.to_dict("records"):
        synthetic_column = target["synthetic_column"]
        if synthetic_column not in communities:
            raise ValueError(
                f"synthetic communities missing configured column: {synthetic_column}"
            )

        synthetic_value = population_weighted_mean(
            communities[synthetic_column],
            communities["population"],
        )
        reference_value = float(target["reference_value"])
        gap = synthetic_value - reference_value

        rows.append(
            {
                **target,
                "synthetic_value": synthetic_value,
                "synthetic_minus_reference": gap,
                "relative_gap_vs_reference": gap / reference_value,
                "synthetic_as_share_of_reference": synthetic_value / reference_value,
            }
        )

    return pd.DataFrame(rows)


def _format_value(value: float, unit: str) -> str:
    if unit == "usd":
        return f"${value:,.0f}"
    if unit == "share":
        return f"{value:.1%}"
    return f"{value:,.3f}"


def plot_public_reference_comparison(
    comparison: pd.DataFrame,
    out_path: Path,
) -> None:
    plot_frame = comparison.copy()
    plot_frame["synthetic_percent_of_reference"] = (
        100 * plot_frame["synthetic_as_share_of_reference"]
    )
    plot_frame["relative_gap_percent"] = 100 * plot_frame["relative_gap_vs_reference"]

    labels = plot_frame["display_name"].tolist()
    positions = np.arange(len(plot_frame))[::-1]
    synthetic_percent = plot_frame["synthetic_percent_of_reference"].to_numpy()
    relative_gap_percent = plot_frame["relative_gap_percent"].to_numpy()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ratio_axis, gap_axis) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.8),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    fig.patch.set_facecolor("#f8fafc")
    for axis in (ratio_axis, gap_axis):
        axis.set_facecolor("#f8fafc")

    ratio_axis.axvline(100, color="#334155", linewidth=1.3, linestyle="--", alpha=0.8)
    for position, synthetic_value, (_, row) in zip(
        positions,
        synthetic_percent,
        plot_frame.iterrows(),
    ):
        label_offset = -24 if position == positions.max() else 15
        ratio_axis.plot(
            [synthetic_value, 100],
            [position, position],
            color="#94a3b8",
            linewidth=4,
            solid_capstyle="round",
            alpha=0.85,
        )
        ratio_axis.scatter(
            synthetic_value,
            position,
            s=145,
            color="#ea580c",
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
        )
        ratio_axis.scatter(
            100,
            position,
            s=145,
            color="#0f766e",
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
        )
        ratio_axis.annotate(
            f"Synthetic {_format_value(row['synthetic_value'], row['unit'])}",
            (synthetic_value, position),
            xytext=(0, label_offset),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#c2410c",
            fontweight="bold",
        )
        ratio_axis.annotate(
            f"Reference {_format_value(row['reference_value'], row['unit'])}",
            (100, position),
            xytext=(0, label_offset),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#0f766e",
            fontweight="bold",
        )

    ratio_axis.set_yticks(positions, labels)
    ratio_axis.set_xlim(
        min(65, synthetic_percent.min() - 8),
        max(108, synthetic_percent.max() + 8),
    )
    ratio_axis.set_xlabel("Synthetic value as a share of public reference (%)")
    ratio_axis.set_title(
        "Directional comparison",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )
    ratio_axis.spines[["top", "right", "left"]].set_visible(False)
    ratio_axis.tick_params(axis="y", length=0)

    colors = np.where(relative_gap_percent < 0, "#f97316", "#0f766e")
    gap_axis.barh(
        positions,
        relative_gap_percent,
        height=0.46,
        color=colors,
        alpha=0.9,
    )
    gap_axis.axvline(0, color="#334155", linewidth=1.1)
    for position, value in zip(positions, relative_gap_percent):
        gap_axis.annotate(
            f"{value:+.1f}%",
            (value, position),
            xytext=(7 if value >= 0 else -7, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            color="#334155",
            fontweight="bold",
        )

    gap_axis.set_yticks(positions, [""] * len(positions))
    gap_axis.set_xlabel("Relative gap versus public reference (%)")
    gap_axis.set_title(
        "Visible synthetic-prior gap",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )
    gap_axis.spines[["top", "right", "left"]].set_visible(False)
    gap_axis.tick_params(axis="y", length=0)

    fig.suptitle(
        "Public-reference calibration diagnostic",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.07,
        0.915,
        "Tracked U.S. Census Bureau QuickFacts anchors make selected synthetic "
        "context priors inspectable.",
        ha="left",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(
        0.07,
        0.035,
        "Diagnostic only: gaps are reported transparently; the synthetic benchmark "
        "is not fitted to or representative of a real population.",
        ha="left",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=(0.04, 0.10, 0.99, 0.86))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    communities = pd.read_csv("data/synthetic/synthetic_communities.csv")
    targets = load_public_reference_targets(Path("config/public_reference_targets.csv"))
    comparison = build_public_reference_comparison(communities, targets)

    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "public_reference_calibration.csv"
    figure_path = assets_dir / "public_reference_calibration.png"
    comparison.to_csv(csv_path, index=False)
    plot_public_reference_comparison(comparison, figure_path)

    print("Public-reference calibration diagnostic:")
    print(
        comparison[
            [
                "display_name",
                "synthetic_value",
                "reference_value",
                "relative_gap_vs_reference",
            ]
        ].to_string(index=False)
    )
    print("\nWrote:")
    print(f"- {csv_path}")
    print(f"- {figure_path}")


if __name__ == "__main__":
    main()
