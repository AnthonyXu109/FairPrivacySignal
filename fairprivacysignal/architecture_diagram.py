from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from fairprivacysignal.benchmark_validation import (
    EXPECTED_CONTEXT_SHIFT_LEVELS,
    EXPECTED_PUBLIC_REFERENCE_METRICS,
    EXPECTED_RANKING_OBJECTIVES,
    EXPECTED_SEEDS,
    EXPECTED_SIGNAL_SCENARIOS,
)


COLORS = {
    "background": "#f8fafc",
    "ink": "#0f172a",
    "muted": "#64748b",
    "line": "#94a3b8",
    "input": "#dbeafe",
    "privacy": "#ffedd5",
    "model": "#dcfce7",
    "evaluation": "#ede9fe",
    "validation": "#e2e8f0",
    "white": "#ffffff",
}


def _draw_card(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: Iterable[str],
    color: str,
) -> None:
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=COLORS["line"],
        facecolor=color,
    )
    axis.add_patch(card)
    axis.text(
        x + 0.18,
        y + height - 0.23,
        title,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        x + 0.18,
        y + height - 0.62,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9.3,
        color=COLORS["ink"],
        linespacing=1.45,
    )


def _draw_arrow(
    axis: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.7,
            "color": COLORS["line"],
            "shrinkA": 3,
            "shrinkB": 3,
        },
    )


def _draw_badge(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
) -> None:
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.6",
            "facecolor": color,
            "edgecolor": COLORS["line"],
            "linewidth": 1.0,
        },
    )


def plot_architecture_diagram(out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["background"])
    axis.set_facecolor(COLORS["background"])
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")

    fig.suptitle(
        "FairPrivacySignal benchmark system map",
        x=0.045,
        y=0.975,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.047,
        0.925,
        "A synthetic, auditable pipeline for measuring privacy-driven signal loss, aggregate recovery, fairness diagnostics, and allocation tradeoffs.",
        ha="left",
        fontsize=11,
        color=COLORS["muted"],
    )

    card_y = 3.55
    card_height = 3.75
    card_width = 3.25
    x_positions = [0.45, 4.35, 8.25, 12.15]

    _draw_card(
        axis,
        x_positions[0],
        card_y,
        card_width,
        card_height,
        "1. Synthetic foundation",
        [
            "120 generated communities",
            "10,000 generated households",
            "6 candidate service categories",
            "Household-service relevance labels",
            f"{len(EXPECTED_PUBLIC_REFERENCE_METRICS)} tracked public anchors",
        ],
        COLORS["input"],
    )
    _draw_card(
        axis,
        x_positions[1],
        card_y,
        card_width,
        card_height,
        "2. Privacy and recovery",
        [
            f"{len(EXPECTED_SIGNAL_SCENARIOS)} signal-loss scenarios",
            "Consent and policy suppression",
            "Train-fitted cohort aggregates",
            "k-threshold service fallback",
            "DP-style aggregate-noise stress",
        ],
        COLORS["privacy"],
    )
    _draw_card(
        axis,
        x_positions[2],
        card_y,
        card_width,
        card_height,
        "3. Ranking and allocation",
        [
            f"{len(EXPECTED_RANKING_OBJECTIVES)} ranking objectives",
            "Logistic and boosted model checks",
            "Privacy-safe recovery variants",
            "Low-signal recovery diagnostic",
            "Capacity-constrained allocation",
        ],
        COLORS["model"],
    )
    _draw_card(
        axis,
        x_positions[3],
        card_y,
        card_width,
        card_height,
        "4. Evidence surfaces",
        [
            "AUC and NDCG@3 utility",
            "Privacy-exposure proxy",
            "Low-signal and calibration gaps",
            "Feature and model ablations",
            "Grouped and context-shift stress",
        ],
        COLORS["evaluation"],
    )

    for left_x, right_x in zip(x_positions[:-1], x_positions[1:]):
        _draw_arrow(
            axis,
            (left_x + card_width + 0.08, card_y + card_height / 2),
            (right_x - 0.08, card_y + card_height / 2),
        )

    _draw_badge(
        axis,
        2.0,
        2.55,
        f"{len(EXPECTED_SEEDS)} paired robustness seeds",
        COLORS["input"],
    )
    _draw_badge(
        axis,
        5.9,
        2.55,
        "train-only aggregate references",
        COLORS["privacy"],
    )
    _draw_badge(
        axis,
        9.9,
        2.55,
        f"{len(EXPECTED_CONTEXT_SHIFT_LEVELS)} controlled shift levels",
        COLORS["model"],
    )
    _draw_badge(
        axis,
        14.0,
        2.55,
        "generated evidence tables + figures",
        COLORS["evaluation"],
    )

    validation_bar = FancyBboxPatch(
        (0.45, 0.65),
        14.95,
        1.15,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=COLORS["line"],
        facecolor=COLORS["validation"],
    )
    axis.add_patch(validation_bar)
    axis.text(
        0.75,
        1.22,
        "Machine-checked validation gate",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        5.05,
        1.22,
        "coverage invariants",
        ha="center",
        va="center",
        fontsize=9.7,
        color=COLORS["muted"],
    )
    axis.text(
        7.8,
        1.22,
        "bounded metrics",
        ha="center",
        va="center",
        fontsize=9.7,
        color=COLORS["muted"],
    )
    axis.text(
        10.35,
        1.22,
        "leakage checks",
        ha="center",
        va="center",
        fontsize=9.7,
        color=COLORS["muted"],
    )
    axis.text(
        13.4,
        1.22,
        "benchmark card + reproducible outputs",
        ha="center",
        va="center",
        fontsize=9.7,
        color=COLORS["muted"],
    )

    fig.text(
        0.047,
        0.02,
        "All populations, outcomes, and community contexts are synthetic. DP-style noise and fairness metrics are diagnostics, not production guarantees.",
        ha="left",
        fontsize=9.2,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.89))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_path = Path("docs/assets/architecture_diagram.png")
    plot_architecture_diagram(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
