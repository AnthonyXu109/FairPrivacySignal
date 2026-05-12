from pathlib import Path

import matplotlib.pyplot as plt


def draw_box(ax, x, y, text, width=3.2, height=0.75):
    box = plt.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        fill=False,
        linewidth=1.6,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.4),
    )


def main() -> None:
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Main pipeline
    draw_box(ax, 5, 9, "Synthetic Public-Service\nOutreach Data")
    draw_box(ax, 5, 7.7, "Signal-Loss Simulator\nfull / severe / consent / policy")
    draw_box(ax, 5, 6.4, "Policy & Consent\nFeature Suppression")
    draw_box(ax, 5, 5.1, "Privacy-Safe Recovery\ncohort aggregation + k-thresholds + DP-style noise")
    draw_box(ax, 5, 3.8, "Ranking Model\nservice relevance prediction")
    draw_box(ax, 5, 2.5, "Evaluation Layer\nutility + privacy exposure + fairness diagnostics")

    draw_arrow(ax, 5, 8.62, 5, 8.08)
    draw_arrow(ax, 5, 7.32, 5, 6.78)
    draw_arrow(ax, 5, 6.02, 5, 5.48)
    draw_arrow(ax, 5, 4.72, 5, 4.18)
    draw_arrow(ax, 5, 3.42, 5, 2.88)

    # Side notes
    draw_box(ax, 1.9, 6.4, "Privacy Constraints\nconsent, sensitive cohort,\ndata minimization", width=2.8)
    draw_arrow(ax, 3.3, 6.4, 3.55, 6.4)

    draw_box(ax, 8.1, 5.1, "Aggregate Signals\ncontextual features,\ncohort statistics", width=2.8)
    draw_arrow(ax, 6.6, 5.1, 6.7, 5.1)

    draw_box(ax, 2.0, 2.5, "Utility Metrics\nAUC, NDCG@3", width=2.6)
    draw_box(ax, 8.0, 2.5, "Fairness Diagnostics\nlow-signal NDCG gap", width=2.6)

    ax.text(
        5,
        0.9,
        "FairPrivacySignal evaluates how privacy-driven signal loss affects ranking utility and low-signal groups, "
        "and whether privacy-safe aggregate features can recover useful signal without restoring raw behavioral exposure.",
        ha="center",
        va="center",
        fontsize=9,
        wrap=True,
    )

    plt.tight_layout()
    out_path = out_dir / "architecture_diagram.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
