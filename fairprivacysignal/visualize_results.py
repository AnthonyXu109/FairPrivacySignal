from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fairprivacysignal.privacy_recovery import main as run_privacy_recovery


NAME_MAP = {
    "full_signal_raw_baseline": "Full signal\nraw baseline",
    "severe_signal_loss_baseline": "Severe signal\nloss",
    "severe_signal_loss_with_privacy_safe_aggregates": "Severe loss\n+ privacy-safe",
    "severe_signal_loss_with_privacy_safe_fairness_aware": "Severe loss\n+ fairness-aware",
    "policy_restricted_baseline": "Policy\nrestricted",
    "policy_restricted_with_privacy_safe_aggregates": "Policy restricted\n+ privacy-safe",
    "policy_restricted_with_privacy_safe_fairness_aware": "Policy restricted\n+ fairness-aware",
}


ANNOTATION_STYLE = {
    "full_signal_raw_baseline": {"xytext": (-5, 5), "ha": "right"},
    "severe_signal_loss_baseline": {"xytext": (5, 5), "ha": "left"},
    "severe_signal_loss_with_privacy_safe_aggregates": {
        "xytext": (5, 5),
        "ha": "left",
    },
    "severe_signal_loss_with_privacy_safe_fairness_aware": {
        "xytext": (5, 14),
        "ha": "left",
    },
    "policy_restricted_baseline": {"xytext": (5, 5), "ha": "left"},
    "policy_restricted_with_privacy_safe_aggregates": {
        "xytext": (5, 8),
        "ha": "left",
    },
    "policy_restricted_with_privacy_safe_fairness_aware": {
        "xytext": (5, -15),
        "ha": "left",
    },
}


def main() -> None:
    metrics_path = Path("outputs/tables/privacy_recovery_metrics.csv")
    assets_dir = Path("docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        run_privacy_recovery()

    df = pd.read_csv(metrics_path)

    df["display_name"] = df["experiment"].map(NAME_MAP)

    # 1. Utility comparison.
    plt.figure(figsize=(9, 4.8))
    plt.bar(df["display_name"], df["overall_ndcg_at_3"])
    plt.ylabel("Overall NDCG@3")
    plt.title("Privacy-safe aggregate features recover ranking utility under signal loss")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(assets_dir / "privacy_recovery_ndcg.png", dpi=180)
    plt.close()

    # 2. Fairness gap comparison.
    plt.figure(figsize=(9, 4.8))
    plt.bar(df["display_name"], df["ndcg_gap_not_low_minus_low"])
    plt.ylabel("NDCG@3 gap: not-low-signal minus low-signal")
    plt.title("Low-signal ranking gaps across privacy scenarios")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(assets_dir / "privacy_recovery_fairness_gap.png", dpi=180)
    plt.close()

    # 3. Privacy-utility tradeoff.
    plt.figure(figsize=(7, 5))
    plt.scatter(df["avg_privacy_exposure_score"], df["overall_ndcg_at_3"], s=80)

    for _, row in df.iterrows():
        annotation_style = ANNOTATION_STYLE[row["experiment"]]
        plt.annotate(
            row["display_name"].replace("\n", " "),
            (row["avg_privacy_exposure_score"], row["overall_ndcg_at_3"]),
            textcoords="offset points",
            xytext=annotation_style["xytext"],
            ha=annotation_style["ha"],
            fontsize=8,
        )

    plt.xlabel("Average privacy exposure score")
    plt.ylabel("Overall NDCG@3")
    plt.title("Privacy-utility tradeoff across signal-loss scenarios")
    plt.tight_layout()
    plt.savefig(assets_dir / "privacy_utility_tradeoff.png", dpi=180)
    plt.close()

    print("Wrote result figures:")
    for path in [
        assets_dir / "privacy_recovery_ndcg.png",
        assets_dir / "privacy_recovery_fairness_gap.png",
        assets_dir / "privacy_utility_tradeoff.png",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
