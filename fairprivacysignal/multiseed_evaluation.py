from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.privacy_recovery import run_experiments


SEEDS = [7, 11, 23, 42, 101]


DISPLAY_NAME = {
    "full_signal_raw_baseline": "Full signal raw baseline",
    "severe_signal_loss_baseline": "Severe signal loss",
    "severe_signal_loss_with_privacy_safe_aggregates": "Severe loss + privacy-safe aggregates",
    "severe_signal_loss_with_privacy_safe_fairness_aware": "Severe loss + fairness-aware recovery",
    "policy_restricted_baseline": "Policy restricted",
    "policy_restricted_with_privacy_safe_aggregates": "Policy restricted + privacy-safe aggregates",
    "policy_restricted_with_privacy_safe_fairness_aware": "Policy restricted + fairness-aware recovery",
}


ORDER = list(DISPLAY_NAME.keys())


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )

    results = run_experiments(events)
    results["seed"] = seed
    return results


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_auc",
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "not_low_signal_ndcg_at_3",
        "ndcg_gap_not_low_minus_low",
        "avg_privacy_exposure_score",
        "behavioral_available_share",
    ]

    summary = results.groupby("experiment")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    full_ndcg = summary.loc[
        summary["experiment"] == "full_signal_raw_baseline",
        "overall_ndcg_at_3_mean",
    ].iloc[0]

    severe_ndcg = summary.loc[
        summary["experiment"] == "severe_signal_loss_baseline",
        "overall_ndcg_at_3_mean",
    ].iloc[0]

    policy_ndcg = summary.loc[
        summary["experiment"] == "policy_restricted_baseline",
        "overall_ndcg_at_3_mean",
    ].iloc[0]

    summary["utility_delta_vs_full_signal"] = (
        summary["overall_ndcg_at_3_mean"] - full_ndcg
    )

    summary["utility_recovered_vs_severe_loss"] = (
        summary["overall_ndcg_at_3_mean"] - severe_ndcg
    )

    summary["utility_recovered_vs_policy_baseline"] = (
        summary["overall_ndcg_at_3_mean"] - policy_ndcg
    )

    return summary


def write_markdown_summary(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    ordered = summary.set_index("experiment").loc[ORDER].reset_index()

    for _, row in ordered.iterrows():
        rows.append(
            {
                "Scenario": DISPLAY_NAME[row["experiment"]],
                "Privacy exposure": f'{row["avg_privacy_exposure_score_mean"]:.3f} ± {row["avg_privacy_exposure_score_std"]:.3f}',
                "NDCG@3": f'{row["overall_ndcg_at_3_mean"]:.3f} ± {row["overall_ndcg_at_3_std"]:.3f}',
                "Low-signal NDCG@3": f'{row["low_signal_ndcg_at_3_mean"]:.3f} ± {row["low_signal_ndcg_at_3_std"]:.3f}',
                "Low-signal gap": f'{row["ndcg_gap_not_low_minus_low_mean"]:.3f} ± {row["ndcg_gap_not_low_minus_low_std"]:.3f}',
            }
        )

    md = pd.DataFrame(rows).to_markdown(index=False)

    out_path.write_text(
        "# Multi-seed Evaluation Summary\n\n"
        "This table reports mean ± standard deviation across five synthetic data seeds.\n\n"
        + md
        + "\n"
    )


def plot_multiseed_ndcg(summary: pd.DataFrame, out_path: Path) -> None:
    ordered = summary.set_index("experiment").loc[ORDER].reset_index()
    labels = [
        DISPLAY_NAME[x]
        .replace(" raw baseline", "")
        .replace(" + ", "\n+ ")
        .replace(" signal loss", "\nsignal loss")
        .replace(" restricted", "\nrestricted")
        for x in ordered["experiment"]
    ]

    plt.figure(figsize=(11, 5.2))
    bars = plt.bar(
        labels,
        ordered["overall_ndcg_at_3_mean"],
        yerr=ordered["overall_ndcg_at_3_std"],
        capsize=4,
    )

    for bar, value in zip(bars, ordered["overall_ndcg_at_3_mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("Overall NDCG@3, mean ± std")
    plt.title("Multi-seed ranking utility under privacy and fairness recovery scenarios")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_multiseed_fairness_gap(summary: pd.DataFrame, out_path: Path) -> None:
    ordered = summary.set_index("experiment").loc[ORDER].reset_index()
    labels = [
        DISPLAY_NAME[x]
        .replace(" raw baseline", "")
        .replace(" + ", "\n+ ")
        .replace(" signal loss", "\nsignal loss")
        .replace(" restricted", "\nrestricted")
        for x in ordered["experiment"]
    ]

    plt.figure(figsize=(11, 5.2))
    bars = plt.bar(
        labels,
        ordered["ndcg_gap_not_low_minus_low_mean"],
        yerr=ordered["ndcg_gap_not_low_minus_low_std"],
        capsize=4,
    )

    for bar, value in zip(bars, ordered["ndcg_gap_not_low_minus_low_mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("NDCG@3 gap: not-low-signal minus low-signal")
    plt.title("Multi-seed low-signal ranking gap diagnostics")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    out_dir = Path("outputs/tables")
    asset_dir = Path("docs/assets")
    docs_dir = Path("docs")

    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for seed in SEEDS:
        print(f"Running seed={seed}")
        frames.append(evaluate_seed(seed))

    results = pd.concat(frames, ignore_index=True)
    summary = build_summary(results)

    raw_path = out_dir / "multiseed_privacy_recovery_raw.csv"
    summary_path = out_dir / "multiseed_privacy_recovery_summary.csv"
    markdown_path = docs_dir / "multiseed_results.md"
    utility_figure_path = asset_dir / "multiseed_privacy_recovery_ndcg.png"
    fairness_figure_path = asset_dir / "multiseed_fairness_gap.png"

    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown_summary(summary, markdown_path)
    plot_multiseed_ndcg(summary, utility_figure_path)
    plot_multiseed_fairness_gap(summary, fairness_figure_path)

    print("\nMulti-seed summary:")
    print(summary.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    print(f"- {markdown_path}")
    print(f"- {utility_figure_path}")
    print(f"- {fairness_figure_path}")


if __name__ == "__main__":
    main()
