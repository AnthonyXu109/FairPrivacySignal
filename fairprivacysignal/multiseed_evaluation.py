from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    evaluate_model,
)
from fairprivacysignal.privacy_transforms import add_privacy_safe_features
from fairprivacysignal.signal_loss import apply_signal_loss


SEEDS = [7, 11, 23, 42, 101]


def evaluate_seed(seed: int) -> list[dict]:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )

    rows = []

    full_signal = apply_signal_loss(events, "full_signal")
    rows.append(
        evaluate_model(
            full_signal,
            "full_signal_raw_baseline",
            BASE_NUMERIC_FEATURES,
        )
    )

    severe_loss = apply_signal_loss(events, "severe_signal_loss")
    rows.append(
        evaluate_model(
            severe_loss,
            "severe_signal_loss_baseline",
            BASE_NUMERIC_FEATURES,
        )
    )

    severe_loss_privacy_safe = add_privacy_safe_features(
        severe_loss,
        min_cohort_size=50,
        dp_noise_scale=1.0,
        seed=seed,
    )
    rows.append(
        evaluate_model(
            severe_loss_privacy_safe,
            "severe_signal_loss_with_privacy_safe_aggregates",
            PRIVACY_SAFE_NUMERIC_FEATURES,
        )
    )

    policy_restricted = apply_signal_loss(events, "policy_restricted")
    rows.append(
        evaluate_model(
            policy_restricted,
            "policy_restricted_baseline",
            BASE_NUMERIC_FEATURES,
        )
    )

    policy_restricted_privacy_safe = add_privacy_safe_features(
        policy_restricted,
        min_cohort_size=50,
        dp_noise_scale=1.0,
        seed=seed,
    )
    rows.append(
        evaluate_model(
            policy_restricted_privacy_safe,
            "policy_restricted_with_privacy_safe_aggregates",
            PRIVACY_SAFE_NUMERIC_FEATURES,
        )
    )

    for row in rows:
        row["seed"] = seed

    return rows


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

    severe_recovery_ndcg = summary.loc[
        summary["experiment"] == "severe_signal_loss_with_privacy_safe_aggregates",
        "overall_ndcg_at_3_mean",
    ].iloc[0]

    policy_ndcg = summary.loc[
        summary["experiment"] == "policy_restricted_baseline",
        "overall_ndcg_at_3_mean",
    ].iloc[0]

    policy_recovery_ndcg = summary.loc[
        summary["experiment"] == "policy_restricted_with_privacy_safe_aggregates",
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
    display_name = {
        "full_signal_raw_baseline": "Full signal raw baseline",
        "severe_signal_loss_baseline": "Severe signal loss",
        "severe_signal_loss_with_privacy_safe_aggregates": "Severe loss + privacy-safe aggregates",
        "policy_restricted_baseline": "Policy restricted",
        "policy_restricted_with_privacy_safe_aggregates": "Policy restricted + privacy-safe aggregates",
    }

    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Scenario": display_name.get(row["experiment"], row["experiment"]),
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
    order = [
        "full_signal_raw_baseline",
        "severe_signal_loss_baseline",
        "severe_signal_loss_with_privacy_safe_aggregates",
        "policy_restricted_baseline",
        "policy_restricted_with_privacy_safe_aggregates",
    ]

    label_map = {
        "full_signal_raw_baseline": "Full signal\nraw baseline",
        "severe_signal_loss_baseline": "Severe signal\nloss",
        "severe_signal_loss_with_privacy_safe_aggregates": "Severe loss\n+ privacy-safe",
        "policy_restricted_baseline": "Policy\nrestricted",
        "policy_restricted_with_privacy_safe_aggregates": "Policy restricted\n+ privacy-safe",
    }

    df = summary.set_index("experiment").loc[order].reset_index()
    labels = [label_map[x] for x in df["experiment"]]

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(
        labels,
        df["overall_ndcg_at_3_mean"],
        yerr=df["overall_ndcg_at_3_std"],
        capsize=4,
    )

    for bar, value in zip(bars, df["overall_ndcg_at_3_mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("Overall NDCG@3, mean ± std")
    plt.title("Multi-seed ranking utility under signal-loss scenarios")
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

    rows = []
    for seed in SEEDS:
        print(f"Running seed={seed}")
        rows.extend(evaluate_seed(seed))

    results = pd.DataFrame(rows)
    summary = build_summary(results)

    raw_path = out_dir / "multiseed_privacy_recovery_raw.csv"
    summary_path = out_dir / "multiseed_privacy_recovery_summary.csv"
    markdown_path = docs_dir / "multiseed_results.md"
    figure_path = asset_dir / "multiseed_privacy_recovery_ndcg.png"

    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown_summary(summary, markdown_path)
    plot_multiseed_ndcg(summary, figure_path)

    print("\nMulti-seed summary:")
    print(summary.round(4).to_string(index=False))
    print(f"\nWrote:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    print(f"- {markdown_path}")
    print(f"- {figure_path}")


if __name__ == "__main__":
    main()
