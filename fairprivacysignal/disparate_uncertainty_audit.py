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
BOOTSTRAP_REPLICATES = 6

EXPERIMENTS = {
    "full_signal_raw_baseline": {
        "display_name": "Full signal",
        "scenario": "full_signal",
        "aggregates": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "severe_signal_loss_baseline": {
        "display_name": "Severe loss",
        "scenario": "severe_signal_loss",
        "aggregates": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "severe_signal_loss_with_privacy_safe_aggregates": {
        "display_name": "Severe loss\n+ aggregates",
        "scenario": "severe_signal_loss",
        "aggregates": True,
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
    },
    "policy_restricted_baseline": {
        "display_name": "Policy restricted",
        "scenario": "policy_restricted",
        "aggregates": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "policy_restricted_with_privacy_safe_aggregates": {
        "display_name": "Policy restricted\n+ aggregates",
        "scenario": "policy_restricted",
        "aggregates": True,
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
    },
}

RECOVERY_PAIRS = {
    "severe_signal_loss": (
        "severe_signal_loss_baseline",
        "severe_signal_loss_with_privacy_safe_aggregates",
    ),
    "policy_restricted": (
        "policy_restricted_baseline",
        "policy_restricted_with_privacy_safe_aggregates",
    ),
}


def bootstrap_households(
    train: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    household_ids = train["household_id"].drop_duplicates().to_numpy()
    sampled_ids = rng.choice(
        household_ids,
        size=len(household_ids),
        replace=True,
    )
    sample_map = pd.DataFrame(
        {
            "source_household_id": sampled_ids,
            "bootstrap_household_id": [
                f"{household_id}__bootstrap_{index}"
                for index, household_id in enumerate(sampled_ids)
            ],
        }
    )
    bootstrapped = sample_map.merge(
        train,
        left_on="source_household_id",
        right_on="household_id",
        how="left",
        validate="many_to_many",
    )
    bootstrapped["household_id"] = bootstrapped["bootstrap_household_id"]
    return bootstrapped.drop(
        columns=["source_household_id", "bootstrap_household_id"]
    )


def top_k_agreement(
    test: pd.DataFrame,
    predictions: np.ndarray,
    mean_predictions: np.ndarray,
    k: int = 3,
) -> tuple[float, float]:
    low_agreements = []
    not_low_agreements = []

    for positions in test.groupby("household_id", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        effective_k = min(k, len(positions))
        reference = set(
            positions[np.argsort(mean_predictions[positions])[-effective_k:]]
        )
        replicate_agreements = []
        for replicate in predictions:
            selected = set(
                positions[np.argsort(replicate[positions])[-effective_k:]]
            )
            replicate_agreements.append(len(reference & selected) / effective_k)

        target = (
            low_agreements
            if bool(test.iloc[positions[0]]["low_signal"])
            else not_low_agreements
        )
        target.append(float(np.mean(replicate_agreements)))

    return float(np.mean(low_agreements)), float(np.mean(not_low_agreements))


def evaluate_experiment(
    events: pd.DataFrame,
    experiment: str,
    seed: int,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    metadata = EXPERIMENTS[experiment]
    signal_limited = apply_signal_loss(events, metadata["scenario"])
    train, test = split_household_events(signal_limited)
    predictions = []

    for replicate in range(bootstrap_replicates):
        bootstrap_seed = seed * 1000 + replicate
        bootstrap_train = bootstrap_households(train, seed=bootstrap_seed)
        scored_test = test.copy()

        if metadata["aggregates"]:
            bootstrap_train = add_privacy_safe_features(
                bootstrap_train,
                reference_events=bootstrap_train,
                seed=seed,
            )
            scored_test = add_privacy_safe_features(
                scored_test,
                reference_events=bootstrap_train,
                seed=seed,
            )

        numeric_features = metadata["numeric_features"]
        features = numeric_features + CATEGORICAL_FEATURES
        model = build_model(numeric_features)
        model.fit(bootstrap_train[features], bootstrap_train["relevant"])
        predictions.append(model.predict_proba(scored_test[features])[:, 1])

    prediction_matrix = np.asarray(predictions)
    mean_prediction = prediction_matrix.mean(axis=0)
    prediction_std = prediction_matrix.std(axis=0, ddof=1)
    scored = test.copy()
    scored["predicted_relevance"] = mean_prediction
    low_mask = scored["low_signal"].astype(bool).to_numpy()
    low_top3, not_low_top3 = top_k_agreement(
        scored,
        prediction_matrix,
        mean_prediction,
        k=3,
    )
    low_uncertainty = float(prediction_std[low_mask].mean())
    not_low_uncertainty = float(prediction_std[~low_mask].mean())

    return {
        "seed": int(seed),
        "experiment": experiment,
        "experiment_display_name": metadata["display_name"],
        "scenario": metadata["scenario"],
        "uses_privacy_safe_aggregates": bool(metadata["aggregates"]),
        "bootstrap_replicates": int(bootstrap_replicates),
        "overall_ndcg_at_3": average_ndcg_at_k(scored, k=3),
        "low_signal_ndcg_at_3": average_ndcg_at_k(
            scored[scored["low_signal"].astype(bool)],
            k=3,
        ),
        "low_signal_prediction_std": low_uncertainty,
        "not_low_signal_prediction_std": not_low_uncertainty,
        "prediction_std_gap_low_minus_not_low": (
            low_uncertainty - not_low_uncertainty
        ),
        "low_signal_top3_agreement": low_top3,
        "not_low_signal_top3_agreement": not_low_top3,
        "top3_agreement_gap_not_low_minus_low": not_low_top3 - low_top3,
        "num_test_events": int(len(test)),
        "aggregate_reference_scope": (
            "train_households_only"
            if metadata["aggregates"]
            else "not_applicable"
        ),
    }


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    return pd.DataFrame(
        [
            evaluate_experiment(events, experiment, seed=seed)
            for experiment in EXPERIMENTS
        ]
    )


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "low_signal_prediction_std",
        "not_low_signal_prediction_std",
        "prediction_std_gap_low_minus_not_low",
        "low_signal_top3_agreement",
        "not_low_signal_top3_agreement",
        "top3_agreement_gap_not_low_minus_low",
    ]
    summary = (
        raw.groupby(["experiment", "experiment_display_name"])[metrics]
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


def build_paired_effects(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, (baseline_name, aggregate_name) in RECOVERY_PAIRS.items():
        baseline = raw[raw["experiment"] == baseline_name].set_index("seed")
        aggregate = raw[raw["experiment"] == aggregate_name].set_index("seed")
        for seed in sorted(set(baseline.index) & set(aggregate.index)):
            rows.append(
                {
                    "seed": int(seed),
                    "scenario": scenario,
                    "overall_ndcg_recovery": (
                        aggregate.loc[seed, "overall_ndcg_at_3"]
                        - baseline.loc[seed, "overall_ndcg_at_3"]
                    ),
                    "low_signal_uncertainty_change": (
                        aggregate.loc[seed, "low_signal_prediction_std"]
                        - baseline.loc[seed, "low_signal_prediction_std"]
                    ),
                    "low_signal_top3_agreement_change": (
                        aggregate.loc[seed, "low_signal_top3_agreement"]
                        - baseline.loc[seed, "low_signal_top3_agreement"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_uncertainty_audit(summary: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    ordered = summary.set_index("experiment").loc[list(EXPERIMENTS)]
    x = np.arange(len(ordered), dtype=float)
    labels = ordered["experiment_display_name"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    fig.patch.set_facecolor("#f8fafc")

    axes[0].errorbar(
        x - 0.07,
        ordered["overall_ndcg_at_3_mean"],
        yerr=ordered["overall_ndcg_at_3_std"],
        marker="o",
        capsize=4,
        linewidth=2,
        label="Overall",
    )
    axes[0].errorbar(
        x + 0.07,
        ordered["low_signal_ndcg_at_3_mean"],
        yerr=ordered["low_signal_ndcg_at_3_std"],
        marker="s",
        capsize=4,
        linewidth=2,
        label="Low-signal",
    )
    axes[0].set_title("Ensemble-mean ranking utility", loc="left")
    axes[0].set_ylabel("NDCG@3, mean +/- std")

    axes[1].errorbar(
        x - 0.07,
        ordered["low_signal_prediction_std_mean"],
        yerr=ordered["low_signal_prediction_std_std"],
        marker="o",
        capsize=4,
        linewidth=2,
        label="Low-signal",
    )
    axes[1].errorbar(
        x + 0.07,
        ordered["not_low_signal_prediction_std_mean"],
        yerr=ordered["not_low_signal_prediction_std_std"],
        marker="s",
        capsize=4,
        linewidth=2,
        label="Not low-signal",
    )
    axes[1].set_title("Training-resample score instability", loc="left")
    axes[1].set_ylabel("Mean prediction standard deviation")

    axes[2].errorbar(
        x - 0.07,
        ordered["low_signal_top3_agreement_mean"],
        yerr=ordered["low_signal_top3_agreement_std"],
        marker="o",
        capsize=4,
        linewidth=2,
        label="Low-signal",
    )
    axes[2].errorbar(
        x + 0.07,
        ordered["not_low_signal_top3_agreement_mean"],
        yerr=ordered["not_low_signal_top3_agreement_std"],
        marker="s",
        capsize=4,
        linewidth=2,
        label="Not low-signal",
    )
    axes[2].set_title("Top-3 membership stability", loc="left")
    axes[2].set_ylabel("Bootstrap agreement share")
    axes[2].set_ylim(0.75, 1.01)

    for axis in axes:
        axis.set_facecolor("#f8fafc")
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=8.2)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        "Disparate uncertainty and ranking-stability audit",
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
        "Household-bootstrap ensembles test whether signal loss changes score variability and Top-3 stability by group.",
        color="#475569",
        fontsize=10.5,
    )
    fig.text(
        0.045,
        0.02,
        "Six household-bootstrap fits per experiment and three synthetic-data seeds. "
        "This measures training-resample instability, not calibrated posterior uncertainty.",
        color="#64748b",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.10, 0.99, 0.87))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    summary: pd.DataFrame,
    paired_effects: pd.DataFrame,
    out_path: Path,
) -> None:
    rows = []
    for experiment in EXPERIMENTS:
        row = summary.set_index("experiment").loc[experiment]
        rows.append(
            {
                "Experiment": row["experiment_display_name"].replace("\n", " "),
                "Overall NDCG@3": (
                    f"{row['overall_ndcg_at_3_mean']:.3f} +/- "
                    f"{row['overall_ndcg_at_3_std']:.3f}"
                ),
                "Low-signal score std": (
                    f"{row['low_signal_prediction_std_mean']:.4f}"
                ),
                "Not-low score std": (
                    f"{row['not_low_signal_prediction_std_mean']:.4f}"
                ),
                "Low-signal Top-3 agreement": (
                    f"{row['low_signal_top3_agreement_mean']:.3f}"
                ),
                "Not-low Top-3 agreement": (
                    f"{row['not_low_signal_top3_agreement_mean']:.3f}"
                ),
            }
        )

    paired_summary = paired_effects.groupby("scenario")[
        [
            "overall_ndcg_recovery",
            "low_signal_uncertainty_change",
            "low_signal_top3_agreement_change",
        ]
    ].mean()
    severe = paired_summary.loc["severe_signal_loss"]
    policy = paired_summary.loc["policy_restricted"]
    out_path.write_text(
        "# Disparate Uncertainty Audit\n\n"
        "This diagnostic fits household-bootstrap ensembles on a fixed holdout split. "
        "It reports ensemble-mean ranking utility, prediction variability, and the "
        "share of Top-3 services that agree with the ensemble-mean ranking.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Paired aggregate effects\n\n"
        + paired_summary.reset_index().to_markdown(
            index=False,
            floatfmt="+.4f",
        )
        + "\n\n"
        "## Current result\n\n"
        "Aggregate recovery improves ensemble-mean overall NDCG@3 by "
        f"`{severe['overall_ndcg_recovery']:+.4f}` under severe signal loss and "
        f"`{policy['overall_ndcg_recovery']:+.4f}` under policy restriction. "
        "However, mean low-signal prediction variability increases by "
        f"`{severe['low_signal_uncertainty_change']:+.4f}` and "
        f"`{policy['low_signal_uncertainty_change']:+.4f}`, respectively, while "
        "mean low-signal Top-3 agreement changes by "
        f"`{severe['low_signal_top3_agreement_change']:+.4f}` and "
        f"`{policy['low_signal_top3_agreement_change']:+.4f}`. The score-standard-"
        "deviation gap between groups is small and not consistently directional in "
        "this configuration.\n\n"
        "## Interpretation limits\n\n"
        "Bootstrap prediction standard deviation is a training-resample instability "
        "diagnostic. It is not a calibrated posterior, a confidence interval for an "
        "individual event, or an implementation of Equal-Opportunity Ranking. Top-3 "
        "agreement measures membership stability, not ranking correctness.\n"
    )


def main(seeds: Iterable[int] = SEEDS) -> None:
    raw = pd.concat([evaluate_seed(seed) for seed in seeds], ignore_index=True)
    summary = build_summary(raw)
    paired_effects = build_paired_effects(raw)
    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    docs_dir = Path("docs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(tables_dir / "disparate_uncertainty_audit_raw.csv", index=False)
    summary.to_csv(
        tables_dir / "disparate_uncertainty_audit_summary.csv",
        index=False,
    )
    paired_effects.to_csv(
        tables_dir / "disparate_uncertainty_audit_paired.csv",
        index=False,
    )
    plot_uncertainty_audit(
        summary,
        assets_dir / "disparate_uncertainty_audit.png",
    )
    write_markdown_summary(
        summary,
        paired_effects,
        docs_dir / "disparate_uncertainty_audit.md",
    )
    print(summary.round(4).to_string(index=False))
    print("\nPaired aggregate effects:")
    print(paired_effects.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
