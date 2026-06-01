from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from fairprivacysignal.privacy_transforms import add_privacy_safe_features
from fairprivacysignal.signal_loss import apply_signal_loss


EXPERIMENTS = [
    ("full_signal_raw_baseline", "full_signal", False, BASE_NUMERIC_FEATURES),
    ("severe_signal_loss_baseline", "severe_signal_loss", False, BASE_NUMERIC_FEATURES),
    (
        "severe_signal_loss_with_privacy_safe_aggregates",
        "severe_signal_loss",
        True,
        PRIVACY_SAFE_NUMERIC_FEATURES,
    ),
    ("policy_restricted_baseline", "policy_restricted", False, BASE_NUMERIC_FEATURES),
    (
        "policy_restricted_with_privacy_safe_aggregates",
        "policy_restricted",
        True,
        PRIVACY_SAFE_NUMERIC_FEATURES,
    ),
]


DISPLAY_NAME = {
    "full_signal_raw_baseline": "Full signal",
    "severe_signal_loss_baseline": "Severe loss",
    "severe_signal_loss_with_privacy_safe_aggregates": "Severe loss + privacy-safe",
    "policy_restricted_baseline": "Policy restricted",
    "policy_restricted_with_privacy_safe_aggregates": "Policy + privacy-safe",
}


def build_model(numeric_features: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def score_experiment(
    events: pd.DataFrame,
    experiment_name: str,
    signal_scenario: str,
    use_privacy_safe_features: bool,
    numeric_features: List[str],
    privacy_noise_seed: int = 42,
) -> pd.DataFrame:
    df = apply_signal_loss(events, signal_scenario)

    if use_privacy_safe_features:
        df = add_privacy_safe_features(df, seed=privacy_noise_seed)

    household_ids = df["household_id"].drop_duplicates()

    train_households, test_households = train_test_split(
        household_ids,
        test_size=0.30,
        random_state=42,
    )

    train = df[df["household_id"].isin(train_households)].copy()
    test = df[df["household_id"].isin(test_households)].copy()

    features = numeric_features + CATEGORICAL_FEATURES

    model = build_model(numeric_features)
    model.fit(train[features], train["relevant"])

    test["predicted_relevance"] = model.predict_proba(test[features])[:, 1]
    test["experiment"] = experiment_name

    return test


def allocate_with_capacity(
    scored_events: pd.DataFrame,
    capacity_rate: float = 0.15,
    allocation_policy: str = "utility_only",
    low_signal_floor_fraction: float = 0.80,
) -> pd.DataFrame:
    """
    Simulate limited public-service outreach capacity.

    utility_only:
        Select the top-ranked households for each service.

    fairness_constrained:
        Reserve a minimum share of each service's capacity for low-signal households.
        The floor is based on a fraction of the low-signal candidate share. This does
        not assume the fairness problem is solved; it exposes the utility/fairness
        tradeoff under capacity constraints.
    """
    frames = []

    for service_category, group in scored_events.groupby("service_category", observed=False):
        group = group.copy()
        capacity = max(1, int(len(group) * capacity_rate))

        group["allocated"] = False
        group["allocation_policy"] = allocation_policy
        group["service_capacity"] = capacity

        ranked = group.sort_values("predicted_relevance", ascending=False)

        if allocation_policy == "utility_only":
            selected_idx = ranked.head(capacity).index

        elif allocation_policy == "fairness_constrained":
            low_candidates = ranked[ranked["low_signal"].astype(bool)]
            not_low_candidates = ranked[~ranked["low_signal"].astype(bool)]

            low_candidate_share = len(low_candidates) / len(ranked)
            low_floor = int(round(capacity * low_candidate_share * low_signal_floor_fraction))
            low_floor = min(low_floor, len(low_candidates), capacity)

            selected_low_idx = low_candidates.head(low_floor).index
            remaining_capacity = capacity - len(selected_low_idx)

            remaining_candidates = ranked.drop(index=selected_low_idx)
            selected_remaining_idx = remaining_candidates.head(remaining_capacity).index

            selected_idx = selected_low_idx.union(selected_remaining_idx)

        else:
            raise ValueError(f"Unknown allocation_policy: {allocation_policy}")

        group.loc[selected_idx, "allocated"] = True
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def summarize_allocation(allocated: pd.DataFrame) -> dict:
    selected = allocated[allocated["allocated"]]
    low = allocated[allocated["low_signal"] == True]
    not_low = allocated[allocated["low_signal"] == False]

    selected_low = selected[selected["low_signal"] == True]
    selected_not_low = selected[selected["low_signal"] == False]

    overall_base_relevance = allocated["relevant"].mean()
    selected_relevance = selected["relevant"].mean()

    low_selection_rate = low["allocated"].mean()
    not_low_selection_rate = not_low["allocated"].mean()

    low_precision = selected_low["relevant"].mean()
    not_low_precision = selected_not_low["relevant"].mean()

    return {
        "experiment": allocated["experiment"].iloc[0],
        "avg_privacy_exposure_score": allocated["privacy_exposure_score"].mean(),
        "behavioral_available_share": allocated["behavioral_available"].mean(),
        "overall_base_relevance_rate": overall_base_relevance,
        "allocated_relevance_rate": selected_relevance,
        "allocation_relevance_lift": selected_relevance - overall_base_relevance,
        "low_signal_selection_rate": low_selection_rate,
        "not_low_signal_selection_rate": not_low_selection_rate,
        "selection_rate_gap_not_low_minus_low": not_low_selection_rate - low_selection_rate,
        "low_signal_allocated_precision": low_precision,
        "not_low_signal_allocated_precision": not_low_precision,
        "allocated_low_signal_share": selected["low_signal"].mean(),
        "overall_low_signal_share": allocated["low_signal"].mean(),
        "num_allocated": int(selected.shape[0]),
        "num_candidate_events": int(allocated.shape[0]),
    }


def plot_capacity_results(results: pd.DataFrame, assets_dir: Path) -> None:
    order = [x[0] for x in EXPERIMENTS]
    ordered = results.copy()
    ordered["scenario_label"] = ordered["experiment"].map(DISPLAY_NAME)
    ordered["experiment"] = pd.Categorical(ordered["experiment"], categories=order, ordered=True)
    ordered = ordered.sort_values(["experiment", "allocation_policy"])

    pivot_precision = ordered.pivot(
        index="scenario_label",
        columns="allocation_policy",
        values="allocated_relevance_rate",
    ).loc[[DISPLAY_NAME[x] for x in order]]

    pivot_gap = ordered.pivot(
        index="scenario_label",
        columns="allocation_policy",
        values="selection_rate_gap_not_low_minus_low",
    ).loc[[DISPLAY_NAME[x] for x in order]]

    pivot_share = ordered.pivot(
        index="scenario_label",
        columns="allocation_policy",
        values="allocated_low_signal_share",
    ).loc[[DISPLAY_NAME[x] for x in order]]

    # 1. Allocation precision: utility vs fairness-constrained.
    ax = pivot_precision[["utility_only", "fairness_constrained"]].plot(
        kind="bar",
        figsize=(10, 5),
        rot=25,
    )
    ax.set_ylabel("Allocated relevance rate")
    ax.set_xlabel("")
    ax.set_title("Capacity-constrained allocation: utility vs fairness constraint")
    ax.legend(["Utility-only allocation", "Fairness-constrained allocation"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plt.savefig(assets_dir / "capacity_allocation_precision.png", dpi=180)
    plt.close()

    # 2. Selection-rate gap: lower is more balanced.
    ax = pivot_gap[["utility_only", "fairness_constrained"]].plot(
        kind="bar",
        figsize=(10, 5),
        rot=25,
    )
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Selection-rate gap: not-low-signal minus low-signal")
    ax.set_xlabel("")
    ax.set_title("Capacity-constrained allocation gap for low-signal households")
    ax.legend(["Utility-only allocation", "Fairness-constrained allocation"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plt.savefig(assets_dir / "capacity_allocation_selection_gap.png", dpi=180)
    plt.close()

    # 3. Allocated low-signal share.
    ax = pivot_share[["utility_only", "fairness_constrained"]].plot(
        kind="bar",
        figsize=(10, 5),
        rot=25,
    )
    ax.set_ylabel("Share of allocated events from low-signal households")
    ax.set_xlabel("")
    ax.set_title("Low-signal representation among allocated outreach slots")
    ax.legend(["Utility-only allocation", "Fairness-constrained allocation"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plt.savefig(assets_dir / "capacity_allocation_low_signal_share.png", dpi=180)
    plt.close()


def main() -> None:
    data_dir = Path("data/synthetic")
    out_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_dir / "synthetic_outreach_events.csv")

    summaries = []

    for experiment_name, signal_scenario, use_privacy_safe_features, numeric_features in EXPERIMENTS:
        scored = score_experiment(
            events,
            experiment_name,
            signal_scenario,
            use_privacy_safe_features,
            numeric_features,
        )

        utility_allocated = allocate_with_capacity(
            scored,
            capacity_rate=0.15,
            allocation_policy="utility_only",
        )
        utility_summary = summarize_allocation(utility_allocated)
        utility_summary["allocation_policy"] = "utility_only"
        summaries.append(utility_summary)

        fairness_allocated = allocate_with_capacity(
            scored,
            capacity_rate=0.15,
            allocation_policy="fairness_constrained",
            low_signal_floor_fraction=0.80,
        )
        fairness_summary = summarize_allocation(fairness_allocated)
        fairness_summary["allocation_policy"] = "fairness_constrained"
        summaries.append(fairness_summary)

    results = pd.DataFrame(summaries)
    out_path = out_dir / "capacity_allocation_metrics.csv"
    results.to_csv(out_path, index=False)

    plot_capacity_results(results, assets_dir)

    print("Capacity-constrained allocation metrics:")
    print(results.round(4).to_string(index=False))
    print(f"\nWrote: {out_path}")
    print("- docs/assets/capacity_allocation_precision.png")
    print("- docs/assets/capacity_allocation_selection_gap.png")
    print("- docs/assets/capacity_allocation_low_signal_share.png")


if __name__ == "__main__":
    main()
