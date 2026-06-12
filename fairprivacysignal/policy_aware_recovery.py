from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from fairprivacysignal.data_generator import generate_all
from fairprivacysignal.privacy_recovery import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    PRIVACY_SAFE_NUMERIC_FEATURES,
    apply_train_fitted_privacy_safe_features,
    build_model,
    split_household_events,
)
from fairprivacysignal.ranking import average_ndcg_at_k, safe_auc
from fairprivacysignal.signal_loss import apply_signal_loss


SEEDS = [7, 11, 23, 42, 101]
SCENARIOS = ["severe_signal_loss", "policy_restricted"]

RECONSTRUCTION_CONTEXT_NUMERIC_FEATURES = [
    feature
    for feature in BASE_NUMERIC_FEATURES
    if feature != "available_historical_service_engagement_count"
]
RECONSTRUCTED_SIGNAL = "reconstructed_historical_service_engagement_count"
RECONSTRUCTION_APPLIED = "signal_reconstruction_applied"

VARIANT_ORDER = [
    "full_signal_oracle",
    "no_recovery",
    "missingness_indicator",
    "flat_privacy_safe_aggregates",
    "signal_reconstruction",
    "policy_aware_signal_recovery",
]

VARIANT_DISPLAY_NAMES = {
    "full_signal_oracle": "Full-signal oracle",
    "no_recovery": "Signal-loss baseline",
    "missingness_indicator": "Missingness indicator",
    "flat_privacy_safe_aggregates": "Flat aggregates",
    "signal_reconstruction": "Cross-fitted reconstruction",
    "policy_aware_signal_recovery": "Policy-aware recovery",
}

SCENARIO_DISPLAY_NAMES = {
    "severe_signal_loss": "Complete behavioral-signal loss",
    "policy_restricted": "Policy-restricted partial signal",
}


@dataclass(frozen=True)
class SignalRecoverySpec:
    """Column contract for adapting reconstruction to another ranking domain."""

    event_id_column: str
    group_column: str
    raw_signal_column: str
    available_signal_column: str
    availability_column: str
    reconstructed_signal_column: str
    context_numeric_features: Tuple[str, ...]
    categorical_features: Tuple[str, ...]
    signal_lower_bound: float
    signal_upper_bound: float


DEFAULT_RECOVERY_SPEC = SignalRecoverySpec(
    event_id_column="event_id",
    group_column="household_id",
    raw_signal_column="historical_service_engagement_count",
    available_signal_column="available_historical_service_engagement_count",
    availability_column="behavioral_available",
    reconstructed_signal_column=RECONSTRUCTED_SIGNAL,
    context_numeric_features=tuple(RECONSTRUCTION_CONTEXT_NUMERIC_FEATURES),
    categorical_features=tuple(CATEGORICAL_FEATURES),
    signal_lower_bound=0.0,
    signal_upper_bound=50.0,
)


def build_signal_reconstructor(
    random_state: int,
    spec: SignalRecoverySpec = DEFAULT_RECOVERY_SPEC,
) -> Pipeline:
    """Build the bounded-context model used to reconstruct the restricted signal."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
                list(spec.categorical_features),
            ),
        ],
        remainder="passthrough",
    )
    categorical_mask = [True] * len(spec.categorical_features) + [False] * len(
        spec.context_numeric_features
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    random_state=random_state,
                    max_iter=100,
                    max_leaf_nodes=15,
                    categorical_features=categorical_mask,
                ),
            ),
        ]
    )


def cross_fit_signal_reconstruction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    n_splits: int = 5,
    spec: SignalRecoverySpec = DEFAULT_RECOVERY_SPEC,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Create grouped out-of-fold train proxies and train-fitted holdout proxies."""
    features = list(spec.context_numeric_features + spec.categorical_features)
    target = spec.raw_signal_column
    groups = train[spec.group_column].to_numpy()
    unique_groups = pd.unique(groups)
    actual_splits = min(int(n_splits), len(unique_groups))

    if actual_splits < 2:
        raise ValueError("signal reconstruction requires at least two households")

    oof_prediction = np.zeros(len(train), dtype=float)
    splitter = GroupKFold(n_splits=actual_splits)

    for fold, (fit_positions, validation_positions) in enumerate(
        splitter.split(train, groups=groups)
    ):
        model = build_signal_reconstructor(
            random_state=seed + fold,
            spec=spec,
        )
        model.fit(
            train.iloc[fit_positions][features],
            train.iloc[fit_positions][target],
        )
        oof_prediction[validation_positions] = model.predict(
            train.iloc[validation_positions][features]
        )

    final_model = build_signal_reconstructor(random_state=seed, spec=spec)
    final_model.fit(train[features], train[target])
    test_prediction = final_model.predict(test[features])

    train_proxy = train[[spec.event_id_column]].copy()
    test_proxy = test[[spec.event_id_column]].copy()
    train_proxy[spec.reconstructed_signal_column] = np.clip(
        oof_prediction,
        spec.signal_lower_bound,
        spec.signal_upper_bound,
    )
    test_proxy[spec.reconstructed_signal_column] = np.clip(
        test_prediction,
        spec.signal_lower_bound,
        spec.signal_upper_bound,
    )

    diagnostics = {
        "reconstruction_oof_mae": float(
            np.mean(np.abs(oof_prediction - train[target].to_numpy(dtype=float)))
        ),
        "reconstruction_oof_correlation": _safe_correlation(
            oof_prediction,
            train[target].to_numpy(dtype=float),
        ),
        "reconstruction_folds": float(actual_splits),
    }
    return train_proxy, test_proxy, diagnostics


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def attach_reconstructed_signal(
    events: pd.DataFrame,
    proxy: pd.DataFrame,
    spec: SignalRecoverySpec = DEFAULT_RECOVERY_SPEC,
) -> pd.DataFrame:
    """Retain permitted values and substitute proxies only where signal is hidden."""
    transformed = events.merge(
        proxy,
        on=spec.event_id_column,
        how="left",
        validate="one_to_one",
    )

    if transformed[spec.reconstructed_signal_column].isna().any():
        raise ValueError("signal reconstruction did not cover every event")

    available = transformed[spec.availability_column].astype(bool)
    transformed[RECONSTRUCTION_APPLIED] = (~available).astype(float)
    transformed[spec.reconstructed_signal_column] = np.where(
        available,
        transformed[spec.available_signal_column],
        transformed[spec.reconstructed_signal_column],
    )
    transformed[spec.availability_column] = available.astype(float)
    return transformed


def _score_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: List[str],
    experiment: str,
    scenario: str,
    diagnostics: Dict[str, float],
) -> Dict[str, float]:
    features = numeric_features + CATEGORICAL_FEATURES
    model = build_model(numeric_features)
    model.fit(train[features], train["relevant"])

    scored = test.copy()
    scored["predicted_relevance"] = model.predict_proba(scored[features])[:, 1]
    low_signal = scored[scored["low_signal"].astype(bool)]
    not_low_signal = scored[~scored["low_signal"].astype(bool)]
    low_signal_ndcg = average_ndcg_at_k(low_signal, k=3)
    not_low_signal_ndcg = average_ndcg_at_k(not_low_signal, k=3)

    return {
        "scenario": scenario,
        "variant": experiment,
        "overall_auc": safe_auc(scored["relevant"], scored["predicted_relevance"]),
        "overall_ndcg_at_3": average_ndcg_at_k(scored, k=3),
        "low_signal_ndcg_at_3": low_signal_ndcg,
        "not_low_signal_ndcg_at_3": not_low_signal_ndcg,
        "ndcg_gap_not_low_minus_low": not_low_signal_ndcg - low_signal_ndcg,
        "avg_privacy_exposure_score": float(
            scored["privacy_exposure_score"].mean()
        ),
        "behavioral_available_share": float(
            scored["behavioral_available"].mean()
        ),
        "reconstruction_applied_share": float(
            scored.get(
                RECONSTRUCTION_APPLIED,
                pd.Series(0.0, index=scored.index),
            ).mean()
        ),
        **diagnostics,
    }


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    raw_train, raw_test = split_household_events(events)
    train_proxy, test_proxy, diagnostics = cross_fit_signal_reconstruction(
        raw_train,
        raw_test,
        seed=seed,
    )

    rows = []
    full_train = apply_signal_loss(raw_train, "full_signal")
    full_test = apply_signal_loss(raw_test, "full_signal")

    for scenario in SCENARIOS:
        masked_train = apply_signal_loss(raw_train, scenario)
        masked_test = apply_signal_loss(raw_test, scenario)

        rows.append(
            _score_variant(
                full_train,
                full_test,
                BASE_NUMERIC_FEATURES,
                "full_signal_oracle",
                scenario,
                diagnostics,
            )
        )
        rows.append(
            _score_variant(
                masked_train,
                masked_test,
                BASE_NUMERIC_FEATURES,
                "no_recovery",
                scenario,
                diagnostics,
            )
        )

        indicator_features = BASE_NUMERIC_FEATURES + ["behavioral_available"]
        rows.append(
            _score_variant(
                masked_train,
                masked_test,
                indicator_features,
                "missingness_indicator",
                scenario,
                diagnostics,
            )
        )

        aggregate_train, aggregate_test = apply_train_fitted_privacy_safe_features(
            masked_train,
            masked_test,
            privacy_safe_feature_options={"seed": seed},
        )
        rows.append(
            _score_variant(
                aggregate_train,
                aggregate_test,
                PRIVACY_SAFE_NUMERIC_FEATURES,
                "flat_privacy_safe_aggregates",
                scenario,
                diagnostics,
            )
        )

        reconstructed_train = attach_reconstructed_signal(
            masked_train,
            train_proxy,
        )
        reconstructed_test = attach_reconstructed_signal(
            masked_test,
            test_proxy,
        )
        reconstruction_features = (
            RECONSTRUCTION_CONTEXT_NUMERIC_FEATURES
            + [
                RECONSTRUCTED_SIGNAL,
                "behavioral_available",
                RECONSTRUCTION_APPLIED,
            ]
        )
        rows.append(
            _score_variant(
                reconstructed_train,
                reconstructed_test,
                reconstruction_features,
                "signal_reconstruction",
                scenario,
                diagnostics,
            )
        )

        if masked_train["behavioral_available"].mean() > 0.0:
            policy_train = attach_reconstructed_signal(
                aggregate_train,
                train_proxy,
            )
            policy_test = attach_reconstructed_signal(
                aggregate_test,
                test_proxy,
            )
            policy_features = [
                feature
                for feature in PRIVACY_SAFE_NUMERIC_FEATURES
                if feature != "available_historical_service_engagement_count"
            ] + [
                RECONSTRUCTED_SIGNAL,
                "behavioral_available",
                RECONSTRUCTION_APPLIED,
            ]
        else:
            policy_train = aggregate_train
            policy_test = aggregate_test
            policy_features = PRIVACY_SAFE_NUMERIC_FEATURES

        rows.append(
            _score_variant(
                policy_train,
                policy_test,
                policy_features,
                "policy_aware_signal_recovery",
                scenario,
                diagnostics,
            )
        )

    result = pd.DataFrame(rows)
    result["seed"] = seed
    return result


def run_policy_aware_recovery(seeds: Iterable[int] = SEEDS) -> pd.DataFrame:
    return pd.concat(
        [evaluate_seed(int(seed)) for seed in seeds],
        ignore_index=True,
    )


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_auc",
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "not_low_signal_ndcg_at_3",
        "ndcg_gap_not_low_minus_low",
        "avg_privacy_exposure_score",
        "behavioral_available_share",
        "reconstruction_applied_share",
        "reconstruction_oof_mae",
        "reconstruction_oof_correlation",
    ]
    summary = raw.groupby(["scenario", "variant"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    paired_rows = []
    for scenario in SCENARIOS:
        scenario_raw = raw[raw["scenario"] == scenario]
        baseline = scenario_raw[scenario_raw["variant"] == "no_recovery"][
            ["seed", "overall_ndcg_at_3", "low_signal_ndcg_at_3"]
        ].rename(
            columns={
                "overall_ndcg_at_3": "baseline_overall_ndcg_at_3",
                "low_signal_ndcg_at_3": "baseline_low_signal_ndcg_at_3",
            }
        )
        oracle = scenario_raw[scenario_raw["variant"] == "full_signal_oracle"][
            ["seed", "overall_ndcg_at_3", "low_signal_ndcg_at_3"]
        ].rename(
            columns={
                "overall_ndcg_at_3": "oracle_overall_ndcg_at_3",
                "low_signal_ndcg_at_3": "oracle_low_signal_ndcg_at_3",
            }
        )

        for variant in VARIANT_ORDER:
            variant_rows = scenario_raw[scenario_raw["variant"] == variant][
                ["seed", "overall_ndcg_at_3", "low_signal_ndcg_at_3"]
            ]
            paired = variant_rows.merge(baseline, on="seed").merge(oracle, on="seed")
            overall_recovery = (
                paired["overall_ndcg_at_3"]
                - paired["baseline_overall_ndcg_at_3"]
            )
            low_signal_recovery = (
                paired["low_signal_ndcg_at_3"]
                - paired["baseline_low_signal_ndcg_at_3"]
            )
            overall_gap = (
                paired["oracle_overall_ndcg_at_3"]
                - paired["baseline_overall_ndcg_at_3"]
            )
            low_signal_gap = (
                paired["oracle_low_signal_ndcg_at_3"]
                - paired["baseline_low_signal_ndcg_at_3"]
            )
            paired_rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "overall_recovery_mean": float(overall_recovery.mean()),
                    "overall_recovery_std": float(overall_recovery.std()),
                    "low_signal_recovery_mean": float(low_signal_recovery.mean()),
                    "low_signal_recovery_std": float(low_signal_recovery.std()),
                    "overall_gap_closed_mean": float(
                        np.mean(np.divide(overall_recovery, overall_gap))
                    ),
                    "low_signal_gap_closed_mean": float(
                        np.mean(np.divide(low_signal_recovery, low_signal_gap))
                    ),
                    "overall_positive_seed_share": float(
                        np.mean(overall_recovery > 0.0)
                    ),
                    "low_signal_positive_seed_share": float(
                        np.mean(low_signal_recovery > 0.0)
                    ),
                }
            )

    paired_summary = pd.DataFrame(paired_rows)
    return summary.merge(
        paired_summary,
        on=["scenario", "variant"],
        how="left",
        validate="one_to_one",
    )


def plot_recovery(summary: pd.DataFrame, out_path: Path) -> None:
    colors = {
        "full_signal_oracle": "#1f2937",
        "no_recovery": "#9ca3af",
        "missingness_indicator": "#c08457",
        "flat_privacy_safe_aggregates": "#0f766e",
        "signal_reconstruction": "#2563eb",
        "policy_aware_signal_recovery": "#7c3aed",
    }
    shown_variants = [
        "full_signal_oracle",
        "no_recovery",
        "flat_privacy_safe_aggregates",
        "policy_aware_signal_recovery",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)

    for axis, scenario in zip(axes, SCENARIOS):
        indexed = (
            summary[summary["scenario"] == scenario]
            .set_index("variant")
            .loc[shown_variants]
        )
        x = np.arange(len(shown_variants))
        values = indexed["overall_ndcg_at_3_mean"].to_numpy(dtype=float)
        errors = indexed["overall_ndcg_at_3_std"].to_numpy(dtype=float)

        baseline = indexed.loc["no_recovery", "overall_ndcg_at_3_mean"]
        oracle = indexed.loc["full_signal_oracle", "overall_ndcg_at_3_mean"]
        axis.axhspan(baseline, oracle, color="#eef2ff", alpha=0.8, zorder=0)
        axis.axhline(oracle, color=colors["full_signal_oracle"], linewidth=1.1)
        axis.plot(x, values, color="#64748b", linewidth=1.4, zorder=2)
        axis.errorbar(
            x,
            values,
            yerr=errors,
            fmt="none",
            ecolor="#64748b",
            capsize=4,
            linewidth=1,
            zorder=2,
        )

        for position, variant, value in zip(x, shown_variants, values):
            axis.scatter(
                position,
                value,
                s=110 if variant == "policy_aware_signal_recovery" else 80,
                color=colors[variant],
                edgecolor="white",
                linewidth=1,
                zorder=3,
            )
            label = f"{value:.3f}"
            if variant in {
                "flat_privacy_safe_aggregates",
                "policy_aware_signal_recovery",
            }:
                gap_closed = indexed.loc[variant, "overall_gap_closed_mean"]
                label += f"\n{gap_closed:.0%} gap closed"
            axis.annotate(
                label,
                (position, value),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#172033",
                fontweight=(
                    "bold"
                    if variant == "policy_aware_signal_recovery"
                    else "normal"
                ),
            )

        axis.set_xticks(
            x,
            [
                "Full-signal\noracle",
                "Signal-loss\nbaseline",
                "Flat privacy-safe\naggregates",
                "Policy-aware\nrecovery",
            ],
        )
        axis.set_title(
            SCENARIO_DISPLAY_NAMES[scenario],
            loc="left",
            fontsize=12,
            fontweight="bold",
            color="#172033",
        )
        axis.grid(axis="y", color="#dbe3ec", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#cbd5e1")
        axis.spines["bottom"].set_color("#cbd5e1")

    axes[0].set_ylabel("Overall NDCG@3, mean +/- standard deviation")
    fig.suptitle(
        "Recovering ranking utility without restoring raw behavioral signal at serving time",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.06,
        0.92,
        "Five paired synthetic-data seeds; shaded band is the utility lost between the no-recovery baseline and full-signal oracle.",
        ha="left",
        fontsize=9,
        color="#526274",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.88])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    ordered = (
        summary.set_index(["scenario", "variant"])
        .loc[
            pd.MultiIndex.from_product(
                [SCENARIOS, VARIANT_ORDER],
                names=["scenario", "variant"],
            )
        ]
        .reset_index()
    )
    for _, row in ordered.iterrows():
        rows.append(
            {
                "Signal-loss regime": SCENARIO_DISPLAY_NAMES[row["scenario"]],
                "Method": VARIANT_DISPLAY_NAMES[row["variant"]],
                "Overall NDCG@3": (
                    f"{row['overall_ndcg_at_3_mean']:.3f} +/- "
                    f"{row['overall_ndcg_at_3_std']:.3f}"
                ),
                "Low-signal NDCG@3": (
                    f"{row['low_signal_ndcg_at_3_mean']:.3f} +/- "
                    f"{row['low_signal_ndcg_at_3_std']:.3f}"
                ),
                "Overall gap closed": (
                    f"{row['overall_gap_closed_mean']:.1%}"
                    if row["variant"] not in {"full_signal_oracle", "no_recovery"}
                    else "-"
                ),
                "Positive seeds": (
                    f"{row['overall_positive_seed_share']:.0%}"
                    if row["variant"] not in {"full_signal_oracle", "no_recovery"}
                    else "-"
                ),
            }
        )

    method_rows = summary[
        summary["variant"] == "policy_aware_signal_recovery"
    ].set_index("scenario")
    severe = method_rows.loc["severe_signal_loss"]
    policy = method_rows.loc["policy_restricted"]

    content = (
        "# Policy-Aware Signal Recovery\n\n"
        "Policy-Aware Signal Recovery is the repository's primary recovery "
        "method for ranking systems whose behavioral features are available during "
        "controlled offline training but unavailable or partially unavailable when "
        "the ranker is served.\n\n"
        "## Method\n\n"
        "1. A reconstruction model learns the training-only behavioral signal from "
        "policy-permitted context and candidate features.\n"
        "2. Household-grouped cross-fitting produces out-of-fold reconstructed "
        "signals for downstream model training, preventing each training event from "
        "being reconstructed by a model fitted on that event.\n"
        "3. At serving time, observed behavioral values are retained when policy "
        "allows them; unavailable values are replaced by the learned reconstruction.\n"
        "4. Under partial restriction, the reconstruction is fused with train-fitted, "
        "thresholded, noise-stressed cohort aggregates. Under complete loss, the "
        "method uses the more stable aggregate path because no observed event-level "
        "signal remains to anchor per-event reconstruction.\n"
        "5. The downstream ranker never receives the hidden raw behavioral value for "
        "an unavailable event.\n\n"
        "![Policy-aware signal recovery results](assets/policy_aware_signal_recovery.png)\n\n"
        "## Paired results\n\n"
        + pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "Across the five paired synthetic draws, policy-aware recovery improves "
        f"overall NDCG@3 by `{severe['overall_recovery_mean']:+.3f}` under complete "
        f"signal loss and `{policy['overall_recovery_mean']:+.3f}` under partial "
        "policy restriction relative to the matching no-recovery baselines. These "
        f"changes close `{severe['overall_gap_closed_mean']:.1%}` and "
        f"`{policy['overall_gap_closed_mean']:.1%}` of the respective synthetic "
        "full-signal utility gaps.\n\n"
        "## Applicability boundary\n\n"
        "This method applies when historical behavioral signal may be used inside a "
        "controlled offline training process but may not be exposed to the serving "
        "ranker. If a policy prohibits use of the signal even during offline model "
        "fitting, the reconstruction path is not applicable and the aggregate-only "
        "path remains the appropriate comparator. The implementation does not claim "
        "formal differential privacy or immunity to model-extraction attacks.\n"
    )
    out_path.write_text(content)


def main() -> None:
    outputs_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    docs_dir = Path("docs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    raw = run_policy_aware_recovery()
    summary = build_summary(raw)

    raw_path = outputs_dir / "policy_aware_signal_recovery_raw.csv"
    summary_path = outputs_dir / "policy_aware_signal_recovery_summary.csv"
    figure_path = assets_dir / "policy_aware_signal_recovery.png"
    markdown_path = docs_dir / "policy_aware_signal_recovery.md"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_recovery(summary, figure_path)
    write_markdown_summary(summary, markdown_path)

    print("Policy-aware signal recovery summary:")
    print(
        summary[
            summary["variant"].isin(
                [
                    "no_recovery",
                    "flat_privacy_safe_aggregates",
                    "policy_aware_signal_recovery",
                ]
            )
        ][
            [
                "scenario",
                "variant",
                "overall_ndcg_at_3_mean",
                "low_signal_ndcg_at_3_mean",
                "overall_recovery_mean",
                "overall_gap_closed_mean",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    print(f"- {figure_path}")
    print(f"- {markdown_path}")


if __name__ == "__main__":
    main()
