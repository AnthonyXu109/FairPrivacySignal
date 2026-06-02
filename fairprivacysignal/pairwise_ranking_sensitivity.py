from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


SEEDS = [7, 42, 101]

EXPERIMENTS = {
    "full_signal_raw_baseline": {
        "display_name": "Full signal",
        "scenario": "full_signal",
        "use_privacy_safe_features": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "severe_signal_loss_baseline": {
        "display_name": "Severe loss",
        "scenario": "severe_signal_loss",
        "use_privacy_safe_features": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "severe_signal_loss_with_privacy_safe_aggregates": {
        "display_name": "Severe loss\n+ aggregates",
        "scenario": "severe_signal_loss",
        "use_privacy_safe_features": True,
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
    },
    "policy_restricted_baseline": {
        "display_name": "Policy restricted",
        "scenario": "policy_restricted",
        "use_privacy_safe_features": False,
        "numeric_features": BASE_NUMERIC_FEATURES,
    },
    "policy_restricted_with_privacy_safe_aggregates": {
        "display_name": "Policy restricted\n+ aggregates",
        "scenario": "policy_restricted",
        "use_privacy_safe_features": True,
        "numeric_features": PRIVACY_SAFE_NUMERIC_FEATURES,
    },
}

RECOVERY_PAIRS = {
    "severe_signal_loss": {
        "display_name": "Severe signal loss",
        "baseline": "severe_signal_loss_baseline",
        "aggregates": "severe_signal_loss_with_privacy_safe_aggregates",
    },
    "policy_restricted": {
        "display_name": "Policy restricted",
        "baseline": "policy_restricted_baseline",
        "aggregates": "policy_restricted_with_privacy_safe_aggregates",
    },
}

OBJECTIVES = {
    "pointwise_logistic": {
        "display_name": "Pointwise logistic",
        "color": "#0f766e",
        "marker": "o",
        "annotation_offset": (-14, -24),
    },
    "linear_pairwise": {
        "display_name": "Linear pairwise ranker",
        "color": "#ea580c",
        "marker": "s",
        "annotation_offset": (0, 14),
    },
    "linear_listwise": {
        "display_name": "Linear listwise ranker",
        "color": "#2563eb",
        "marker": "^",
        "annotation_offset": (14, -7),
    },
}


def _dense_array(values) -> np.ndarray:
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


def _build_linear_preprocessor(numeric_features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


class LinearPairwiseRanker:
    """Lightweight linear pairwise ranking comparator for sensitivity analysis."""

    def __init__(self, numeric_features: List[str]) -> None:
        self.numeric_features = numeric_features
        self.features = numeric_features + CATEGORICAL_FEATURES
        self.preprocessor = _build_linear_preprocessor(numeric_features)
        self.classifier = LogisticRegression(
            max_iter=1000,
            fit_intercept=False,
        )
        self.num_training_pairs_ = 0

    def fit(self, events: pd.DataFrame) -> "LinearPairwiseRanker":
        prepared = events.reset_index(drop=True)
        encoded = _dense_array(
            self.preprocessor.fit_transform(prepared[self.features])
        )
        pair_differences = []

        for positions in prepared.groupby("household_id", sort=False).indices.values():
            group_positions = np.asarray(positions, dtype=int)
            relevant_mask = prepared.iloc[group_positions]["relevant"].to_numpy(
                dtype=bool
            )
            positive_positions = group_positions[relevant_mask]
            negative_positions = group_positions[~relevant_mask]

            if not len(positive_positions) or not len(negative_positions):
                continue

            differences = (
                encoded[positive_positions, np.newaxis, :]
                - encoded[np.newaxis, negative_positions, :]
            ).reshape(-1, encoded.shape[1])
            pair_differences.append(differences)

        if not pair_differences:
            raise ValueError("pairwise ranking requires at least one ordered pair")

        positive_differences = np.vstack(pair_differences)
        training_features = np.vstack(
            [positive_differences, -positive_differences]
        )
        training_labels = np.concatenate(
            [
                np.ones(len(positive_differences), dtype=int),
                np.zeros(len(positive_differences), dtype=int),
            ]
        )
        self.classifier.fit(training_features, training_labels)
        self.num_training_pairs_ = int(len(positive_differences))
        return self

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        encoded = _dense_array(
            self.preprocessor.transform(events[self.features])
        )
        decision = self.classifier.decision_function(encoded)
        positive = 1.0 / (1.0 + np.exp(-np.clip(decision, -30.0, 30.0)))
        return np.column_stack([1.0 - positive, positive])


class LinearListwiseRanker:
    """Lightweight linear top-one listwise comparator for sensitivity analysis."""

    def __init__(
        self,
        numeric_features: List[str],
        l2_strength: float = 1e-3,
    ) -> None:
        self.numeric_features = numeric_features
        self.features = numeric_features + CATEGORICAL_FEATURES
        self.preprocessor = _build_linear_preprocessor(numeric_features)
        self.l2_strength = float(l2_strength)
        self.num_training_lists_ = 0

    def fit(self, events: pd.DataFrame) -> "LinearListwiseRanker":
        prepared = events.reset_index(drop=True)
        encoded = _dense_array(
            self.preprocessor.fit_transform(prepared[self.features])
        )
        comparable_positions = []
        target_probabilities = []
        group_sizes = []

        for positions in prepared.groupby("household_id", sort=False).indices.values():
            group_positions = np.asarray(positions, dtype=int)
            relevance = prepared.iloc[group_positions]["relevant"].to_numpy(
                dtype=float
            )

            if relevance.min() == relevance.max():
                continue

            comparable_positions.append(group_positions)
            target_probabilities.append(_softmax(relevance))
            group_sizes.append(len(group_positions))

        if not comparable_positions:
            raise ValueError("listwise ranking requires at least one comparable list")

        list_features = encoded[np.concatenate(comparable_positions)]
        targets = np.concatenate(target_probabilities)
        sizes = np.asarray(group_sizes, dtype=int)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.num_training_lists_ = int(len(sizes))

        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            scores = list_features @ weights
            maximums = np.maximum.reduceat(scores, starts)
            shifted = scores - np.repeat(maximums, sizes)
            exponentiated = np.exp(shifted)
            denominators = np.add.reduceat(exponentiated, starts)
            repeated_denominators = np.repeat(denominators, sizes)
            probabilities = exponentiated / repeated_denominators
            log_probabilities = shifted - np.log(repeated_denominators)
            loss = (
                -float(targets @ log_probabilities) / self.num_training_lists_
                + 0.5 * self.l2_strength * float(weights @ weights)
            )
            gradient = (
                list_features.T @ (probabilities - targets)
            ) / self.num_training_lists_ + self.l2_strength * weights
            return loss, np.asarray(gradient)

        result = minimize(
            objective,
            np.zeros(list_features.shape[1], dtype=float),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        if not result.success:
            raise RuntimeError(f"listwise optimization failed: {result.message}")

        self.coef_ = np.asarray(result.x)
        return self

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        encoded = _dense_array(
            self.preprocessor.transform(events[self.features])
        )
        decision = encoded @ self.coef_
        positive = 1.0 / (1.0 + np.exp(-np.clip(decision, -30.0, 30.0)))
        return np.column_stack([1.0 - positive, positive])


def _score_pointwise(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: List[str],
) -> tuple[pd.DataFrame, int, int]:
    features = numeric_features + CATEGORICAL_FEATURES
    model = build_model(numeric_features)
    model.fit(train[features], train["relevant"])

    scored = test.copy()
    scored["predicted_relevance"] = model.predict_proba(scored[features])[:, 1]
    return scored, 0, 0


def _score_pairwise(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: List[str],
) -> tuple[pd.DataFrame, int, int]:
    model = LinearPairwiseRanker(numeric_features)
    model.fit(train)

    scored = test.copy()
    scored["predicted_relevance"] = model.predict_proba(scored)[:, 1]
    return scored, model.num_training_pairs_, 0


def _score_listwise(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: List[str],
) -> tuple[pd.DataFrame, int, int]:
    model = LinearListwiseRanker(numeric_features)
    model.fit(train)

    scored = test.copy()
    scored["predicted_relevance"] = model.predict_proba(scored)[:, 1]
    return scored, 0, model.num_training_lists_


def _summarize_scored(
    scored: pd.DataFrame,
    seed: int,
    objective: str,
    experiment: str,
    num_training_pairs: int,
    num_training_lists: int,
    aggregate_reference_scope: str,
) -> dict:
    low_signal = scored[scored["low_signal"].astype(bool)]
    not_low_signal = scored[~scored["low_signal"].astype(bool)]
    low_signal_ndcg = average_ndcg_at_k(low_signal, k=3)
    not_low_signal_ndcg = average_ndcg_at_k(not_low_signal, k=3)
    return {
        "seed": int(seed),
        "objective": objective,
        "objective_display_name": OBJECTIVES[objective]["display_name"],
        "experiment": experiment,
        "experiment_display_name": EXPERIMENTS[experiment]["display_name"],
        "overall_auc": safe_auc(scored["relevant"], scored["predicted_relevance"]),
        "overall_ndcg_at_3": average_ndcg_at_k(scored, k=3),
        "low_signal_ndcg_at_3": low_signal_ndcg,
        "not_low_signal_ndcg_at_3": not_low_signal_ndcg,
        "ndcg_gap_not_low_minus_low": not_low_signal_ndcg - low_signal_ndcg,
        "num_training_pairs": int(num_training_pairs),
        "num_training_lists": int(num_training_lists),
        "aggregate_reference_scope": aggregate_reference_scope,
    }


def run_pairwise_ranking_sensitivity(
    events: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    scenario_frames = {
        scenario: apply_signal_loss(events, scenario)
        for scenario in ["full_signal", "severe_signal_loss", "policy_restricted"]
    }
    rows = []

    for experiment, metadata in EXPERIMENTS.items():
        train, test = split_household_events(scenario_frames[metadata["scenario"]])
        aggregate_reference_scope = "not_applicable"

        if metadata["use_privacy_safe_features"]:
            train, test = apply_train_fitted_privacy_safe_features(
                train,
                test,
                privacy_safe_feature_options={"seed": seed},
            )
            aggregate_reference_scope = "train_households_only"

        for objective, scorer in [
            ("pointwise_logistic", _score_pointwise),
            ("linear_pairwise", _score_pairwise),
            ("linear_listwise", _score_listwise),
        ]:
            scored, num_training_pairs, num_training_lists = scorer(
                train,
                test,
                metadata["numeric_features"],
            )
            rows.append(
                _summarize_scored(
                    scored,
                    seed=seed,
                    objective=objective,
                    experiment=experiment,
                    num_training_pairs=num_training_pairs,
                    num_training_lists=num_training_lists,
                    aggregate_reference_scope=aggregate_reference_scope,
                )
            )

    return pd.DataFrame(rows)


def evaluate_seed(seed: int) -> pd.DataFrame:
    _, _, _, events = generate_all(
        n_communities=120,
        n_households=10000,
        seed=seed,
    )
    return run_pairwise_ranking_sensitivity(events, seed=seed)


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_auc",
        "overall_ndcg_at_3",
        "low_signal_ndcg_at_3",
        "not_low_signal_ndcg_at_3",
        "ndcg_gap_not_low_minus_low",
        "num_training_pairs",
        "num_training_lists",
    ]
    summary = (
        raw.groupby(
            [
                "objective",
                "objective_display_name",
                "experiment",
                "experiment_display_name",
            ]
        )[metrics]
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


def build_paired_recovery_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scenario, metadata in RECOVERY_PAIRS.items():
        baseline = raw[raw["experiment"] == metadata["baseline"]][
            [
                "seed",
                "objective",
                "objective_display_name",
                "overall_ndcg_at_3",
                "low_signal_ndcg_at_3",
            ]
        ].rename(
            columns={
                "overall_ndcg_at_3": "baseline_overall_ndcg_at_3",
                "low_signal_ndcg_at_3": "baseline_low_signal_ndcg_at_3",
            }
        )
        aggregates = raw[raw["experiment"] == metadata["aggregates"]][
            [
                "seed",
                "objective",
                "overall_ndcg_at_3",
                "low_signal_ndcg_at_3",
            ]
        ].rename(
            columns={
                "overall_ndcg_at_3": "aggregate_overall_ndcg_at_3",
                "low_signal_ndcg_at_3": "aggregate_low_signal_ndcg_at_3",
            }
        )
        paired = baseline.merge(
            aggregates,
            on=["seed", "objective"],
            how="inner",
            validate="one_to_one",
        )
        paired["scenario"] = scenario
        paired["scenario_display_name"] = metadata["display_name"]
        paired["overall_recovery"] = (
            paired["aggregate_overall_ndcg_at_3"]
            - paired["baseline_overall_ndcg_at_3"]
        )
        paired["low_signal_recovery"] = (
            paired["aggregate_low_signal_ndcg_at_3"]
            - paired["baseline_low_signal_ndcg_at_3"]
        )
        rows.append(paired)

    paired_raw = pd.concat(rows, ignore_index=True)
    summary = (
        paired_raw.groupby(
            [
                "objective",
                "objective_display_name",
                "scenario",
                "scenario_display_name",
            ]
        )[["overall_recovery", "low_signal_recovery"]]
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
    paired_recovery: pd.DataFrame,
    metric: str,
    recovery_metric: str,
    title: str,
) -> None:
    x = np.arange(len(EXPERIMENTS), dtype=float)

    for objective, metadata in OBJECTIVES.items():
        objective_rows = (
            summary[summary["objective"] == objective]
            .set_index("experiment")
            .loc[list(EXPERIMENTS)]
        )
        mean = objective_rows[f"{metric}_mean"].to_numpy(dtype=float)
        std = objective_rows[f"{metric}_std"].to_numpy(dtype=float)

        axis.errorbar(
            x,
            mean,
            yerr=std,
            color=metadata["color"],
            marker=metadata["marker"],
            linewidth=2.4,
            elinewidth=1.4,
            capsize=4,
            markersize=7,
            label=metadata["display_name"],
        )

        recovery_rows = paired_recovery[
            paired_recovery["objective"] == objective
        ].set_index("scenario")
        for scenario, x_position in [
            ("severe_signal_loss", 2),
            ("policy_restricted", 4),
        ]:
            delta = recovery_rows.loc[scenario, f"{recovery_metric}_mean"]
            horizontal_offset, vertical_offset = metadata["annotation_offset"]
            axis.annotate(
                f"{delta:+.3f}",
                xy=(x_position, mean[x_position]),
                xytext=(horizontal_offset, vertical_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=metadata["color"],
                fontweight="bold",
            )

    axis.set_xticks(
        x,
        [metadata["display_name"] for metadata in EXPERIMENTS.values()],
    )
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#0f172a")
    axis.set_ylabel("NDCG@3, mean +/- std")
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)


def plot_pairwise_ranking_sensitivity(
    summary: pd.DataFrame,
    paired_recovery: pd.DataFrame,
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
    fig.patch.set_facecolor("#f8fafc")
    for axis in axes:
        axis.set_facecolor("#f8fafc")

    _plot_metric(
        axes[0],
        summary,
        paired_recovery,
        metric="overall_ndcg_at_3",
        recovery_metric="overall_recovery",
        title="Overall ranking utility",
    )
    _plot_metric(
        axes[1],
        summary,
        paired_recovery,
        metric="low_signal_ndcg_at_3",
        recovery_metric="low_signal_recovery",
        title="Low-signal ranking utility",
    )
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Ranking-objective sensitivity: pointwise, pairwise, and listwise training",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.055,
        0.91,
        "The lightweight comparators learn from ordered pairs or complete service lists within each synthetic household; the pointwise logistic model remains the primary baseline.",
        ha="left",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(
        0.055,
        0.035,
        "Labels at aggregate points report paired NDCG@3 recovery versus the same-objective, "
        "same-seed signal-loss baseline. Mean +/- standard deviation across seeds 7, 42, and 101.",
        ha="left",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=(0.03, 0.11, 0.99, 0.86))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    summary: pd.DataFrame,
    paired_recovery: pd.DataFrame,
    out_path: Path,
) -> None:
    recovery_rows = []

    for objective in OBJECTIVES:
        indexed = paired_recovery[
            paired_recovery["objective"] == objective
        ].set_index("scenario")
        for scenario in RECOVERY_PAIRS:
            row = indexed.loc[scenario]
            recovery_rows.append(
                {
                    "Training objective": row["objective_display_name"],
                    "Scenario": row["scenario_display_name"],
                    "Overall recovery": (
                        f"{row['overall_recovery_mean']:+.3f} +/- "
                        f"{row['overall_recovery_std']:.3f}"
                    ),
                    "Low-signal recovery": (
                        f"{row['low_signal_recovery_mean']:+.3f} +/- "
                        f"{row['low_signal_recovery_std']:.3f}"
                    ),
                }
            )

    score_rows = []
    for objective in OBJECTIVES:
        indexed = summary[
            summary["objective"] == objective
        ].set_index("experiment")
        for experiment in EXPERIMENTS:
            row = indexed.loc[experiment]
            score_rows.append(
                {
                    "Training objective": row["objective_display_name"],
                    "Scenario": row["experiment_display_name"].replace("\n", " "),
                    "Overall NDCG@3": (
                        f"{row['overall_ndcg_at_3_mean']:.3f} +/- "
                        f"{row['overall_ndcg_at_3_std']:.3f}"
                    ),
                    "Low-signal NDCG@3": (
                        f"{row['low_signal_ndcg_at_3_mean']:.3f} +/- "
                        f"{row['low_signal_ndcg_at_3_std']:.3f}"
                    ),
                }
            )

    out_path.write_text(
        "# Ranking-Objective Sensitivity Diagnostic\n\n"
        "This diagnostic compares the interpretable pointwise logistic primary "
        "baseline with lightweight linear pairwise and listwise rankers. The pairwise "
        "comparator creates "
        "ordered relevant-versus-nonrelevant service pairs within each synthetic "
        "household, then learns a linear score from feature differences. The listwise "
        "comparator uses each household's complete candidate-service list as one "
        "training instance and optimizes a top-one softmax cross-entropy loss.\n\n"
        "All three objectives receive the same synthetic-data draws, household-level "
        "train/test split, signal-loss scenarios, and training-fitted privacy-safe "
        "aggregate features.\n\n"
        "## Paired aggregate-recovery deltas\n\n"
        + pd.DataFrame(recovery_rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Scenario scores\n\n"
        + pd.DataFrame(score_rows).to_markdown(index=False, disable_numparse=True)
        + "\n\n"
        "## Interpretation limits\n\n"
        "The pairwise and listwise comparators are lightweight linear sensitivity "
        "checks inspired by learning-to-rank formulations. The listwise comparator "
        "uses a top-one softmax objective but is not an implementation of the "
        "original neural ListNet architecture. Neither comparator replaces the "
        "interpretable primary baseline.\n"
    )


def main(seeds: Iterable[int] = SEEDS) -> None:
    frames = []
    for seed in seeds:
        print(f"Running seed={seed}")
        frames.append(evaluate_seed(seed))

    raw = pd.concat(frames, ignore_index=True)
    summary = build_summary(raw)
    paired_recovery = build_paired_recovery_summary(raw)

    tables_dir = Path("outputs/tables")
    assets_dir = Path("docs/assets")
    docs_dir = Path("docs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    raw_path = tables_dir / "pairwise_ranking_sensitivity_raw.csv"
    summary_path = tables_dir / "pairwise_ranking_sensitivity_summary.csv"
    paired_path = tables_dir / "pairwise_ranking_sensitivity_paired_recovery.csv"
    figure_path = assets_dir / "pairwise_ranking_sensitivity.png"
    markdown_path = docs_dir / "pairwise_ranking_sensitivity.md"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired_recovery.to_csv(paired_path, index=False)
    plot_pairwise_ranking_sensitivity(summary, paired_recovery, figure_path)
    write_markdown_summary(summary, paired_recovery, markdown_path)

    print("\nRanking-objective recovery:")
    print(paired_recovery.round(4).to_string(index=False))
    print("\nWrote:")
    print(f"- {raw_path}")
    print(f"- {summary_path}")
    print(f"- {paired_path}")
    print(f"- {figure_path}")
    print(f"- {markdown_path}")


if __name__ == "__main__":
    main()
