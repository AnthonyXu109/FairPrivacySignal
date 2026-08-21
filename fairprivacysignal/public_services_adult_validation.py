from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from fairprivacysignal.public_data_visuals import (
    write_gallery_card_svg,
    write_recovery_profile_svg,
)


TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
DATA_DIR = Path("data/raw/public_services_adult")
TABLE_DIR = Path("outputs/tables")
ASSET_DIR = Path("docs/assets")
REPORT_PATH = Path("docs/public_services_adult_validation.md")
FIGURE_PATH = ASSET_DIR / "public_services_adult_validation.svg"
PROFILE_FIGURE_PATH = ASSET_DIR / "public_services_adult_recovery_profile.svg"
GALLERY_FIGURE_PATH = ASSET_DIR / "public_services_adult_gallery.svg"

K = 1000
COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
PERMITTED_FEATURES = [
    "age",
    "marital_status",
    "relationship",
    "race",
    "sex",
    "native_country",
]
ERROR_FLOOR = 1e-6
RELIABILITY_SHRINKAGE = 100.0
MIN_RECONSTRUCTION_WEIGHT = 0.10
MAX_RECONSTRUCTION_WEIGHT = 0.95
CROSS_FIT_SPLITS = 5
CROSS_FIT_SEED = 42
FIXED_RECONSTRUCTION_WEIGHT = 0.85
LOW_SIGNAL_NDCG_TOLERANCE = 0.0005
FIXED_RECOVERY_WEIGHTS = (0.85, 0.0, 0.15)
RANKING_WEIGHT_CANDIDATES = tuple(
    float(value) for value in np.round(np.arange(0.10, 1.00, 0.05), 2)
)
RECOVERY_WEIGHT_CANDIDATES = tuple(
    (ridge / 10.0, nonlinear / 10.0, cohort / 10.0)
    for ridge in range(11)
    for nonlinear in range(11 - ridge)
    for cohort in [10 - ridge - nonlinear]
) + (FIXED_RECOVERY_WEIGHTS,)


def ensure_adult_data(data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    if not train_path.exists():
        print(f"Downloading UCI Adult training data from {TRAIN_URL}")
        urlretrieve(TRAIN_URL, train_path)
    if not test_path.exists():
        print(f"Downloading UCI Adult test data from {TEST_URL}")
        urlretrieve(TEST_URL, test_path)
    return train_path, test_path


def load_adult_frames(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = ensure_adult_data(data_dir)
    train = pd.read_csv(
        train_path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        skipinitialspace=True,
    )
    test = pd.read_csv(
        test_path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        skipinitialspace=True,
        skiprows=1,
    )
    test["income"] = test["income"].str.replace(".", "", regex=False)
    return train.dropna().copy(), test.dropna().copy()


def minmax_from_train(train_series: pd.Series, apply_series: pd.Series) -> pd.Series:
    low = float(train_series.min())
    high = float(train_series.max())
    if high == low:
        return pd.Series(np.zeros(len(apply_series)), index=apply_series.index)
    return ((apply_series - low) / (high - low)).clip(0.0, 1.0)


def add_signal_features(
    train_reference: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    reference = train_reference.copy()
    reference["needs_support"] = (reference["income"] == "<=50K").astype(int)

    scored = frame.copy()
    scored["needs_support"] = (scored["income"] == "<=50K").astype(int)
    scored["education_hardship"] = 1.0 - minmax_from_train(
        reference["education_num"], scored["education_num"]
    )
    scored["low_hours"] = 1.0 - minmax_from_train(
        reference["hours_per_week"].clip(upper=80),
        scored["hours_per_week"].clip(upper=80),
    )
    reference_capital = np.log1p(
        reference["capital_gain"].clip(upper=100000)
        + reference["capital_loss"].clip(upper=10000)
    )
    scored_capital = np.log1p(
        scored["capital_gain"].clip(upper=100000)
        + scored["capital_loss"].clip(upper=10000)
    )
    scored["capital_hardship"] = 1.0 - minmax_from_train(
        reference_capital, scored_capital
    )

    global_need = float(reference["needs_support"].mean())
    occupation_need = reference.groupby("occupation")["needs_support"].mean()
    workclass_need = reference.groupby("workclass")["needs_support"].mean()
    scored["occupation_hardship"] = scored["occupation"].map(
        lambda value: occupation_need.get(value, global_need)
    )
    scored["workclass_hardship"] = scored["workclass"].map(
        lambda value: workclass_need.get(value, global_need)
    )
    scored["restricted_economic_signal"] = (
        0.30 * scored["education_hardship"]
        + 0.25 * scored["occupation_hardship"]
        + 0.20 * scored["workclass_hardship"]
        + 0.15 * scored["low_hours"]
        + 0.10 * scored["capital_hardship"]
    )

    age_position = minmax_from_train(reference["age"], scored["age"])
    scored["age_context"] = (1.0 - (age_position - 0.45).abs() * 2.0).clip(0.0, 1.0)
    scored["family_context"] = scored["relationship"].isin(
        ["Own-child", "Unmarried", "Not-in-family"]
    ).astype(float)
    scored["marital_context"] = (
        scored["marital_status"] != "Married-civ-spouse"
    ).astype(float)
    scored["context_score"] = (
        0.35 * scored["age_context"]
        + 0.25 * scored["family_context"]
        + 0.20 * scored["marital_context"]
        + 0.10 * (scored["sex"] == "Female").astype(float)
        + 0.10
        * scored["race"].isin(["Black", "Amer-Indian-Eskimo", "Other"]).astype(float)
    )
    scored["low_signal"] = scored["context_score"] <= scored["context_score"].median()
    return scored


def design_matrices(
    train: pd.DataFrame,
    apply_to: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    combined = pd.concat(
        [train[feature_columns], apply_to[feature_columns]],
        keys=["train", "apply"],
    )
    categorical = [
        column for column in feature_columns if combined[column].dtype == "object"
    ]
    encoded = pd.get_dummies(combined, columns=categorical, dtype=float).fillna(0.0)
    train_matrix = encoded.xs("train").to_numpy(dtype=float)
    apply_matrix = encoded.xs("apply").to_numpy(dtype=float)

    means = train_matrix.mean(axis=0)
    stds = train_matrix.std(axis=0)
    stds[stds == 0] = 1.0
    train_matrix = (train_matrix - means) / stds
    apply_matrix = (apply_matrix - means) / stds
    return (
        np.column_stack([np.ones(len(train_matrix)), train_matrix]),
        np.column_stack([np.ones(len(apply_matrix)), apply_matrix]),
    )


def reconstruct_economic_signal(
    train: pd.DataFrame,
    apply_to: pd.DataFrame,
    alpha: float = 100.0,
) -> np.ndarray:
    train_matrix, apply_matrix = design_matrices(train, apply_to, PERMITTED_FEATURES)
    target = train["restricted_economic_signal"].to_numpy(dtype=float)
    penalty = np.eye(train_matrix.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = (
        np.linalg.pinv(train_matrix.T @ train_matrix + penalty)
        @ train_matrix.T
        @ target
    )
    return np.clip(apply_matrix @ coefficients, 0.0, 1.0)


def reconstruct_nonlinear_economic_signal(
    train: pd.DataFrame,
    apply_to: pd.DataFrame,
    seed: int = CROSS_FIT_SEED,
) -> np.ndarray:
    train_matrix, apply_matrix = design_matrices(train, apply_to, PERMITTED_FEATURES)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=0.5,
        random_state=seed,
    )
    model.fit(
        train_matrix,
        train["restricted_economic_signal"].to_numpy(dtype=float),
    )
    return np.clip(model.predict(apply_matrix), 0.0, 1.0)


def cohort_economic_signal(train: pd.DataFrame, apply_to: pd.DataFrame) -> pd.Series:
    grouped = train.groupby(["marital_status", "relationship", "sex"])[
        "restricted_economic_signal"
    ].mean()
    fallback = train.groupby("relationship")["restricted_economic_signal"].mean()
    global_mean = float(train["restricted_economic_signal"].mean())
    return apply_to.apply(
        lambda row: grouped.get(
            (row["marital_status"], row["relationship"], row["sex"]),
            fallback.get(row["relationship"], global_mean),
        ),
        axis=1,
    )


def _reconstruction_weight(
    reconstruction_error: float,
    cohort_error: float,
    min_weight: float,
    max_weight: float,
) -> float:
    reconstruction_error = max(float(reconstruction_error), ERROR_FLOOR)
    cohort_error = max(float(cohort_error), ERROR_FLOOR)
    weight = cohort_error / (reconstruction_error + cohort_error)
    return float(np.clip(weight, min_weight, max_weight))


def estimate_reliability_weights(
    calibration: pd.DataFrame,
    shrinkage: float = RELIABILITY_SHRINKAGE,
    min_weight: float = MIN_RECONSTRUCTION_WEIGHT,
    max_weight: float = MAX_RECONSTRUCTION_WEIGHT,
) -> tuple[float, dict[object, float]]:
    errors = calibration.assign(
        reconstruction_error=(
            calibration["reconstructed_economic_signal"]
            - calibration["restricted_economic_signal"]
        ).abs(),
        cohort_error=(
            calibration["cohort_economic_signal"]
            - calibration["restricted_economic_signal"]
        ).abs(),
    )
    global_reconstruction_error = float(errors["reconstruction_error"].mean())
    global_cohort_error = float(errors["cohort_error"].mean())
    global_weight = _reconstruction_weight(
        global_reconstruction_error,
        global_cohort_error,
        min_weight,
        max_weight,
    )

    relationship_weights: dict[object, float] = {}
    for relationship, group in errors.groupby("relationship", dropna=False):
        support = float(len(group))
        reconstruction_error = (
            support * float(group["reconstruction_error"].mean())
            + shrinkage * global_reconstruction_error
        ) / (support + shrinkage)
        cohort_error = (
            support * float(group["cohort_error"].mean())
            + shrinkage * global_cohort_error
        ) / (support + shrinkage)
        relationship_weights[relationship] = _reconstruction_weight(
            reconstruction_error,
            cohort_error,
            min_weight,
            max_weight,
        )
    return global_weight, relationship_weights


def reliability_weighted_signal(
    frame: pd.DataFrame,
    global_weight: float,
    relationship_weights: dict[object, float],
    low_signal_weight: float | None = None,
) -> pd.Series:
    weights = frame["relationship"].map(relationship_weights).fillna(global_weight)
    if low_signal_weight is not None:
        weights = weights.where(~frame["low_signal"], float(low_signal_weight))
    return (
        weights * frame["reconstructed_economic_signal"]
        + (1.0 - weights) * frame["cohort_economic_signal"]
    )


def select_ranking_calibrated_weight(
    calibration: pd.DataFrame,
    candidate_weights: list[float] | tuple[float, ...] = RANKING_WEIGHT_CANDIDATES,
    baseline_weight: float = FIXED_RECONSTRUCTION_WEIGHT,
    low_signal_tolerance: float = LOW_SIGNAL_NDCG_TOLERANCE,
) -> float:
    def ranking_metrics(weight: float) -> tuple[float, float]:
        weights = np.where(
            calibration["low_signal"],
            baseline_weight,
            weight,
        )
        recovered_signal = (
            weights * calibration["reconstructed_economic_signal"]
            + (1.0 - weights) * calibration["cohort_economic_signal"]
        )
        scored = calibration.assign(
            _candidate_score=calibration["context_score"] + 2.0 * recovered_signal
        )
        low_signal = scored[scored["low_signal"]]
        return (
            ndcg_at_k(scored, "_candidate_score"),
            (
                ndcg_at_k(low_signal, "_candidate_score")
                if not low_signal.empty
                else 1.0
            ),
        )

    _, baseline_low_signal = ranking_metrics(baseline_weight)
    eligible: list[tuple[float, float, float, float]] = []
    for candidate in candidate_weights:
        weight = float(candidate)
        overall_ndcg, low_signal_ndcg = ranking_metrics(weight)
        if low_signal_ndcg >= baseline_low_signal - low_signal_tolerance:
            eligible.append(
                (
                    overall_ndcg,
                    low_signal_ndcg,
                    -abs(weight - baseline_weight),
                    weight,
                )
            )

    if not eligible:
        return float(baseline_weight)
    return max(eligible)[-1]


def select_recovery_weights(
    calibration: pd.DataFrame,
    candidate_weights: list[tuple[float, float, float]],
    baseline_weights: tuple[float, float, float],
    low_signal_tolerance: float = LOW_SIGNAL_NDCG_TOLERANCE,
) -> tuple[float, float, float]:
    signal_columns = [
        "reconstructed_economic_signal",
        "nonlinear_economic_signal",
        "cohort_economic_signal",
    ]

    def recovered_signal(weights: tuple[float, float, float]) -> pd.Series:
        return sum(
            float(weight) * calibration[column]
            for weight, column in zip(weights, signal_columns)
        )

    def low_signal_ndcg(weights: tuple[float, float, float]) -> float:
        scored = calibration.assign(
            _candidate_score=(
                calibration["context_score"] + 2.0 * recovered_signal(weights)
            )
        )
        low_signal = scored[scored["low_signal"]]
        return (
            ndcg_at_k(low_signal, "_candidate_score")
            if not low_signal.empty
            else 1.0
        )

    baseline_low_signal = low_signal_ndcg(baseline_weights)
    eligible: list[
        tuple[float, float, float, tuple[float, float, float]]
    ] = []
    for candidate in candidate_weights:
        weights = tuple(float(value) for value in candidate)
        candidate_signal = recovered_signal(weights)
        reconstruction_mae = float(
            (candidate_signal - calibration["restricted_economic_signal"])
            .abs()
            .mean()
        )
        candidate_low_signal_ndcg = low_signal_ndcg(weights)
        if candidate_low_signal_ndcg >= baseline_low_signal - low_signal_tolerance:
            distance = sum(
                abs(value - baseline)
                for value, baseline in zip(weights, baseline_weights)
            )
            eligible.append(
                (
                    -reconstruction_mae,
                    candidate_low_signal_ndcg,
                    -distance,
                    weights,
                )
            )

    if not eligible:
        return baseline_weights
    return max(eligible)[-1]


def cross_fitted_recovery_calibration(
    train_raw: pd.DataFrame,
    n_splits: int = CROSS_FIT_SPLITS,
    seed: int = CROSS_FIT_SEED,
) -> pd.DataFrame:
    if n_splits < 2 or n_splits > len(train_raw):
        raise ValueError(
            "n_splits must be between 2 and the number of training rows"
        )

    rng = np.random.default_rng(seed)
    shuffled_positions = rng.permutation(len(train_raw))
    fold_ids = np.empty(len(train_raw), dtype=int)
    fold_ids[shuffled_positions] = np.arange(len(train_raw)) % n_splits
    parts = []

    for fold_id in range(n_splits):
        fit_raw = train_raw.iloc[fold_ids != fold_id]
        holdout_raw = train_raw.iloc[fold_ids == fold_id]
        fit = add_signal_features(fit_raw, fit_raw)
        holdout = add_signal_features(fit_raw, holdout_raw)
        holdout["reconstructed_economic_signal"] = reconstruct_economic_signal(
            fit,
            holdout,
        )
        holdout["nonlinear_economic_signal"] = (
            reconstruct_nonlinear_economic_signal(
                fit,
                holdout,
                seed=seed + fold_id,
            )
        )
        holdout["cohort_economic_signal"] = cohort_economic_signal(fit, holdout)
        holdout["_original_position"] = np.flatnonzero(fold_ids == fold_id)
        parts.append(
            holdout[
                [
                    "_original_position",
                    "relationship",
                    "restricted_economic_signal",
                    "reconstructed_economic_signal",
                    "nonlinear_economic_signal",
                    "cohort_economic_signal",
                    "context_score",
                    "needs_support",
                    "low_signal",
                ]
            ]
        )

    return (
        pd.concat(parts)
        .sort_values("_original_position")
        .drop(columns="_original_position")
        .reset_index(drop=True)
    )


def learn_reliability_weights(
    train_raw: pd.DataFrame,
    calibration: pd.DataFrame | None = None,
) -> tuple[float, dict[object, float]]:
    if calibration is None:
        calibration = cross_fitted_recovery_calibration(train_raw)
    error_weight, error_relationship_weights = estimate_reliability_weights(
        calibration
    )
    ranking_weight = select_ranking_calibrated_weight(calibration)
    relationship_weights = {
        relationship: float(
            np.clip(
                ranking_weight + relationship_weight - error_weight,
                MIN_RECONSTRUCTION_WEIGHT,
                MAX_RECONSTRUCTION_WEIGHT,
            )
        )
        for relationship, relationship_weight in error_relationship_weights.items()
    }
    return ranking_weight, relationship_weights


def score_people(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    train = add_signal_features(train_raw, train_raw)
    scored = add_signal_features(train_raw, test_raw)

    scored["reconstructed_economic_signal"] = reconstruct_economic_signal(train, scored)
    scored["nonlinear_economic_signal"] = reconstruct_nonlinear_economic_signal(
        train,
        scored,
    )
    scored["cohort_economic_signal"] = cohort_economic_signal(train, scored)
    fixed_recovered_signal = (
        FIXED_RECONSTRUCTION_WEIGHT * scored["reconstructed_economic_signal"]
        + (1.0 - FIXED_RECONSTRUCTION_WEIGHT)
        * scored["cohort_economic_signal"]
    )
    calibration = cross_fitted_recovery_calibration(train_raw)
    global_weight, relationship_weights = learn_reliability_weights(
        train_raw,
        calibration=calibration,
    )
    reliability_recovered_signal = reliability_weighted_signal(
        scored,
        global_weight,
        relationship_weights,
        low_signal_weight=FIXED_RECONSTRUCTION_WEIGHT,
    )
    scored["reconstruction_weight"] = (
        scored["relationship"].map(relationship_weights).fillna(global_weight)
    )
    scored["reconstruction_weight"] = scored["reconstruction_weight"].where(
        ~scored["low_signal"],
        FIXED_RECONSTRUCTION_WEIGHT,
    )
    selected_weights = select_recovery_weights(
        calibration,
        candidate_weights=list(RECOVERY_WEIGHT_CANDIDATES),
        baseline_weights=FIXED_RECOVERY_WEIGHTS,
    )
    selected_recovered_signal = (
        selected_weights[0] * scored["reconstructed_economic_signal"]
        + selected_weights[1] * scored["nonlinear_economic_signal"]
        + selected_weights[2] * scored["cohort_economic_signal"]
    )
    scored["selected_ridge_weight"] = selected_weights[0]
    scored["selected_nonlinear_weight"] = selected_weights[1]
    scored["selected_cohort_weight"] = selected_weights[2]

    scored["full_signal_score"] = (
        scored["context_score"] + 2.0 * scored["restricted_economic_signal"]
    )
    scored["context_only_score"] = scored["context_score"]
    scored["fixed_signal_recovery_score"] = (
        scored["context_score"] + 2.0 * fixed_recovered_signal
    )
    scored["signal_recovery_score"] = (
        scored["context_score"] + 2.0 * reliability_recovered_signal
    )
    scored["selected_signal_recovery_score"] = (
        scored["context_score"] + 2.0 * selected_recovered_signal
    )
    scored["policy_aware_score"] = np.where(
        scored["low_signal"],
        scored["selected_signal_recovery_score"],
        scored["full_signal_score"],
    )
    return scored


def ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = K) -> float:
    ranked = df.sort_values(score_col, ascending=False)
    gains = ranked["needs_support"].to_numpy(dtype=float)[: min(k, len(ranked))]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(df["needs_support"].to_numpy(dtype=float))[::-1][: len(gains)]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else float("nan")


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    truth = y_true.to_numpy(dtype=int)
    scores = y_score.to_numpy(dtype=float)
    positives = truth == 1
    negatives = truth == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[positives].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def summarize_method(
    scored: pd.DataFrame,
    method: str,
    score_col: str,
    economic_signal_exposure: float,
) -> dict[str, float | str]:
    low_signal = scored[scored["low_signal"]]
    not_low_signal = scored[~scored["low_signal"]]
    return {
        "sector": "public_services",
        "dataset": "UCI Adult",
        "method": method,
        "overall_auc": safe_auc(scored["needs_support"], scored[score_col]),
        "overall_ndcg_at_1000": ndcg_at_k(scored, score_col),
        "low_signal_ndcg_at_1000": ndcg_at_k(low_signal, score_col),
        "not_low_signal_ndcg_at_1000": ndcg_at_k(not_low_signal, score_col),
        "economic_signal_exposure": economic_signal_exposure,
        "num_people": int(len(scored)),
        "low_signal_share": float(scored["low_signal"].mean()),
    }


def summarize_results(scored: pd.DataFrame) -> pd.DataFrame:
    policy_exposure = float((~scored["low_signal"]).mean())
    rows = [
        summarize_method(scored, "Full detailed-economic signal", "full_signal_score", 1.0),
        summarize_method(scored, "Context-only baseline", "context_only_score", 0.0),
        summarize_method(
            scored,
            "Train-fitted nonlinear recovery",
            "selected_signal_recovery_score",
            0.0,
        ),
        summarize_method(scored, "Policy-aware partial recovery", "policy_aware_score", policy_exposure),
    ]
    summary = pd.DataFrame(rows)
    full = float(
        summary.loc[
            summary["method"] == "Full detailed-economic signal",
            "overall_ndcg_at_1000",
        ].iloc[0]
    )
    loss = float(
        summary.loc[
            summary["method"] == "Context-only baseline",
            "overall_ndcg_at_1000",
        ].iloc[0]
    )
    denominator = full - loss
    summary["full_signal_gap_closed"] = np.nan
    if denominator > 0:
        summary["full_signal_gap_closed"] = (
            summary["overall_ndcg_at_1000"] - loss
        ) / denominator
    summary.loc[
        summary["method"].eq("Full detailed-economic signal"),
        "full_signal_gap_closed",
    ] = 1.0
    summary.loc[
        summary["method"].eq("Context-only baseline"),
        "full_signal_gap_closed",
    ] = 0.0
    return summary


def summarize_recovery_comparison(scored: pd.DataFrame) -> pd.DataFrame:
    comparison = pd.DataFrame(
        [
            summarize_method(
                scored,
                "Fixed 85/15 recovery",
                "fixed_signal_recovery_score",
                0.0,
            ),
            summarize_method(
                scored,
                "Reliability-weighted recovery",
                "signal_recovery_score",
                0.0,
            ),
            summarize_method(
                scored,
                "OOF-selected nonlinear recovery",
                "selected_signal_recovery_score",
                0.0,
            ),
        ]
    )
    full = ndcg_at_k(scored, "full_signal_score")
    loss = ndcg_at_k(scored, "context_only_score")
    denominator = full - loss
    comparison["full_signal_gap_closed"] = np.nan
    if denominator > 0:
        comparison["full_signal_gap_closed"] = (
            comparison["overall_ndcg_at_1000"] - loss
        ) / denominator
    comparison["ridge_weight"] = [
        FIXED_RECOVERY_WEIGHTS[0],
        np.nan,
        float(scored["selected_ridge_weight"].iloc[0]),
    ]
    comparison["nonlinear_weight"] = [
        FIXED_RECOVERY_WEIGHTS[1],
        np.nan,
        float(scored["selected_nonlinear_weight"].iloc[0]),
    ]
    comparison["cohort_weight"] = [
        FIXED_RECOVERY_WEIGHTS[2],
        np.nan,
        float(scored["selected_cohort_weight"].iloc[0]),
    ]
    return comparison


def validate_recovery_comparison(comparison: pd.DataFrame) -> None:
    reliability = comparison[
        comparison["method"] == "Reliability-weighted recovery"
    ].iloc[0]
    selected = comparison[
        comparison["method"] == "OOF-selected nonlinear recovery"
    ].iloc[0]

    if float(selected["overall_ndcg_at_1000"]) <= float(
        reliability["overall_ndcg_at_1000"]
    ):
        raise ValueError("Selected recovery must improve overall NDCG@1000")
    if float(selected["low_signal_ndcg_at_1000"]) < float(
        reliability["low_signal_ndcg_at_1000"]
    ):
        raise ValueError("Selected recovery must preserve low-signal NDCG@1000")
    if not np.isclose(float(selected["economic_signal_exposure"]), 0.0):
        raise ValueError("Selected recovery must keep restricted-signal exposure at zero")


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{100.0 * value:.1f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def write_svg(summary: pd.DataFrame, path: Path = FIGURE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = summary["method"].tolist()
    overall = summary["overall_ndcg_at_1000"].tolist()
    low_signal = summary["low_signal_ndcg_at_1000"].tolist()
    max_value = 1.0

    width = 940
    height = 550
    left = 92
    top = 92
    plot_width = 760
    plot_height = 288
    group_width = plot_width / len(methods)
    bar_width = 34

    def y_pos(value: float) -> float:
        return top + plot_height - (value / max_value) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Public-services public-data validation</text>',
        '<text x="40" y="60" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Ranking low-income support outreach when detailed economic signals are suppressed; higher NDCG@1000 is better.</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]

    for tick in np.linspace(0.0, max_value, 5):
        y = y_pos(float(tick))
        lines.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#4b5563">{tick:.2f}</text>')

    colors = {"overall": "#4338ca", "low": "#be123c"}
    for idx, method in enumerate(methods):
        center = left + group_width * idx + group_width / 2
        for offset, value, key in [
            (-bar_width / 2 - 4, overall[idx], "overall"),
            (bar_width / 2 + 4, low_signal[idx], "low"),
        ]:
            x = center + offset - bar_width / 2
            y = y_pos(float(value))
            h = top + plot_height - y
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" rx="3" fill="{colors[key]}"/>')
            lines.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#111827">{value:.3f}</text>')
        for line_idx, label_line in enumerate(method.replace(" ", "\n").split("\n")):
            lines.append(f'<text x="{center:.1f}" y="{top + plot_height + 22 + 14 * line_idx}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{label_line}</text>')

    legend_y = top + plot_height + 102
    lines.extend(
        [
            f'<rect x="{left}" y="{legend_y}" width="14" height="14" fill="{colors["overall"]}" rx="2"/>',
            f'<text x="{left + 22}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Overall adults</text>',
            f'<rect x="{left + 210}" y="{legend_y}" width="14" height="14" fill="{colors["low"]}" rx="2"/>',
            f'<text x="{left + 232}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Low-signal adults</text>',
            f'<text x="{left}" y="{legend_y + 42}" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">Raw UCI data is downloaded at runtime and is not redistributed in this repository.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    path: Path = REPORT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = summary.copy()
    for column in [
        "overall_auc",
        "overall_ndcg_at_1000",
        "low_signal_ndcg_at_1000",
        "not_low_signal_ndcg_at_1000",
        "economic_signal_exposure",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    display["full_signal_gap_closed"] = summary["full_signal_gap_closed"].map(format_percent)

    table = markdown_table(
        display[
            [
                "method",
                "overall_ndcg_at_1000",
                "low_signal_ndcg_at_1000",
                "full_signal_gap_closed",
                "economic_signal_exposure",
            ]
        ]
    )

    comparison_display = comparison.copy()
    for column in [
        "overall_ndcg_at_1000",
        "low_signal_ndcg_at_1000",
        "economic_signal_exposure",
    ]:
        comparison_display[column] = comparison_display[column].map(
            lambda value: f"{value:.3f}"
        )
    comparison_display["full_signal_gap_closed"] = comparison[
        "full_signal_gap_closed"
    ].map(format_percent)
    comparison_table = markdown_table(
        comparison_display[
            [
                "method",
                "overall_ndcg_at_1000",
                "low_signal_ndcg_at_1000",
                "full_signal_gap_closed",
                "economic_signal_exposure",
            ]
        ]
    )

    recovery = summary[
        summary["method"] == "Train-fitted nonlinear recovery"
    ].iloc[0]
    policy = summary[summary["method"] == "Policy-aware partial recovery"].iloc[0]
    weighted = comparison[
        comparison["method"] == "Reliability-weighted recovery"
    ].iloc[0]
    selected = comparison[
        comparison["method"] == "OOF-selected nonlinear recovery"
    ].iloc[0]
    overall_change = (
        float(selected["overall_ndcg_at_1000"])
        - float(weighted["overall_ndcg_at_1000"])
    )
    low_signal_change = (
        float(selected["low_signal_ndcg_at_1000"])
        - float(weighted["low_signal_ndcg_at_1000"])
    )
    selected_weights = (
        float(selected["ridge_weight"]),
        float(selected["nonlinear_weight"]),
        float(selected["cohort_weight"]),
    )

    content = f"""# Public Services Adult Census Validation

This public-data pilot adapts the FairPrivacySignal signal-loss pattern to a
public-service outreach setting using the UCI Adult/Census Income dataset. Adults
are ranked for low-income support outreach; detailed employment and economic
fields are treated as signals that may be unavailable under data minimization,
while coarse demographic and household context remains available.

The raw UCI files are downloaded at runtime and are not redistributed in this repository.

![Public-services public-data validation](assets/public_services_adult_validation.svg)

![Public-services recovery profile](assets/public_services_adult_recovery_profile.svg)

## Task

- **Ranked candidate:** adults for public or nonprofit support outreach
- **Restricted economic signal:** education, occupation, workclass, hours, and capital-gain/loss fields
- **Permitted context:** age, marital status, relationship, race, sex, and native country
- **Low-signal group:** adults below the median permitted-context score
- **Metric:** NDCG@1000, with binary relevance defined as income `<=50K`

## Results

{table}

The train-fitted nonlinear recovery path closes {format_percent(float(recovery["full_signal_gap_closed"]))}
of the full-signal NDCG@1000 gap without exposing the restricted detailed
economic features at scoring time. The policy-aware partial path keeps detailed
economic signal for higher-signal records while substituting recovered signal for
low-signal records, closing {format_percent(float(policy["full_signal_gap_closed"]))}
of the same gap in this pilot.

## Recovery method comparison

{comparison_table}

Five-fold out-of-fold predictions on the UCI training split compare a linear
ridge reconstruction, a histogram gradient-boosted reconstruction, and a
cohort aggregate. Convex weights are selected by out-of-fold reconstruction
error among candidates that preserve low-signal NDCG. The selected ridge,
nonlinear, and cohort weights are `{selected_weights[0]:.2f}`,
`{selected_weights[1]:.2f}`, and `{selected_weights[2]:.2f}`. The official test
split is not used to fit either base estimator or select the weights. Relative
to the reliability-weighted recovery method, the selected nonlinear recovery
changes held-out overall NDCG@1000 by
`{overall_change:+.6f}` and low-signal NDCG@1000 by
`{low_signal_change:+.6f}` while keeping restricted-signal exposure at `0.000`.

## Interpretation

This is an external public-data validation of the system shape, not a deployed
benefits or nonprofit-service model. It shows how the method can be instantiated
in a census-like public-services workflow: define detailed economic signals,
suppress them at scoring time, substitute a train-fitted reconstruction with a
richer nonlinear candidate, select it on training-only folds with a low-signal
guardrail, and measure outreach-ranking recovery. The dataset is a public income benchmark
rather than a service-interaction log, so the availability policy is simulated
for evaluation.
"""
    path.write_text(content, encoding="utf-8")


def run_validation() -> pd.DataFrame:
    train, test = load_adult_frames()
    scored = score_people(train, test)
    summary = summarize_results(scored)
    comparison = summarize_recovery_comparison(scored)
    validate_recovery_comparison(comparison)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(TABLE_DIR / "public_services_adult_scored_people.csv", index=False)
    summary.to_csv(TABLE_DIR / "public_services_adult_validation_summary.csv", index=False)
    comparison.to_csv(
        TABLE_DIR / "public_services_adult_recovery_comparison.csv",
        index=False,
    )
    write_svg(summary)
    write_recovery_profile_svg(
        summary,
        PROFILE_FIGURE_PATH,
        title="Public-Services Recovery Profile",
        subtitle="Low-income outreach ranking under detailed economic signal loss.",
        metric_col="overall_ndcg_at_1000",
        low_signal_col="low_signal_ndcg_at_1000",
        exposure_col="economic_signal_exposure",
        low_signal_label="Low-signal NDCG@1000",
    )
    write_gallery_card_svg(
        summary,
        GALLERY_FIGURE_PATH,
        title="Public Services",
        subtitle="Low-income outreach under detailed economic signal loss.",
        metric_col="overall_ndcg_at_1000",
        exposure_col="economic_signal_exposure",
    )
    write_report(summary, comparison)
    return summary


def main() -> None:
    summary = run_validation()
    print("Public services Adult Census validation:")
    print(summary.round(4).to_string(index=False))
    print(f"\nWrote: {REPORT_PATH}")
    print(f"Wrote: {FIGURE_PATH}")
    print(f"Wrote: {PROFILE_FIGURE_PATH}")
    print(f"Wrote: {GALLERY_FIGURE_PATH}")


if __name__ == "__main__":
    main()
