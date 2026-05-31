from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairprivacysignal.privacy_transforms import add_privacy_safe_features
from fairprivacysignal.ranking import average_ndcg_at_k, safe_auc
from fairprivacysignal.signal_loss import apply_signal_loss


BASE_NUMERIC_FEATURES = [
    "available_historical_service_engagement_count",
    "employment_need",
    "median_income",
    "unemployment_rate",
    "broadband_access",
    "food_access_risk",
    "health_need_score",
    "housing_pressure",
    "underserved_score",
]

PRIVACY_SAFE_NUMERIC_FEATURES = [
    "available_historical_service_engagement_count",
    "privacy_safe_engagement_signal",
    "privacy_safe_cohort_avg_underserved",
    "privacy_safe_cohort_avg_food_risk",
    "privacy_safe_cohort_avg_health_need",
    "privacy_safe_cohort_avg_housing_pressure",
    "employment_need",
    "median_income",
    "unemployment_rate",
    "broadband_access",
    "food_access_risk",
    "health_need_score",
    "housing_pressure",
    "underserved_score",
]

CATEGORICAL_FEATURES = [
    "service_category",
    "age_group",
    "income_band",
    "urbanicity",
]


def correct_positive_reweighting(
    predictions: np.ndarray,
    positive_weight_multiplier: float,
) -> np.ndarray:
    """Undo the probability-scale shift introduced by extra positive weighting."""
    clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    odds = clipped / (1.0 - clipped)
    corrected_odds = odds / positive_weight_multiplier
    return corrected_odds / (1.0 + corrected_odds)


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
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def evaluate_model(
    df: pd.DataFrame,
    experiment: str,
    numeric_features: List[str],
    fairness_aware: bool = False,
    low_signal_blend_weight: float = 0.75,
    relevant_low_signal_weight: float = 4.0,
) -> Dict[str, float]:
    household_ids = df["household_id"].drop_duplicates()

    train_households, test_households = train_test_split(
        household_ids,
        test_size=0.30,
        random_state=42,
    )

    train = df[df["household_id"].isin(train_households)].copy()
    test = df[df["household_id"].isin(test_households)].copy()

    features = numeric_features + CATEGORICAL_FEATURES
    target = "relevant"

    global_model = build_model(numeric_features)
    global_model.fit(train[features], train[target])

    test["predicted_relevance"] = global_model.predict_proba(test[features])[:, 1]

    if fairness_aware:
        low_train = train[train["low_signal"].astype(bool)].copy()
        low_test_mask = test["low_signal"].astype(bool)

        if len(low_train) > 100 and low_train[target].nunique() > 1:
            low_signal_model = clone(global_model)

            low_sample_weight = np.ones(len(low_train), dtype=float)
            low_sample_weight += relevant_low_signal_weight * low_train[target].astype(bool).to_numpy()

            low_signal_model.fit(
                low_train[features],
                low_train[target],
                classifier__sample_weight=low_sample_weight,
            )

            low_specific_pred = low_signal_model.predict_proba(
                test.loc[low_test_mask, features]
            )[:, 1]
            low_specific_pred = correct_positive_reweighting(
                low_specific_pred,
                positive_weight_multiplier=1.0 + relevant_low_signal_weight,
            )

            global_pred = test.loc[low_test_mask, "predicted_relevance"].to_numpy()

            test.loc[low_test_mask, "predicted_relevance"] = (
                (1.0 - low_signal_blend_weight) * global_pred
                + low_signal_blend_weight * low_specific_pred
            ).clip(0.001, 0.999)

    low = test[test["low_signal"] == True]
    not_low = test[test["low_signal"] == False]

    overall_auc = safe_auc(test[target], test["predicted_relevance"])
    low_auc = safe_auc(low[target], low["predicted_relevance"])
    not_low_auc = safe_auc(not_low[target], not_low["predicted_relevance"])

    overall_ndcg = average_ndcg_at_k(test, k=3)
    low_ndcg = average_ndcg_at_k(low, k=3)
    not_low_ndcg = average_ndcg_at_k(not_low, k=3)

    return {
        "experiment": experiment,
        "fairness_aware": fairness_aware,
        "overall_auc": overall_auc,
        "low_signal_auc": low_auc,
        "not_low_signal_auc": not_low_auc,
        "auc_gap_not_low_minus_low": not_low_auc - low_auc,
        "overall_ndcg_at_3": overall_ndcg,
        "low_signal_ndcg_at_3": low_ndcg,
        "not_low_signal_ndcg_at_3": not_low_ndcg,
        "ndcg_gap_not_low_minus_low": not_low_ndcg - low_ndcg,
        "avg_privacy_exposure_score": test["privacy_exposure_score"].mean(),
        "behavioral_available_share": test["behavioral_available"].mean(),
    }


def run_experiments(
    events: pd.DataFrame,
    privacy_noise_seed: int = 42,
) -> pd.DataFrame:
    experiments = []

    full_signal = apply_signal_loss(events, "full_signal")
    experiments.append(
        evaluate_model(
            full_signal,
            "full_signal_raw_baseline",
            BASE_NUMERIC_FEATURES,
        )
    )

    severe_loss = apply_signal_loss(events, "severe_signal_loss")
    experiments.append(
        evaluate_model(
            severe_loss,
            "severe_signal_loss_baseline",
            BASE_NUMERIC_FEATURES,
        )
    )

    severe_loss_privacy_safe = add_privacy_safe_features(
        severe_loss,
        seed=privacy_noise_seed,
    )
    experiments.append(
        evaluate_model(
            severe_loss_privacy_safe,
            "severe_signal_loss_with_privacy_safe_aggregates",
            PRIVACY_SAFE_NUMERIC_FEATURES,
        )
    )

    experiments.append(
        evaluate_model(
            severe_loss_privacy_safe,
            "severe_signal_loss_with_privacy_safe_fairness_aware",
            PRIVACY_SAFE_NUMERIC_FEATURES,
            fairness_aware=True,
        )
    )

    policy_restricted = apply_signal_loss(events, "policy_restricted")
    experiments.append(
        evaluate_model(
            policy_restricted,
            "policy_restricted_baseline",
            BASE_NUMERIC_FEATURES,
        )
    )

    policy_restricted_privacy_safe = add_privacy_safe_features(
        policy_restricted,
        seed=privacy_noise_seed,
    )
    experiments.append(
        evaluate_model(
            policy_restricted_privacy_safe,
            "policy_restricted_with_privacy_safe_aggregates",
            PRIVACY_SAFE_NUMERIC_FEATURES,
        )
    )

    experiments.append(
        evaluate_model(
            policy_restricted_privacy_safe,
            "policy_restricted_with_privacy_safe_fairness_aware",
            PRIVACY_SAFE_NUMERIC_FEATURES,
            fairness_aware=True,
        )
    )

    return pd.DataFrame(experiments)


def main() -> None:
    data_dir = Path("data/synthetic")
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_dir / "synthetic_outreach_events.csv")

    results = run_experiments(events)
    out_path = out_dir / "privacy_recovery_metrics.csv"
    results.to_csv(out_path, index=False)

    print("Privacy recovery metrics:")
    print(results.round(4).to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
