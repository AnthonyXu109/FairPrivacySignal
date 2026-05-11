from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairprivacysignal.privacy_transforms import add_privacy_safe_features
from fairprivacysignal.ranking import average_ndcg_at_k, safe_auc
from fairprivacysignal.signal_loss import apply_signal_loss


BASE_NUMERIC_FEATURES = [
    "available_historical_engagement_count",
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


def evaluate_model(
    df: pd.DataFrame,
    experiment: str,
    numeric_features: List[str],
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

    model.fit(train[features], train[target])
    test["predicted_relevance"] = model.predict_proba(test[features])[:, 1]

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


def main() -> None:
    data_dir = Path("data/synthetic")
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_dir / "synthetic_outreach_events.csv")

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

    severe_loss_privacy_safe = add_privacy_safe_features(severe_loss)
    experiments.append(
        evaluate_model(
            severe_loss_privacy_safe,
            "severe_signal_loss_with_privacy_safe_aggregates",
            PRIVACY_SAFE_NUMERIC_FEATURES,
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

    policy_restricted_privacy_safe = add_privacy_safe_features(policy_restricted)
    experiments.append(
        evaluate_model(
            policy_restricted_privacy_safe,
            "policy_restricted_with_privacy_safe_aggregates",
            PRIVACY_SAFE_NUMERIC_FEATURES,
        )
    )

    results = pd.DataFrame(experiments)
    out_path = out_dir / "privacy_recovery_metrics.csv"
    results.to_csv(out_path, index=False)

    print("Privacy recovery metrics:")
    print(results.round(4).to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
