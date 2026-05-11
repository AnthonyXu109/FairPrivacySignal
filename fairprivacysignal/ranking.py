from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairprivacysignal.signal_loss import SCENARIOS, apply_signal_loss


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "service_category",
    "age_group",
    "income_band",
    "urbanicity",
]


def average_ndcg_at_k(df: pd.DataFrame, k: int = 3) -> float:
    scores = []

    for _, group in df.groupby("household_id"):
        ranked = group.sort_values("predicted_relevance", ascending=False)
        gains = ranked["relevant"].to_numpy()[:k]

        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = np.sum(gains * discounts)

        ideal = np.sort(group["relevant"].to_numpy())[::-1][:k]
        ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        idcg = np.sum(ideal * ideal_discounts)

        if idcg > 0:
            scores.append(dcg / idcg)

    return float(np.mean(scores)) if scores else float("nan")


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def evaluate_scenario(events: pd.DataFrame, scenario: str) -> Dict[str, float]:
    df = apply_signal_loss(events, scenario)

    household_ids = df["household_id"].drop_duplicates()
    train_households, test_households = train_test_split(
        household_ids,
        test_size=0.30,
        random_state=42,
    )

    train = df[df["household_id"].isin(train_households)].copy()
    test = df[df["household_id"].isin(test_households)].copy()

    features: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    target = "relevant"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
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
        "scenario": scenario,
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

    rows = []
    for scenario in SCENARIOS:
        rows.append(evaluate_scenario(events, scenario))

    results = pd.DataFrame(rows)
    out_path = out_dir / "ranking_baseline_metrics.csv"
    results.to_csv(out_path, index=False)

    print("Ranking baseline metrics:")
    print(results.round(4).to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
