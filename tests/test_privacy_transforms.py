import pandas as pd

from fairprivacysignal.privacy_transforms import add_privacy_safe_features


def _events(
    households,
    engagements,
    age_group="adult",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": households,
            "service_category": ["food_assistance"] * len(households),
            "urbanicity": ["urban"] * len(households),
            "income_band": ["middle"] * len(households),
            "age_group": [age_group] * len(households),
            "historical_service_engagement_count": engagements,
            "underserved_score": [0.50] * len(households),
            "food_access_risk": [0.40] * len(households),
            "health_need_score": [0.30] * len(households),
            "housing_pressure": [0.20] * len(households),
        }
    )


def test_reference_events_control_cohort_aggregates() -> None:
    reference = _events(["H1", "H2"], [10.0, 20.0])
    holdout = _events(["H3"], [1000.0])

    transformed = add_privacy_safe_features(
        holdout,
        min_cohort_size=1,
        dp_noise_scale=0.0,
        reference_events=reference,
    )

    assert transformed["cohort_size"].tolist() == [2]
    assert transformed["privacy_safe_engagement_signal"].tolist() == [15.0]


def test_unseen_holdout_cohort_uses_training_service_fallback() -> None:
    reference = _events(["H1", "H2"], [10.0, 20.0], age_group="adult")
    holdout = _events(["H3"], [1000.0], age_group="senior")

    transformed = add_privacy_safe_features(
        holdout,
        min_cohort_size=1,
        dp_noise_scale=0.0,
        reference_events=reference,
    )

    assert transformed["cohort_size"].tolist() == [0]
    assert transformed["cohort_suppressed"].tolist() == [True]
    assert transformed["privacy_safe_engagement_signal"].tolist() == [15.0]
