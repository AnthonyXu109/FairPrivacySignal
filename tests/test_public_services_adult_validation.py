import numpy as np
import pandas as pd
import pytest

from fairprivacysignal.public_services_adult_validation import (
    cross_fitted_recovery_calibration,
    estimate_reliability_weights,
    reliability_weighted_signal,
    score_people,
    select_ranking_calibrated_weight,
    summarize_recovery_comparison,
    summarize_results,
)


def adult_fixture(size: int = 36) -> pd.DataFrame:
    index = np.arange(size)
    return pd.DataFrame(
        {
            "age": 20 + index % 45,
            "workclass": np.where(index % 3 == 0, "Private", "Local-gov"),
            "fnlwgt": 100000 + index * 100,
            "education": np.where(index % 2 == 0, "HS-grad", "Bachelors"),
            "education_num": 9 + index % 8,
            "marital_status": np.where(
                index % 2 == 0,
                "Never-married",
                "Married-civ-spouse",
            ),
            "occupation": np.where(index % 3 == 0, "Sales", "Tech-support"),
            "relationship": np.where(
                index % 2 == 0,
                "Not-in-family",
                "Husband",
            ),
            "race": np.where(index % 4 == 0, "Black", "White"),
            "sex": np.where(index % 2 == 0, "Female", "Male"),
            "capital_gain": np.where(index % 5 == 0, 5000, 0),
            "capital_loss": np.where(index % 7 == 0, 1000, 0),
            "hours_per_week": 20 + index % 40,
            "native_country": "United-States",
            "income": np.where(index % 3 == 0, ">50K", "<=50K"),
        }
    )


def test_estimate_reliability_weights_prefers_the_more_accurate_estimator() -> None:
    calibration = pd.DataFrame(
        {
            "relationship": ["A"] * 4 + ["B"] * 4,
            "restricted_economic_signal": [0.0, 0.2, 0.8, 1.0] * 2,
            "reconstructed_economic_signal": [
                0.05,
                0.25,
                0.75,
                0.95,
                0.50,
                0.50,
                0.50,
                0.50,
            ],
            "cohort_economic_signal": [
                0.50,
                0.50,
                0.50,
                0.50,
                0.05,
                0.25,
                0.75,
                0.95,
            ],
        }
    )

    global_weight, relationship_weights = estimate_reliability_weights(
        calibration,
        shrinkage=1.0,
    )

    assert 0.10 <= global_weight <= 0.95
    assert relationship_weights["A"] > global_weight
    assert relationship_weights["B"] < global_weight


def test_reliability_weighted_signal_uses_global_weight_for_unseen_group() -> None:
    frame = pd.DataFrame(
        {
            "relationship": ["known", "unseen"],
            "low_signal": [False, False],
            "reconstructed_economic_signal": [0.8, 0.8],
            "cohort_economic_signal": [0.2, 0.2],
        }
    )

    result = reliability_weighted_signal(
        frame,
        global_weight=0.25,
        relationship_weights={"known": 0.75},
    )

    assert result.tolist() == pytest.approx([0.65, 0.35])


def test_reliability_weighted_signal_keeps_low_signal_weight_fixed() -> None:
    frame = pd.DataFrame(
        {
            "relationship": ["known", "known"],
            "low_signal": [True, False],
            "reconstructed_economic_signal": [0.8, 0.8],
            "cohort_economic_signal": [0.2, 0.2],
        }
    )

    result = reliability_weighted_signal(
        frame,
        global_weight=0.25,
        relationship_weights={"known": 0.75},
        low_signal_weight=0.85,
    )

    assert result.tolist() == pytest.approx([0.71, 0.65])


def test_ranking_calibration_selects_the_weight_with_better_ordering() -> None:
    calibration = pd.DataFrame(
        {
            "needs_support": [0, 1, 0, 1],
            "low_signal": [False, False, False, False],
            "context_score": [0.0, 0.0, 0.0, 0.0],
            "reconstructed_economic_signal": [0.1, 0.9, 0.2, 0.8],
            "cohort_economic_signal": [0.9, 0.1, 0.8, 0.2],
        }
    )

    selected = select_ranking_calibrated_weight(
        calibration,
        candidate_weights=[0.1, 0.5, 0.9],
        baseline_weight=0.5,
        low_signal_tolerance=0.0,
    )

    assert selected == 0.9


def test_cross_fitted_recovery_calibration_is_deterministic() -> None:
    train = adult_fixture()

    first = cross_fitted_recovery_calibration(train, n_splits=3, seed=42)
    second = cross_fitted_recovery_calibration(train, n_splits=3, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(train)
    assert set(first.columns) == {
        "relationship",
        "restricted_economic_signal",
        "reconstructed_economic_signal",
        "cohort_economic_signal",
        "context_score",
        "needs_support",
        "low_signal",
    }


def test_score_people_reports_fixed_and_reliability_weighted_recovery() -> None:
    train = adult_fixture(36)
    test = adult_fixture(12)

    scored = score_people(train, test)
    comparison = summarize_recovery_comparison(scored)
    summary = summarize_results(scored)

    assert {
        "fixed_signal_recovery_score",
        "signal_recovery_score",
        "reconstruction_weight",
    }.issubset(scored.columns)
    assert (
        scored.loc[scored["low_signal"], "reconstruction_weight"] == 0.85
    ).all()
    assert comparison["method"].tolist() == [
        "Fixed 85/15 recovery",
        "Reliability-weighted recovery",
    ]
    recovery = summary.loc[
        summary["method"].eq("Train-fitted reliability-weighted recovery")
    ].iloc[0]
    assert recovery["economic_signal_exposure"] == 0.0
