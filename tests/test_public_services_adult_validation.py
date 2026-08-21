import numpy as np
import pandas as pd
import pytest

from fairprivacysignal.public_services_adult_validation import (
    add_signal_features,
    cross_fitted_recovery_calibration,
    estimate_reliability_weights,
    reconstruct_nonlinear_economic_signal,
    reliability_weighted_signal,
    score_people,
    select_recovery_weights,
    select_ranking_calibrated_weight,
    summarize_recovery_comparison,
    summarize_results,
    validate_recovery_comparison,
    write_report,
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


def test_nonlinear_reconstruction_is_deterministic_and_bounded() -> None:
    train_raw = adult_fixture(90)
    apply_raw = adult_fixture(18)
    train = add_signal_features(train_raw, train_raw)
    apply_to = add_signal_features(train_raw, apply_raw)

    first = reconstruct_nonlinear_economic_signal(train, apply_to, seed=17)
    second = reconstruct_nonlinear_economic_signal(train, apply_to, seed=17)

    assert first == pytest.approx(second)
    assert np.ptp(first) > 0.0
    assert np.all((0.0 <= first) & (first <= 1.0))


def test_recovery_selection_prefers_lower_oof_reconstruction_error() -> None:
    calibration = pd.DataFrame(
        {
            "needs_support": [0, 1, 0, 1],
            "low_signal": [False, False, False, False],
            "context_score": [0.0, 0.0, 0.0, 0.0],
            "restricted_economic_signal": [0.9, 0.1, 0.8, 0.2],
            "reconstructed_economic_signal": [0.1, 0.9, 0.2, 0.8],
            "nonlinear_economic_signal": [0.9, 0.1, 0.8, 0.2],
            "cohort_economic_signal": [0.5, 0.5, 0.5, 0.5],
        }
    )

    selected = select_recovery_weights(
        calibration,
        candidate_weights=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        baseline_weights=(1.0, 0.0, 0.0),
        low_signal_tolerance=0.0,
    )

    assert selected == (0.0, 1.0, 0.0)


def test_recovery_selection_rejects_low_signal_regression() -> None:
    calibration = pd.DataFrame(
        {
            "needs_support": [1, 0, 1, 0],
            "low_signal": [False, False, True, True],
            "context_score": [0.0, 0.0, 0.0, 0.0],
            "restricted_economic_signal": [0.9, 0.1, 0.1, 0.9],
            "reconstructed_economic_signal": [0.9, 0.1, 0.9, 0.1],
            "nonlinear_economic_signal": [0.9, 0.1, 0.1, 0.9],
            "cohort_economic_signal": [0.5, 0.5, 0.5, 0.5],
        }
    )

    selected = select_recovery_weights(
        calibration,
        candidate_weights=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        baseline_weights=(1.0, 0.0, 0.0),
        low_signal_tolerance=0.0,
    )

    assert selected == (1.0, 0.0, 0.0)


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
        "nonlinear_economic_signal",
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
        "selected_signal_recovery_score",
        "nonlinear_economic_signal",
        "reconstruction_weight",
        "selected_ridge_weight",
        "selected_nonlinear_weight",
        "selected_cohort_weight",
    }.issubset(scored.columns)
    assert (
        scored.loc[scored["low_signal"], "reconstruction_weight"] == 0.85
    ).all()
    assert comparison["method"].tolist() == [
        "Fixed 85/15 recovery",
        "Reliability-weighted recovery",
        "OOF-selected nonlinear recovery",
    ]
    recovery = summary.loc[
        summary["method"].eq("Train-fitted nonlinear recovery")
    ].iloc[0]
    assert recovery["economic_signal_exposure"] == 0.0
    assert scored[
        [
            "selected_ridge_weight",
            "selected_nonlinear_weight",
            "selected_cohort_weight",
        ]
    ].sum(axis=1).tolist() == pytest.approx([1.0] * len(scored))


def test_report_generation_accepts_the_nonlinear_recovery_summary(tmp_path) -> None:
    scored = score_people(adult_fixture(36), adult_fixture(12))
    summary = summarize_results(scored)
    comparison = summarize_recovery_comparison(scored)
    report_path = tmp_path / "adult.md"

    write_report(summary, comparison, path=report_path)

    assert report_path.is_file()


def recovery_comparison_fixture(
    selected_overall: float = 0.82,
    selected_low_signal: float = 0.76,
    selected_exposure: float = 0.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": [
                "Reliability-weighted recovery",
                "OOF-selected nonlinear recovery",
            ],
            "overall_ndcg_at_1000": [0.80, selected_overall],
            "low_signal_ndcg_at_1000": [0.75, selected_low_signal],
            "economic_signal_exposure": [0.0, selected_exposure],
        }
    )


def test_recovery_validation_accepts_improvement_without_exposure() -> None:
    validate_recovery_comparison(recovery_comparison_fixture())


@pytest.mark.parametrize(
    ("comparison", "message"),
    [
        (recovery_comparison_fixture(selected_overall=0.79), "overall NDCG"),
        (recovery_comparison_fixture(selected_low_signal=0.74), "low-signal NDCG"),
        (recovery_comparison_fixture(selected_exposure=0.01), "exposure"),
    ],
)
def test_recovery_validation_rejects_failed_acceptance_gate(
    comparison: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_recovery_comparison(comparison)
