import pandas as pd
import pytest

from fairprivacysignal.public_services_adult_validation import (
    estimate_reliability_weights,
    reliability_weighted_signal,
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
