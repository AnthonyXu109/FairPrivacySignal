import json

import numpy as np
import pandas as pd
import pytest

from fairprivacysignal.policy_rules import (
    DEFAULT_RULES_PATH,
    behavioral_availability_mask,
    load_policy_rules,
    privacy_exposure_score,
    validate_policy_rules,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "consent_behavioral": [True, False, True],
            "sensitive_cohort": [False, False, True],
            "age_group": ["adult", "adult", "youth"],
            "income_band": ["middle", "low", "middle"],
        }
    )


def test_default_policy_rules_load_from_readable_config() -> None:
    config = load_policy_rules()

    assert DEFAULT_RULES_PATH.name == "policy_rules.json"
    assert set(config["behavioral_signal_scenarios"]) == {
        "full_signal",
        "consent_restricted",
        "policy_restricted",
        "severe_signal_loss",
    }


def test_policy_rules_produce_expected_behavioral_masks() -> None:
    events = _events()

    np.testing.assert_array_equal(
        behavioral_availability_mask(events, "full_signal"),
        np.array([True, True, True]),
    )
    np.testing.assert_array_equal(
        behavioral_availability_mask(events, "consent_restricted"),
        np.array([True, False, True]),
    )
    np.testing.assert_array_equal(
        behavioral_availability_mask(events, "policy_restricted"),
        np.array([True, False, False]),
    )
    np.testing.assert_array_equal(
        behavioral_availability_mask(events, "severe_signal_loss"),
        np.array([False, False, False]),
    )


def test_privacy_exposure_score_uses_configured_weights() -> None:
    events = _events()
    behavioral_available = np.array([True, False, False])

    scores = privacy_exposure_score(events, behavioral_available)

    np.testing.assert_allclose(scores, np.array([1.0, 0.40, 0.40]))


def test_policy_rule_validation_rejects_weight_drift() -> None:
    config = json.loads(DEFAULT_RULES_PATH.read_text())
    config["privacy_exposure_weights"]["behavioral_available"] = 0.50

    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_policy_rules(config)
