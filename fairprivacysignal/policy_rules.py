import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "policy_rules.json"


REQUIRED_SCENARIOS = {
    "full_signal",
    "consent_restricted",
    "policy_restricted",
    "severe_signal_loss",
}


SCENARIO_FLAGS = {
    "require_consent",
    "exclude_sensitive_cohort",
    "remove_all_behavioral_history",
}


EXPOSURE_FEATURES = {
    "behavioral_available",
    "consent_behavioral",
    "not_sensitive_cohort",
    "age_group_present",
    "income_band_present",
}


def validate_policy_rules(config: Dict) -> Dict:
    scenarios = config.get("behavioral_signal_scenarios")
    weights = config.get("privacy_exposure_weights")

    if not isinstance(scenarios, dict):
        raise ValueError("policy rules must define behavioral_signal_scenarios")

    missing_scenarios = sorted(REQUIRED_SCENARIOS - set(scenarios))
    if missing_scenarios:
        raise ValueError(f"policy rules are missing scenarios: {missing_scenarios}")

    for scenario_name, scenario in scenarios.items():
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario {scenario_name} must be an object")
        if not isinstance(scenario.get("description"), str):
            raise ValueError(f"scenario {scenario_name} must define a description")

        missing_flags = sorted(SCENARIO_FLAGS - set(scenario))
        if missing_flags:
            raise ValueError(
                f"scenario {scenario_name} is missing flags: {missing_flags}"
            )

        for flag in SCENARIO_FLAGS:
            if not isinstance(scenario[flag], bool):
                raise ValueError(f"scenario {scenario_name} flag {flag} must be boolean")

    if not isinstance(weights, dict):
        raise ValueError("policy rules must define privacy_exposure_weights")

    missing_weights = sorted(EXPOSURE_FEATURES - set(weights))
    if missing_weights:
        raise ValueError(f"privacy exposure weights are missing: {missing_weights}")

    for feature, weight in weights.items():
        if feature not in EXPOSURE_FEATURES:
            raise ValueError(f"unknown privacy exposure feature: {feature}")
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"privacy exposure weight for {feature} must be non-negative")

    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("privacy exposure weights must sum to 1.0")

    return config


def load_policy_rules(path: Path = DEFAULT_RULES_PATH) -> Dict:
    config = json.loads(path.read_text())
    return validate_policy_rules(config)


DEFAULT_POLICY_RULES = load_policy_rules()


def behavioral_availability_mask(
    events: pd.DataFrame,
    scenario: str,
    config: Optional[Dict] = None,
) -> np.ndarray:
    rules = DEFAULT_POLICY_RULES if config is None else validate_policy_rules(config)
    scenarios = rules["behavioral_signal_scenarios"]

    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario}")

    scenario_rules = scenarios[scenario]
    behavioral_available = np.ones(len(events), dtype=bool)

    if scenario_rules["require_consent"]:
        behavioral_available &= events["consent_behavioral"].astype(bool).to_numpy()

    if scenario_rules["exclude_sensitive_cohort"]:
        behavioral_available &= ~events["sensitive_cohort"].astype(bool).to_numpy()

    if scenario_rules["remove_all_behavioral_history"]:
        behavioral_available[:] = False

    return behavioral_available


def privacy_exposure_score(
    events: pd.DataFrame,
    behavioral_available: np.ndarray,
    config: Optional[Dict] = None,
) -> pd.Series:
    rules = DEFAULT_POLICY_RULES if config is None else validate_policy_rules(config)
    weights = rules["privacy_exposure_weights"]

    return (
        weights["behavioral_available"] * behavioral_available.astype(float)
        + weights["consent_behavioral"]
        * events["consent_behavioral"].astype(float)
        + weights["not_sensitive_cohort"]
        * (~events["sensitive_cohort"].astype(bool)).astype(float)
        + weights["age_group_present"] * events["age_group"].notna().astype(float)
        + weights["income_band_present"]
        * events["income_band"].notna().astype(float)
    )
