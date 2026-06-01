import numpy as np
import pandas as pd

from fairprivacysignal import multiseed_evaluation, privacy_recovery
from fairprivacysignal.visualize_results import ANNOTATION_STYLE, NAME_MAP


def test_run_experiments_forwards_privacy_noise_seed(monkeypatch) -> None:
    monkeypatch.setattr(
        privacy_recovery,
        "apply_signal_loss",
        lambda events, scenario: events,
    )
    observed_options = []

    def fake_evaluate_model(
        df,
        experiment,
        numeric_features,
        fairness_aware=False,
        privacy_safe_feature_options=None,
    ):
        observed_options.append(privacy_safe_feature_options)
        return {
            "experiment": experiment,
            "fairness_aware": fairness_aware,
        }

    monkeypatch.setattr(
        privacy_recovery,
        "evaluate_model",
        fake_evaluate_model,
    )

    results = privacy_recovery.run_experiments(
        pd.DataFrame(),
        privacy_noise_seed=23,
    )

    assert observed_options == [
        None,
        None,
        {"seed": 23},
        {"seed": 23},
        None,
        {"seed": 23},
        {"seed": 23},
    ]
    assert results["experiment"].tolist() == [
        "full_signal_raw_baseline",
        "severe_signal_loss_baseline",
        "severe_signal_loss_with_privacy_safe_aggregates",
        "severe_signal_loss_with_privacy_safe_fairness_aware",
        "policy_restricted_baseline",
        "policy_restricted_with_privacy_safe_aggregates",
        "policy_restricted_with_privacy_safe_fairness_aware",
    ]
    assert results["fairness_aware"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_evaluate_seed_uses_same_seed_for_data_and_privacy_noise(monkeypatch) -> None:
    events = pd.DataFrame({"household_id": [1]})
    observed_noise_seeds = []

    monkeypatch.setattr(
        multiseed_evaluation,
        "generate_all",
        lambda n_communities, n_households, seed: (None, None, None, events),
    )

    def fake_run_experiments(received_events, privacy_noise_seed):
        assert received_events is events
        observed_noise_seeds.append(privacy_noise_seed)
        return pd.DataFrame([{"experiment": "example"}])

    monkeypatch.setattr(
        multiseed_evaluation,
        "run_experiments",
        fake_run_experiments,
    )

    results = multiseed_evaluation.evaluate_seed(11)

    assert observed_noise_seeds == [11]
    assert results["seed"].tolist() == [11]


def test_correct_positive_reweighting_reverses_odds_shift() -> None:
    predictions = np.array([0.50, 0.75])

    corrected = privacy_recovery.correct_positive_reweighting(
        predictions,
        positive_weight_multiplier=3.0,
    )

    np.testing.assert_allclose(corrected, np.array([0.25, 0.50]))


def test_single_seed_visualization_covers_all_experiments(monkeypatch) -> None:
    monkeypatch.setattr(
        privacy_recovery,
        "apply_signal_loss",
        lambda events, scenario: events,
    )
    monkeypatch.setattr(
        privacy_recovery,
        "evaluate_model",
        lambda df, experiment, numeric_features, fairness_aware=False, privacy_safe_feature_options=None: {
            "experiment": experiment,
        },
    )

    results = privacy_recovery.run_experiments(pd.DataFrame())

    assert set(results["experiment"]) == set(NAME_MAP)
    assert set(results["experiment"]) == set(ANNOTATION_STYLE)
