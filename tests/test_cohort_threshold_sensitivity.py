from pathlib import Path

import pandas as pd

from fairprivacysignal import cohort_threshold_sensitivity


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "display_name": display_name,
                "min_cohort_size": threshold,
                "suppressed_event_share": suppressed_share,
                "overall_utility_recovery": 0.02 - 0.01 * suppressed_share,
                "low_signal_utility_recovery": 0.03 - 0.01 * suppressed_share,
            }
            for scenario, display_name in [
                ("severe_signal_loss", "Severe signal loss"),
                ("policy_restricted", "Policy restricted"),
            ]
            for threshold, suppressed_share in [(25, 0.0), (50, 0.1), (100, 0.3)]
        ]
    )


def test_run_cohort_threshold_sensitivity_forwards_threshold(monkeypatch) -> None:
    observed_thresholds = []

    monkeypatch.setattr(
        cohort_threshold_sensitivity,
        "apply_signal_loss",
        lambda events, scenario: events.assign(scenario=scenario),
    )

    def fake_add_privacy_safe_features(
        events,
        min_cohort_size,
        dp_noise_scale,
        seed,
    ):
        observed_thresholds.append(min_cohort_size)
        return events.assign(
            cohort_suppressed=min_cohort_size > 25,
            service_category="example",
            urbanicity="urban",
            income_band="middle",
            age_group="adult",
        )

    monkeypatch.setattr(
        cohort_threshold_sensitivity,
        "add_privacy_safe_features",
        fake_add_privacy_safe_features,
    )
    monkeypatch.setattr(
        cohort_threshold_sensitivity,
        "evaluate_model",
        lambda frame, experiment, numeric_features: {
            "overall_ndcg_at_3": 0.53 if "cohort_suppressed" in frame else 0.50,
            "low_signal_ndcg_at_3": 0.44 if "cohort_suppressed" in frame else 0.40,
        },
    )

    results = cohort_threshold_sensitivity.run_cohort_threshold_sensitivity(
        pd.DataFrame({"event": [1]}),
        cohort_thresholds=[25, 50],
    )

    assert len(results) == 4
    assert observed_thresholds == [25, 50, 25, 50]
    assert results["suppressed_event_share"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_plot_cohort_threshold_sensitivity(tmp_path: Path) -> None:
    out_path = tmp_path / "cohort_threshold_sensitivity.png"

    cohort_threshold_sensitivity.plot_cohort_threshold_sensitivity(
        _results(),
        out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
