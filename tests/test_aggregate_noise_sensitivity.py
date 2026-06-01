from pathlib import Path

import pandas as pd

from fairprivacysignal import aggregate_noise_sensitivity


def _raw_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "display_name": display_name,
                "noise_scale": noise_scale,
                "noise_seed": noise_seed,
                "baseline_overall_ndcg_at_3": 0.50,
                "aggregate_overall_ndcg_at_3": 0.53 - 0.005 * noise_scale,
                "overall_utility_recovery": 0.03 - 0.005 * noise_scale,
                "baseline_low_signal_ndcg_at_3": 0.42,
                "aggregate_low_signal_ndcg_at_3": 0.46 - 0.005 * noise_scale,
                "low_signal_utility_recovery": 0.04 - 0.005 * noise_scale,
            }
            for scenario, display_name in [
                ("severe_signal_loss", "Severe signal loss"),
                ("policy_restricted", "Policy restricted"),
            ]
            for noise_scale in [0.0, 1.0]
            for noise_seed in [7, 42]
        ]
    )


def test_run_noise_sensitivity_forwards_scale_and_seed(monkeypatch) -> None:
    observed_parameters = []

    monkeypatch.setattr(
        aggregate_noise_sensitivity,
        "apply_signal_loss",
        lambda events, scenario: events.assign(scenario=scenario),
    )

    def fake_add_privacy_safe_features(events, dp_noise_scale, seed):
        observed_parameters.append((dp_noise_scale, seed))
        return events.assign(dp_noise_scale=dp_noise_scale)

    monkeypatch.setattr(
        aggregate_noise_sensitivity,
        "add_privacy_safe_features",
        fake_add_privacy_safe_features,
    )
    monkeypatch.setattr(
        aggregate_noise_sensitivity,
        "evaluate_model",
        lambda frame, experiment, numeric_features: {
            "overall_ndcg_at_3": 0.55 if "dp_noise_scale" in frame else 0.50,
            "low_signal_ndcg_at_3": 0.45 if "dp_noise_scale" in frame else 0.40,
        },
    )

    results = aggregate_noise_sensitivity.run_noise_sensitivity(
        pd.DataFrame({"event": [1]}),
        noise_scales=[0.0, 1.0],
        noise_seeds=[7, 42],
    )

    assert len(results) == 8
    assert observed_parameters == [
        (0.0, 7),
        (0.0, 42),
        (1.0, 7),
        (1.0, 42),
        (0.0, 7),
        (0.0, 42),
        (1.0, 7),
        (1.0, 42),
    ]


def test_noise_sensitivity_summary_and_plot(tmp_path: Path) -> None:
    summary = aggregate_noise_sensitivity.build_noise_sensitivity_summary(
        _raw_results()
    )
    out_path = tmp_path / "aggregate_noise_sensitivity.png"

    aggregate_noise_sensitivity.plot_noise_sensitivity(summary, out_path)

    assert len(summary) == 4
    assert out_path.exists()
    assert out_path.stat().st_size > 0
