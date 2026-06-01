from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import recovery_feature_ablation


def _raw_results() -> pd.DataFrame:
    rows = []

    for scenario, scenario_metadata in recovery_feature_ablation.SCENARIOS.items():
        for seed in [7, 42]:
            for index, (variant, variant_metadata) in enumerate(
                recovery_feature_ablation.ABLATIONS.items()
            ):
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "scenario_display_name": scenario_metadata["display_name"],
                        "variant": variant,
                        "variant_display_name": variant_metadata["display_name"],
                        "overall_ndcg_at_3": 0.50 + 0.01 * index,
                        "low_signal_ndcg_at_3": 0.40 + 0.015 * index,
                        "ndcg_gap_not_low_minus_low": 0.10 - 0.005 * index,
                    }
                )

    return pd.DataFrame(rows)


def test_run_feature_ablation_uses_all_feature_groups(monkeypatch) -> None:
    observed_features = []
    observed_seeds = []

    monkeypatch.setattr(
        recovery_feature_ablation,
        "apply_signal_loss",
        lambda events, scenario: events.assign(scenario=scenario),
    )

    def fake_add_privacy_safe_features(events, seed):
        observed_seeds.append(seed)
        return events.assign(privacy_safe=True)

    monkeypatch.setattr(
        recovery_feature_ablation,
        "add_privacy_safe_features",
        fake_add_privacy_safe_features,
    )

    def fake_evaluate_model(frame, experiment, numeric_features):
        observed_features.append(numeric_features)
        return {
            "overall_ndcg_at_3": 0.55,
            "low_signal_ndcg_at_3": 0.45,
            "ndcg_gap_not_low_minus_low": 0.10,
        }

    monkeypatch.setattr(
        recovery_feature_ablation,
        "evaluate_model",
        fake_evaluate_model,
    )

    results = recovery_feature_ablation.run_feature_ablation(
        pd.DataFrame({"event": [1]}),
        seed=42,
    )

    assert len(results) == 8
    assert observed_seeds == [42, 42]
    assert observed_features == [
        metadata["numeric_features"]
        for _ in recovery_feature_ablation.SCENARIOS
        for metadata in recovery_feature_ablation.ABLATIONS.values()
    ]


def test_feature_ablation_summary_plot_and_markdown(tmp_path: Path) -> None:
    summary = recovery_feature_ablation.build_feature_ablation_summary(
        _raw_results()
    )
    figure_path = tmp_path / "recovery_feature_ablation.png"
    markdown_path = tmp_path / "recovery_feature_ablation.md"

    recovery_feature_ablation.plot_feature_ablation(summary, figure_path)
    recovery_feature_ablation.write_markdown_summary(summary, markdown_path)

    combined = summary[
        summary["variant"] == "combined_privacy_safe_aggregates"
    ]
    assert combined["overall_recovery_vs_no_aggregates_mean"].tolist() == pytest.approx(
        [0.03, 0.03]
    )
    assert combined["low_signal_recovery_vs_no_aggregates_mean"].tolist() == pytest.approx(
        [0.045, 0.045]
    )
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert "## Interpretation limits" in markdown_path.read_text()
