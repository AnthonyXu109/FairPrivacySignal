from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import model_sensitivity


def _raw_results() -> pd.DataFrame:
    baseline_by_scenario = {
        "full_signal_raw_baseline": 0.56,
        "severe_signal_loss_baseline": 0.50,
        "severe_signal_loss_with_privacy_safe_aggregates": 0.52,
        "policy_restricted_baseline": 0.53,
        "policy_restricted_with_privacy_safe_aggregates": 0.54,
    }
    rows = []

    for model_index, (model, model_metadata) in enumerate(
        model_sensitivity.MODELS.items()
    ):
        for experiment, experiment_metadata in model_sensitivity.EXPERIMENTS.items():
            for seed in [7, 42]:
                overall = baseline_by_scenario[experiment] - 0.005 * model_index
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "model_display_name": model_metadata["display_name"],
                        "experiment": experiment,
                        "experiment_display_name": experiment_metadata["display_name"],
                        "overall_auc": overall + 0.10,
                        "overall_ndcg_at_3": overall,
                        "low_signal_ndcg_at_3": overall - 0.08,
                        "ndcg_gap_not_low_minus_low": 0.10,
                    }
                )

    return pd.DataFrame(rows)


def test_run_model_sensitivity_uses_both_model_builders(monkeypatch) -> None:
    observed_builders = []
    observed_options = []

    monkeypatch.setattr(
        model_sensitivity,
        "apply_signal_loss",
        lambda events, scenario: events.assign(scenario=scenario),
    )

    def fake_evaluate_model(
        frame,
        experiment,
        numeric_features,
        model_builder,
        privacy_safe_feature_options,
    ):
        observed_builders.append(model_builder)
        observed_options.append(privacy_safe_feature_options)
        return {
            "overall_auc": 0.60,
            "overall_ndcg_at_3": 0.55,
            "low_signal_ndcg_at_3": 0.45,
            "ndcg_gap_not_low_minus_low": 0.10,
            "aggregate_reference_scope": (
                "train_households_only"
                if privacy_safe_feature_options is not None
                else "not_applicable"
            ),
        }

    monkeypatch.setattr(
        model_sensitivity,
        "evaluate_model",
        fake_evaluate_model,
    )

    results = model_sensitivity.run_model_sensitivity(
        pd.DataFrame({"event": [1]}),
        seed=42,
    )

    assert len(results) == 10
    assert observed_builders == [
        metadata["builder"]
        for metadata in model_sensitivity.MODELS.values()
        for _ in model_sensitivity.EXPERIMENTS
    ]
    assert observed_options == [
        (
            {"seed": 42}
            if metadata["use_privacy_safe_features"]
            else None
        )
        for _ in model_sensitivity.MODELS
        for metadata in model_sensitivity.EXPERIMENTS.values()
    ]


def test_model_sensitivity_summary_plot_and_markdown(tmp_path: Path) -> None:
    raw = _raw_results()
    summary = model_sensitivity.build_model_sensitivity_summary(raw)
    paired = model_sensitivity.build_paired_recovery_summary(raw)
    figure_path = tmp_path / "model_sensitivity.png"
    markdown_path = tmp_path / "model_sensitivity.md"

    model_sensitivity.plot_model_sensitivity(summary, paired, figure_path)
    model_sensitivity.write_markdown_summary(summary, paired, markdown_path)

    severe = paired[paired["scenario"] == "severe_signal_loss"]
    assert severe["overall_recovery_mean"].tolist() == pytest.approx([0.02, 0.02])
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert "## Interpretation limits" in markdown_path.read_text()
