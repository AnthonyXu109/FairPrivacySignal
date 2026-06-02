from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fairprivacysignal import pairwise_ranking_sensitivity
from fairprivacysignal.privacy_recovery import BASE_NUMERIC_FEATURES


def _pairwise_events() -> pd.DataFrame:
    rows = []
    for household_index in range(8):
        for service_index, relevant in enumerate([0, 1, 0]):
            row = {
                "household_id": f"H{household_index}",
                "service_category": f"service_{service_index}",
                "age_group": "adult",
                "income_band": "middle",
                "urbanicity": "urban",
                "relevant": relevant,
            }
            row.update({feature: 0.0 for feature in BASE_NUMERIC_FEATURES})
            row["available_historical_service_engagement_count"] = float(
                service_index == 1
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _raw_results() -> pd.DataFrame:
    baseline_by_scenario = {
        "full_signal_raw_baseline": 0.56,
        "severe_signal_loss_baseline": 0.50,
        "severe_signal_loss_with_privacy_safe_aggregates": 0.52,
        "policy_restricted_baseline": 0.53,
        "policy_restricted_with_privacy_safe_aggregates": 0.54,
    }
    rows = []

    for objective_index, (objective, metadata) in enumerate(
        pairwise_ranking_sensitivity.OBJECTIVES.items()
    ):
        for experiment, experiment_metadata in (
            pairwise_ranking_sensitivity.EXPERIMENTS.items()
        ):
            for seed in [7, 42]:
                overall = baseline_by_scenario[experiment] - 0.005 * objective_index
                rows.append(
                    {
                        "seed": seed,
                        "objective": objective,
                        "objective_display_name": metadata["display_name"],
                        "experiment": experiment,
                        "experiment_display_name": experiment_metadata["display_name"],
                        "overall_auc": overall + 0.10,
                        "overall_ndcg_at_3": overall,
                        "low_signal_ndcg_at_3": overall - 0.08,
                        "not_low_signal_ndcg_at_3": overall + 0.02,
                        "ndcg_gap_not_low_minus_low": 0.10,
                        "num_training_pairs": 100,
                    }
                )

    return pd.DataFrame(rows)


def test_linear_pairwise_ranker_learns_ordered_service_pairs() -> None:
    events = _pairwise_events()
    ranker = pairwise_ranking_sensitivity.LinearPairwiseRanker(
        BASE_NUMERIC_FEATURES
    ).fit(events)
    scored = ranker.predict_proba(events)[:, 1]

    assert ranker.num_training_pairs_ == 16
    for household_id, group in events.assign(score=scored).groupby("household_id"):
        assert household_id
        positive = group[group["relevant"] == 1]["score"].iloc[0]
        negative = group[group["relevant"] == 0]["score"].max()
        assert positive > negative


def test_run_sensitivity_scores_both_training_objectives(monkeypatch) -> None:
    observed = []
    events = pd.DataFrame({"event": [1]})
    scored = pd.DataFrame(
        {
            "household_id": ["H1", "H1", "H2", "H2"],
            "low_signal": [True, True, False, False],
            "relevant": [1, 0, 1, 0],
            "predicted_relevance": [0.8, 0.2, 0.8, 0.2],
        }
    )

    monkeypatch.setattr(
        pairwise_ranking_sensitivity,
        "apply_signal_loss",
        lambda received, scenario: received.assign(scenario=scenario),
    )
    monkeypatch.setattr(
        pairwise_ranking_sensitivity,
        "split_household_events",
        lambda frame: (frame, frame),
    )
    monkeypatch.setattr(
        pairwise_ranking_sensitivity,
        "apply_train_fitted_privacy_safe_features",
        lambda train, test, privacy_safe_feature_options: (train, test),
    )

    def fake_score(train, test, numeric_features):
        observed.append(tuple(numeric_features))
        return scored, 3

    monkeypatch.setattr(
        pairwise_ranking_sensitivity,
        "_score_pointwise",
        fake_score,
    )
    monkeypatch.setattr(
        pairwise_ranking_sensitivity,
        "_score_pairwise",
        fake_score,
    )

    results = pairwise_ranking_sensitivity.run_pairwise_ranking_sensitivity(
        events,
        seed=42,
    )

    assert len(results) == 10
    assert len(observed) == 10
    assert set(results["objective"]) == set(
        pairwise_ranking_sensitivity.OBJECTIVES
    )


def test_summary_plot_and_markdown_render(tmp_path: Path) -> None:
    raw = _raw_results()
    summary = pairwise_ranking_sensitivity.build_summary(raw)
    paired = pairwise_ranking_sensitivity.build_paired_recovery_summary(raw)
    figure_path = tmp_path / "pairwise_ranking_sensitivity.png"
    markdown_path = tmp_path / "pairwise_ranking_sensitivity.md"

    pairwise_ranking_sensitivity.plot_pairwise_ranking_sensitivity(
        summary,
        paired,
        figure_path,
    )
    pairwise_ranking_sensitivity.write_markdown_summary(
        summary,
        paired,
        markdown_path,
    )

    severe = paired[paired["scenario"] == "severe_signal_loss"]
    assert severe["overall_recovery_mean"].tolist() == pytest.approx([0.02, 0.02])
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert "## Interpretation limits" in markdown_path.read_text()
