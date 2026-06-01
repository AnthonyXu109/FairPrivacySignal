from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import community_holdout_robustness


def _split_events() -> pd.DataFrame:
    rows = []
    for community_index in range(10):
        for household_index in range(3):
            rows.append(
                {
                    "community_id": f"C{community_index}",
                    "household_id": f"H{community_index}-{household_index}",
                    "service_category": "food_assistance",
                    "urbanicity": "urban",
                    "income_band": "low",
                    "age_group": "adult",
                }
            )
    return pd.DataFrame(rows)


def _raw_results() -> pd.DataFrame:
    rows = []
    recovery = {
        "household_holdout": 0.015,
        "community_holdout": 0.009,
    }
    for seed in [7, 42]:
        for scenario, display_name in community_holdout_robustness.SCENARIOS.items():
            for split_strategy, split_display_name in (
                community_holdout_robustness.SPLIT_STRATEGIES.items()
            ):
                for variant in community_holdout_robustness.VARIANTS:
                    delta = (
                        recovery[split_strategy]
                        if variant == community_holdout_robustness.AGGREGATE_VARIANT
                        else 0.0
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "scenario_display_name": display_name,
                            "split_strategy": split_strategy,
                            "split_display_name": split_display_name,
                            "variant": variant,
                            "overall_auc": 0.60 + delta,
                            "overall_ndcg_at_3": 0.50 + delta,
                            "low_signal_ndcg_at_3": 0.40 + delta,
                            "not_low_signal_ndcg_at_3": 0.55 + delta,
                            "fallback_event_share": (
                                0.05
                                if split_strategy == "household_holdout"
                                else 0.12
                            ),
                            "unseen_cohort_share": 0.0,
                            "heldout_community_share": (
                                0.0
                                if split_strategy == "household_holdout"
                                else 1.0
                            ),
                            "num_test_events": 100,
                            "num_test_households": 20,
                            "num_test_communities": 4,
                        }
                    )
    return pd.DataFrame(rows)


def test_split_community_events_keeps_communities_disjoint() -> None:
    train, test = community_holdout_robustness.split_community_events(
        _split_events()
    )

    assert set(train["community_id"]).isdisjoint(test["community_id"])
    assert len(test["community_id"].unique()) == 3


def test_context_coverage_metrics_identify_unseen_holdout_rows() -> None:
    train = _split_events().iloc[:3].copy()
    test = pd.concat(
        [
            _split_events().iloc[3:4],
            pd.DataFrame(
                [
                    {
                        "community_id": "C-unseen",
                        "household_id": "H-unseen",
                        "service_category": "housing_support",
                        "urbanicity": "rural",
                        "income_band": "high",
                        "age_group": "senior",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    assert (
        community_holdout_robustness.calculate_unseen_cohort_share(train, test)
        == pytest.approx(0.5)
    )
    assert (
        community_holdout_robustness.calculate_heldout_community_share(train, test)
        == pytest.approx(1.0)
    )


def test_run_diagnostic_scores_all_scenario_split_variant_combinations(
    monkeypatch,
) -> None:
    events = _split_events()
    observed = []

    def fake_apply_signal_loss(received_events, scenario):
        assert received_events is events
        return received_events.assign(scenario=scenario)

    def fake_split(received_events):
        return received_events.iloc[:15].copy(), received_events.iloc[15:].copy()

    def fake_apply_train_fitted(train, test, privacy_safe_feature_options):
        assert privacy_safe_feature_options == {"seed": 23}
        return train, test.assign(cohort_suppressed=False)

    def fake_score_variant(*args, **kwargs):
        observed.append(
            (
                kwargs["scenario"],
                kwargs["split_strategy"],
                kwargs["variant"],
            )
        )
        return {
            "scenario": kwargs["scenario"],
            "split_strategy": kwargs["split_strategy"],
            "variant": kwargs["variant"],
        }

    monkeypatch.setattr(
        community_holdout_robustness,
        "apply_signal_loss",
        fake_apply_signal_loss,
    )
    monkeypatch.setattr(
        community_holdout_robustness,
        "split_household_events",
        fake_split,
    )
    monkeypatch.setattr(
        community_holdout_robustness,
        "split_community_events",
        fake_split,
    )
    monkeypatch.setattr(
        community_holdout_robustness,
        "apply_train_fitted_privacy_safe_features",
        fake_apply_train_fitted,
    )
    monkeypatch.setattr(
        community_holdout_robustness,
        "score_variant",
        fake_score_variant,
    )

    results = community_holdout_robustness.run_community_holdout_robustness(
        events,
        seed=23,
    )

    assert len(results) == 8
    assert len(observed) == 8


def test_summary_plot_and_markdown_render(tmp_path: Path) -> None:
    paired = community_holdout_robustness.build_paired_recovery(_raw_results())
    summary = community_holdout_robustness.build_summary(paired)
    figure_path = tmp_path / "community_holdout_robustness.png"
    markdown_path = tmp_path / "community_holdout_robustness.md"

    community_holdout_robustness.plot_community_holdout_robustness(
        summary,
        figure_path,
    )
    community_holdout_robustness.write_markdown_summary(summary, markdown_path)

    community = summary[
        summary["split_strategy"] == "community_holdout"
    ]
    assert community["overall_recovery_mean"].tolist() == pytest.approx(
        [0.009, 0.009]
    )
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert "## Interpretation limits" in markdown_path.read_text()
