from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import underserved_recovery_profile


def _scored_events() -> pd.DataFrame:
    rows = []
    for community_index in range(8):
        for household_index in range(2):
            household_id = f"H{community_index}-{household_index}"
            rows.extend(
                [
                    {
                        "community_id": f"C{community_index}",
                        "underserved_score": float(community_index),
                        "household_id": household_id,
                        "low_signal": household_index == 0,
                        "relevant": 1,
                        "predicted_relevance": 0.80,
                    },
                    {
                        "community_id": f"C{community_index}",
                        "underserved_score": float(community_index),
                        "household_id": household_id,
                        "low_signal": household_index == 0,
                        "relevant": 0,
                        "predicted_relevance": 0.20,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _raw_results() -> pd.DataFrame:
    rows = []
    quartile_deltas = {
        "q1_lower": 0.010,
        "q2": 0.020,
        "q3": 0.005,
        "q4_higher": -0.015,
    }
    for scenario, scenario_metadata in underserved_recovery_profile.SCENARIOS.items():
        for seed in [7, 42]:
            for quartile, quartile_label in underserved_recovery_profile.QUARTILES.items():
                for variant, variant_metadata in underserved_recovery_profile.VARIANTS.items():
                    delta = (
                        quartile_deltas[quartile]
                        if variant == underserved_recovery_profile.AGGREGATE_VARIANT
                        else 0.0
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "scenario_display_name": scenario_metadata["display_name"],
                            "variant": variant,
                            "variant_display_name": variant_metadata["display_name"],
                            "underserved_quartile": quartile,
                            "quartile_display_name": quartile_label,
                            "overall_ndcg_at_3": 0.50 + delta,
                            "low_signal_ndcg_at_3": 0.40 + delta,
                            "low_signal_share": 0.30,
                            "num_test_events": 100,
                            "num_low_signal_events": 30,
                            "num_communities": 30,
                        }
                    )
    return pd.DataFrame(rows)


def test_assign_underserved_quartiles_uses_distinct_communities() -> None:
    quartiles = underserved_recovery_profile.assign_underserved_quartiles(
        _scored_events()
    )

    assert len(quartiles) == 8
    assert quartiles["underserved_quartile"].value_counts().to_dict() == {
        "q1_lower": 2,
        "q2": 2,
        "q3": 2,
        "q4_higher": 2,
    }


def test_run_profile_forwards_seed_and_scores_all_pairs(monkeypatch) -> None:
    observed = []
    events = _scored_events()

    def fake_score_experiment(
        received_events,
        experiment_name,
        signal_scenario,
        use_privacy_safe_features,
        numeric_features,
        privacy_noise_seed,
    ):
        assert received_events is events
        observed.append(
            (experiment_name, signal_scenario, use_privacy_safe_features, privacy_noise_seed)
        )
        return received_events

    monkeypatch.setattr(
        underserved_recovery_profile,
        "score_experiment",
        fake_score_experiment,
    )

    results = underserved_recovery_profile.run_underserved_recovery_profile(
        events,
        seed=23,
    )

    assert len(results) == 16
    assert len(observed) == 4
    assert {item[3] for item in observed} == {23}


def test_profile_summary_plot_and_markdown(tmp_path: Path) -> None:
    paired = underserved_recovery_profile.build_paired_recovery(_raw_results())
    summary = underserved_recovery_profile.build_profile_summary(paired)
    figure_path = tmp_path / "underserved_recovery_profile.png"
    markdown_path = tmp_path / "underserved_recovery_profile.md"

    underserved_recovery_profile.plot_underserved_recovery_profile(
        summary,
        figure_path,
    )
    underserved_recovery_profile.write_markdown_summary(summary, markdown_path)

    q4 = summary[summary["underserved_quartile"] == "q4_higher"]
    assert q4["low_signal_recovery_mean"].tolist() == pytest.approx([-0.015, -0.015])
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert "## Interpretation limits" in markdown_path.read_text()
