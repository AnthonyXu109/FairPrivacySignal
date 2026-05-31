from pathlib import Path

import pandas as pd

from fairprivacysignal import capacity_sensitivity


def _scored_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service_category": ["example"] * 8,
            "predicted_relevance": [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
            "low_signal": [False, False, False, False, True, True, True, True],
            "relevant": [True, True, True, False, True, False, False, False],
            "experiment": ["example"] * 8,
            "privacy_exposure_score": [0.50] * 8,
            "behavioral_available": [False] * 8,
        }
    )


def test_capacity_sensitivity_tracks_grid_and_relevance_cost(monkeypatch) -> None:
    monkeypatch.setattr(
        capacity_sensitivity,
        "EXPERIMENTS",
        [("example", "example", False, [])],
    )
    monkeypatch.setattr(capacity_sensitivity, "CAPACITY_RATES", [0.50])
    monkeypatch.setattr(
        capacity_sensitivity,
        "LOW_SIGNAL_FLOOR_FRACTIONS",
        [0.0, 1.0],
    )
    monkeypatch.setattr(
        capacity_sensitivity,
        "score_experiment",
        lambda events, experiment_name, signal_scenario, use_privacy_safe_features, numeric_features: _scored_events(),
    )

    results = capacity_sensitivity.run_capacity_sensitivity(pd.DataFrame())

    assert results["low_signal_floor_fraction"].tolist() == [0.0, 1.0]
    assert results.loc[0, "allocated_relevance_cost_vs_utility_only"] == 0.0
    assert results.loc[1, "allocated_low_signal_share"] > results.loc[
        0, "allocated_low_signal_share"
    ]


def test_capacity_sensitivity_plot_renders(tmp_path: Path) -> None:
    rows = []

    for scenario_index, experiment in enumerate(capacity_sensitivity.FRONTIER_SCENARIOS):
        for capacity_rate in capacity_sensitivity.CAPACITY_RATES:
            for floor_fraction in capacity_sensitivity.LOW_SIGNAL_FLOOR_FRACTIONS:
                rows.append(
                    {
                        "experiment": experiment,
                        "capacity_rate": capacity_rate,
                        "low_signal_floor_fraction": floor_fraction,
                        "allocated_relevance_rate": (
                            0.55
                            - scenario_index * 0.04
                            - floor_fraction * 0.05
                        ),
                        "allocated_low_signal_share": 0.02 + floor_fraction * 0.28,
                        "selection_rate_gap_not_low_minus_low": (
                            0.22 - floor_fraction * 0.18
                        ),
                        "allocated_relevance_cost_vs_utility_only": (
                            floor_fraction * 0.05
                        ),
                    }
                )

    out_path = tmp_path / "capacity_sensitivity_frontier.png"

    capacity_sensitivity.plot_capacity_sensitivity(
        pd.DataFrame(rows),
        out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
