from pathlib import Path

import pandas as pd

from fairprivacysignal import multiseed_capacity_sensitivity
from fairprivacysignal.capacity_sensitivity import (
    CAPACITY_RATES,
    FRONTIER_SCENARIOS,
    LOW_SIGNAL_FLOOR_FRACTIONS,
)


def _raw_results() -> pd.DataFrame:
    rows = []

    for seed in [7, 11]:
        for scenario_index, experiment in enumerate(FRONTIER_SCENARIOS):
            for capacity_rate in CAPACITY_RATES:
                for floor_fraction in LOW_SIGNAL_FLOOR_FRACTIONS:
                    rows.append(
                        {
                            "seed": seed,
                            "experiment": experiment,
                            "capacity_rate": capacity_rate,
                            "low_signal_floor_fraction": floor_fraction,
                            "allocated_relevance_rate": (
                                0.56
                                - scenario_index * 0.04
                                - floor_fraction * 0.05
                                + seed * 0.0001
                            ),
                            "allocated_low_signal_share": (
                                0.03 + floor_fraction * 0.30 + seed * 0.0001
                            ),
                            "selection_rate_gap_not_low_minus_low": (
                                0.22 - floor_fraction * 0.18 + seed * 0.0001
                            ),
                            "allocated_relevance_cost_vs_utility_only": (
                                floor_fraction * 0.05 + seed * 0.0001
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def test_multiseed_capacity_summary_tracks_mean_and_std() -> None:
    summary = multiseed_capacity_sensitivity.build_summary(_raw_results())
    row = summary[
        (summary["experiment"] == FRONTIER_SCENARIOS[0])
        & (summary["capacity_rate"] == 0.15)
        & (summary["low_signal_floor_fraction"] == 0.50)
    ].iloc[0]

    assert row["allocated_relevance_rate_mean"] > 0
    assert row["allocated_relevance_rate_std"] > 0
    assert row["allocated_relevance_cost_vs_utility_only_mean"] > 0


def test_multiseed_capacity_plot_renders(tmp_path: Path) -> None:
    summary = multiseed_capacity_sensitivity.build_summary(_raw_results())
    out_path = tmp_path / "multiseed_capacity_sensitivity.png"

    multiseed_capacity_sensitivity.plot_multiseed_capacity_sensitivity(
        summary,
        out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
