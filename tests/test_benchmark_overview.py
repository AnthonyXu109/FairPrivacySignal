from pathlib import Path

import pandas as pd

from fairprivacysignal.benchmark_overview import (
    CAPACITY_SCENARIOS,
    MULTISEED_ORDER,
    plot_benchmark_overview,
)


def test_benchmark_overview_renders_from_expected_metrics(tmp_path: Path) -> None:
    multiseed = pd.DataFrame(
        [
            {
                "experiment": experiment,
                "overall_ndcg_at_3_mean": 0.50 + index * 0.005,
                "overall_ndcg_at_3_std": 0.01,
                "ndcg_gap_not_low_minus_low_mean": 0.10 + index * 0.002,
                "avg_privacy_exposure_score_mean": 0.45 + index * 0.07,
            }
            for index, experiment in enumerate(MULTISEED_ORDER)
        ]
    )
    capacity = pd.DataFrame(
        [
            {
                "experiment": experiment,
                "allocation_policy": policy,
                "allocated_relevance_rate": 0.52 - index * 0.03 - policy_index * 0.04,
                "allocated_low_signal_share": 0.02 + index * 0.02 + policy_index * 0.20,
            }
            for index, experiment in enumerate(CAPACITY_SCENARIOS)
            for policy_index, policy in enumerate(
                ["utility_only", "fairness_constrained"]
            )
        ]
    )
    out_path = tmp_path / "benchmark_overview.png"

    plot_benchmark_overview(multiseed, capacity, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
