from pathlib import Path

import numpy as np
import pandas as pd

from fairprivacysignal import disparate_uncertainty_audit as audit


def test_bootstrap_households_preserves_event_count_and_relabels_draws() -> None:
    train = pd.DataFrame(
        {
            "household_id": ["h1", "h1", "h2", "h2"],
            "value": [1, 2, 3, 4],
        }
    )

    bootstrapped = audit.bootstrap_households(train, seed=42)

    assert len(bootstrapped) == len(train)
    assert bootstrapped["household_id"].nunique() == 2
    assert bootstrapped["household_id"].str.contains("__bootstrap_").all()


def test_top_k_agreement_detects_stable_and_unstable_groups() -> None:
    test = pd.DataFrame(
        {
            "household_id": ["low"] * 4 + ["not_low"] * 4,
            "low_signal": [True] * 4 + [False] * 4,
        }
    )
    predictions = np.array(
        [
            [0.9, 0.8, 0.7, 0.1, 0.9, 0.8, 0.7, 0.1],
            [0.1, 0.8, 0.7, 0.9, 0.9, 0.8, 0.7, 0.1],
        ]
    )
    low, not_low = audit.top_k_agreement(
        test,
        predictions,
        predictions.mean(axis=0),
        k=2,
    )

    assert low < not_low
    assert not_low == 1.0


def test_summary_plot_and_markdown_render(tmp_path: Path) -> None:
    rows = []
    for experiment, metadata in audit.EXPERIMENTS.items():
        for seed in audit.SEEDS:
            rows.append(
                {
                    "seed": seed,
                    "experiment": experiment,
                    "experiment_display_name": metadata["display_name"],
                    "overall_ndcg_at_3": 0.53,
                    "low_signal_ndcg_at_3": 0.44,
                    "low_signal_prediction_std": 0.03,
                    "not_low_signal_prediction_std": 0.02,
                    "prediction_std_gap_low_minus_not_low": 0.01,
                    "low_signal_top3_agreement": 0.88,
                    "not_low_signal_top3_agreement": 0.92,
                    "top3_agreement_gap_not_low_minus_low": 0.04,
                }
            )
    raw = pd.DataFrame(rows)
    summary = audit.build_summary(raw)
    paired = audit.build_paired_effects(raw)
    figure_path = tmp_path / "uncertainty.png"
    markdown_path = tmp_path / "uncertainty.md"

    audit.plot_uncertainty_audit(summary, figure_path)
    audit.write_markdown_summary(summary, paired, markdown_path)

    assert figure_path.exists()
    assert "training-resample instability" in markdown_path.read_text()
