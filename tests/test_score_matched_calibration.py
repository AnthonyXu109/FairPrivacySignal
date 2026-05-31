from pathlib import Path

import pandas as pd

from fairprivacysignal import score_matched_calibration


def _scored_events(experiment: str = "example") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment": [experiment] * 8,
            "predicted_relevance": [0.90, 0.88, 0.60, 0.58, 0.89, 0.87, 0.59, 0.57],
            "relevant": [True, True, True, False, True, False, False, False],
            "low_signal": [False, False, False, False, True, True, True, True],
        }
    )


def test_score_matched_bins_measure_within_score_gap() -> None:
    bins = score_matched_calibration.compute_score_matched_bins(
        _scored_events(),
        n_bins=2,
        min_group_count=1,
    )
    summary = score_matched_calibration.summarize_score_matched_bins(bins)

    assert bins["eligible_for_matched_comparison"].tolist() == [True, True]
    assert bins["matched_relevance_gap_not_low_minus_low"].tolist() == [0.5, 0.5]
    assert summary["mean_absolute_matched_relevance_gap"] == 0.5
    assert summary["signed_matched_relevance_gap_not_low_minus_low"] == 0.5
    assert summary["num_matched_bins"] == 2


def test_score_matched_calibration_plot_renders(tmp_path: Path) -> None:
    frames = []
    summaries = []

    for experiment_name, _, _, _ in score_matched_calibration.EXPERIMENTS:
        bins = score_matched_calibration.compute_score_matched_bins(
            _scored_events(experiment_name),
            n_bins=2,
            min_group_count=1,
        )
        frames.append(bins)
        summaries.append(
            score_matched_calibration.summarize_score_matched_bins(bins)
        )

    out_path = tmp_path / "score_matched_calibration.png"
    score_matched_calibration.plot_score_matched_calibration(
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(summaries),
        out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
