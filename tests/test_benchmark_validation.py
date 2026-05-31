from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import benchmark_validation


def _signal_loss() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": benchmark_validation.EXPECTED_SIGNAL_SCENARIOS,
            "num_events": [100, 100, 100, 100],
            "behavioral_available_share": [1.0, 0.8, 0.6, 0.0],
            "avg_privacy_exposure_score": [0.95, 0.82, 0.70, 0.45],
        }
    )


def _recovery() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment": benchmark_validation.EXPECTED_RECOVERY_EXPERIMENTS,
            "overall_auc": [0.60] * 7,
            "overall_ndcg_at_3": [0.60, 0.50, 0.55, 0.56, 0.52, 0.54, 0.55],
            "low_signal_ndcg_at_3": [0.50] * 7,
            "not_low_signal_ndcg_at_3": [0.60] * 7,
        }
    )


def _capacity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "allocated_relevance_rate": [0.50],
            "low_signal_selection_rate": [0.10],
            "not_low_signal_selection_rate": [0.20],
            "allocated_low_signal_share": [0.30],
            "num_allocated": [10],
            "num_candidate_events": [100],
        }
    )


def _score_calibration() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_matched_bins": [4],
            "low_signal_ece": [0.20],
            "not_low_signal_ece": [0.10],
            "mean_absolute_matched_relevance_gap": [0.15],
        }
    )


def _multiseed_recovery_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"experiment": experiment, "seed": seed}
            for experiment in benchmark_validation.EXPECTED_RECOVERY_EXPERIMENTS
            for seed in benchmark_validation.EXPECTED_SEEDS
        ]
    )


def _multiseed_capacity_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": "example",
                "capacity_rate": 0.15,
                "low_signal_floor_fraction": 0.50,
                "seed": seed,
            }
            for seed in benchmark_validation.EXPECTED_SEEDS
        ]
    )


def _checks() -> pd.DataFrame:
    return benchmark_validation.build_validation_checks(
        signal_loss=_signal_loss(),
        recovery=_recovery(),
        capacity=_capacity(),
        score_calibration=_score_calibration(),
        multiseed_recovery_raw=_multiseed_recovery_raw(),
        multiseed_capacity_raw=_multiseed_capacity_raw(),
    )


def test_benchmark_validation_checks_pass_for_consistent_metrics() -> None:
    checks = _checks()

    benchmark_validation.raise_for_failed_required_checks(checks)

    assert (checks[checks["required"]]["status"] == "PASS").all()


def test_benchmark_validation_rejects_signal_loss_drift() -> None:
    checks = _checks()
    checks.loc[
        checks["check"] == "signal-loss availability endpoints are correct",
        "status",
    ] = "FAIL"

    with pytest.raises(RuntimeError, match="availability endpoints"):
        benchmark_validation.raise_for_failed_required_checks(checks)


def test_benchmark_validation_report_renders(tmp_path: Path) -> None:
    out_path = tmp_path / "validation_report.md"

    benchmark_validation.write_markdown_report(_checks(), out_path)

    assert "**Overall required-check status:** PASS" in out_path.read_text()
