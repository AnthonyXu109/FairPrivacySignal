from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


EXPECTED_SIGNAL_SCENARIOS = [
    "full_signal",
    "consent_restricted",
    "policy_restricted",
    "severe_signal_loss",
]


EXPECTED_RECOVERY_EXPERIMENTS = [
    "full_signal_raw_baseline",
    "severe_signal_loss_baseline",
    "severe_signal_loss_with_privacy_safe_aggregates",
    "severe_signal_loss_with_privacy_safe_fairness_aware",
    "policy_restricted_baseline",
    "policy_restricted_with_privacy_safe_aggregates",
    "policy_restricted_with_privacy_safe_fairness_aware",
]


EXPECTED_SEEDS = {7, 11, 23, 42, 101}


def _check(
    name: str,
    category: str,
    passed: bool,
    details: str,
    required: bool = True,
) -> dict:
    return {
        "check": name,
        "category": category,
        "required": required,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


def _columns_are_bounded(
    frame: pd.DataFrame,
    columns: List[str],
) -> bool:
    values = frame[columns].to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all())


def build_validation_checks(
    signal_loss: pd.DataFrame,
    recovery: pd.DataFrame,
    capacity: pd.DataFrame,
    score_calibration: pd.DataFrame,
    multiseed_recovery_raw: pd.DataFrame,
    multiseed_capacity_raw: pd.DataFrame,
) -> pd.DataFrame:
    checks = []

    signal_scenarios = set(signal_loss["scenario"])
    missing_signal_scenarios = sorted(
        set(EXPECTED_SIGNAL_SCENARIOS) - signal_scenarios
    )
    checks.append(
        _check(
            "signal-loss scenarios are complete",
            "signal loss",
            not missing_signal_scenarios,
            (
                "all four expected scenarios are present"
                if not missing_signal_scenarios
                else f"missing scenarios: {missing_signal_scenarios}"
            ),
        )
    )

    indexed_signal = signal_loss.set_index("scenario")
    if not missing_signal_scenarios:
        availability = indexed_signal.loc[
            EXPECTED_SIGNAL_SCENARIOS,
            "behavioral_available_share",
        ]
        exposure = indexed_signal.loc[
            EXPECTED_SIGNAL_SCENARIOS,
            "avg_privacy_exposure_score",
        ]
        event_counts = indexed_signal.loc[EXPECTED_SIGNAL_SCENARIOS, "num_events"]

        checks.append(
            _check(
                "signal-loss availability endpoints are correct",
                "signal loss",
                np.isclose(availability.iloc[0], 1.0)
                and np.isclose(availability.iloc[-1], 0.0),
                f"full={availability.iloc[0]:.3f}; severe={availability.iloc[-1]:.3f}",
            )
        )
        checks.append(
            _check(
                "signal-loss availability is monotonic",
                "signal loss",
                availability.is_monotonic_decreasing,
                " >= ".join(f"{value:.3f}" for value in availability),
            )
        )
        checks.append(
            _check(
                "privacy exposure is monotonic",
                "signal loss",
                exposure.is_monotonic_decreasing,
                " >= ".join(f"{value:.3f}" for value in exposure),
            )
        )
        checks.append(
            _check(
                "signal-loss scenarios use the same event count",
                "signal loss",
                event_counts.nunique() == 1 and event_counts.iloc[0] > 0,
                f"events per scenario={int(event_counts.iloc[0])}",
            )
        )

    recovery_experiments = set(recovery["experiment"])
    missing_recovery_experiments = sorted(
        set(EXPECTED_RECOVERY_EXPERIMENTS) - recovery_experiments
    )
    checks.append(
        _check(
            "privacy-recovery experiments are complete",
            "ranking",
            not missing_recovery_experiments,
            (
                "all seven expected recovery experiments are present"
                if not missing_recovery_experiments
                else f"missing experiments: {missing_recovery_experiments}"
            ),
        )
    )
    checks.append(
        _check(
            "privacy-recovery utility metrics are bounded",
            "ranking",
            _columns_are_bounded(
                recovery,
                [
                    "overall_auc",
                    "overall_ndcg_at_3",
                    "low_signal_ndcg_at_3",
                    "not_low_signal_ndcg_at_3",
                ],
            ),
            "AUC and NDCG metrics are finite values in [0, 1]",
        )
    )
    checks.append(
        _check(
            "capacity allocation metrics are bounded",
            "allocation",
            _columns_are_bounded(
                capacity,
                [
                    "allocated_relevance_rate",
                    "low_signal_selection_rate",
                    "not_low_signal_selection_rate",
                    "allocated_low_signal_share",
                ],
            )
            and (capacity["num_allocated"] > 0).all()
            and (capacity["num_allocated"] <= capacity["num_candidate_events"]).all(),
            "allocation rates are in [0, 1] and allocated counts are valid",
        )
    )
    checks.append(
        _check(
            "score-matched calibration has eligible bins",
            "calibration",
            (score_calibration["num_matched_bins"] > 0).all()
            and _columns_are_bounded(
                score_calibration,
                [
                    "low_signal_ece",
                    "not_low_signal_ece",
                    "mean_absolute_matched_relevance_gap",
                ],
            ),
            "every reported scenario has eligible bins and bounded diagnostics",
        )
    )

    recovery_seed_counts = multiseed_recovery_raw.groupby("experiment")["seed"].nunique()
    checks.append(
        _check(
            "privacy-recovery multi-seed coverage is complete",
            "reproducibility",
            set(multiseed_recovery_raw["seed"]) == EXPECTED_SEEDS
            and (recovery_seed_counts == len(EXPECTED_SEEDS)).all(),
            f"seeds={sorted(multiseed_recovery_raw['seed'].unique())}",
        )
    )

    capacity_seed_counts = multiseed_capacity_raw.groupby(
        ["experiment", "capacity_rate", "low_signal_floor_fraction"]
    )["seed"].nunique()
    checks.append(
        _check(
            "allocation-sensitivity multi-seed coverage is complete",
            "reproducibility",
            set(multiseed_capacity_raw["seed"]) == EXPECTED_SEEDS
            and (capacity_seed_counts == len(EXPECTED_SEEDS)).all(),
            f"seeds={sorted(multiseed_capacity_raw['seed'].unique())}",
        )
    )

    recovery_index = recovery.set_index("experiment")
    severe_baseline = recovery_index.loc[
        "severe_signal_loss_baseline",
        "overall_ndcg_at_3",
    ]
    severe_aggregates = recovery_index.loc[
        "severe_signal_loss_with_privacy_safe_aggregates",
        "overall_ndcg_at_3",
    ]
    checks.append(
        _check(
            "privacy-safe aggregates recover severe-loss utility",
            "result diagnostic",
            severe_aggregates > severe_baseline,
            f"severe baseline={severe_baseline:.3f}; aggregates={severe_aggregates:.3f}",
            required=False,
        )
    )

    return pd.DataFrame(checks)


def write_markdown_report(checks: pd.DataFrame, out_path: Path) -> None:
    required = checks[checks["required"]]
    overall_status = "PASS" if (required["status"] == "PASS").all() else "FAIL"
    rows = checks.copy()
    rows["Required"] = rows["required"].map({True: "yes", False: "no"})
    rows["Status"] = rows["status"]
    rows = rows.rename(
        columns={
            "check": "Check",
            "category": "Category",
            "details": "Details",
        }
    )[["Check", "Category", "Required", "Status", "Details"]]

    out_path.write_text(
        "# Benchmark Validation Report\n\n"
        "This report is generated by the reproducible benchmark pipeline. Required "
        "checks enforce stable methodological invariants; informational checks expose "
        "current result behavior without blocking future experimentation.\n\n"
        f"**Overall required-check status:** {overall_status}\n\n"
        + rows.to_markdown(index=False)
        + "\n"
    )


def raise_for_failed_required_checks(checks: pd.DataFrame) -> None:
    failures = checks[(checks["required"]) & (checks["status"] != "PASS")]

    if not failures.empty:
        failed_names = ", ".join(failures["check"])
        raise RuntimeError(f"required benchmark validations failed: {failed_names}")


def main() -> None:
    tables_dir = Path("outputs/tables")
    docs_dir = Path("docs")

    checks = build_validation_checks(
        signal_loss=pd.read_csv(tables_dir / "signal_loss_summary.csv"),
        recovery=pd.read_csv(tables_dir / "privacy_recovery_metrics.csv"),
        capacity=pd.read_csv(tables_dir / "capacity_allocation_metrics.csv"),
        score_calibration=pd.read_csv(
            tables_dir / "score_matched_calibration_summary.csv"
        ),
        multiseed_recovery_raw=pd.read_csv(
            tables_dir / "multiseed_privacy_recovery_raw.csv"
        ),
        multiseed_capacity_raw=pd.read_csv(
            tables_dir / "multiseed_capacity_sensitivity_raw.csv"
        ),
    )

    csv_path = tables_dir / "benchmark_validation_checks.csv"
    markdown_path = docs_dir / "validation_report.md"

    checks.to_csv(csv_path, index=False)
    write_markdown_report(checks, markdown_path)

    print("Benchmark validation checks:")
    print(checks.to_string(index=False))
    print("\nWrote:")
    print(f"- {csv_path}")
    print(f"- {markdown_path}")

    raise_for_failed_required_checks(checks)


if __name__ == "__main__":
    main()
