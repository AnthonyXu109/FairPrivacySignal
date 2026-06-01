import pandas as pd

from fairprivacysignal.benchmark_card import build_benchmark_card


def test_build_benchmark_card_summarizes_auditable_outputs() -> None:
    experiments = [
        "full_signal_raw_baseline",
        "severe_signal_loss_baseline",
        "severe_signal_loss_with_privacy_safe_aggregates",
        "policy_restricted_baseline",
        "policy_restricted_with_privacy_safe_aggregates",
    ]
    recovery_summary = pd.DataFrame(
        [
            {
                "experiment": experiment,
                "avg_privacy_exposure_score_mean": 0.50,
                "avg_privacy_exposure_score_std": 0.01,
                "overall_ndcg_at_3_mean": 0.55,
                "overall_ndcg_at_3_std": 0.01,
                "low_signal_ndcg_at_3_mean": 0.45,
                "low_signal_ndcg_at_3_std": 0.02,
                "ndcg_gap_not_low_minus_low_mean": 0.10,
                "ndcg_gap_not_low_minus_low_std": 0.01,
            }
            for experiment in experiments
        ]
    )
    aggregate_noise_summary = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "display_name": display_name,
                "noise_scale": 1.0,
                "overall_utility_recovery_mean": 0.01,
                "overall_utility_recovery_std": 0.002,
                "low_signal_utility_recovery_mean": 0.012,
                "low_signal_utility_recovery_std": 0.003,
            }
            for scenario, display_name in [
                ("severe_signal_loss", "Severe signal loss"),
                ("policy_restricted", "Policy restricted"),
            ]
        ]
    )
    threshold_sensitivity = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "display_name": display_name,
                "min_cohort_size": threshold,
                "suppressed_event_share": suppressed_share,
                "overall_utility_recovery": 0.01,
                "low_signal_utility_recovery": 0.012,
            }
            for scenario, display_name in [
                ("severe_signal_loss", "Severe signal loss"),
                ("policy_restricted", "Policy restricted"),
            ]
            for threshold, suppressed_share in [(50, 0.01), (200, 0.12)]
        ]
    )

    card = build_benchmark_card(
        communities=pd.DataFrame({"community_id": ["C1", "C2"]}),
        households=pd.DataFrame({"household_id": ["H1", "H2", "H3"]}),
        events=pd.DataFrame({"event": [1, 2, 3, 4]}),
        signal_loss=pd.DataFrame(
            {"scenario": ["full_signal", "severe_signal_loss"]}
        ),
        multiseed_recovery_raw=pd.DataFrame({"seed": [7, 42, 101]}),
        recovery_summary=recovery_summary,
        multiseed_capacity_raw=pd.DataFrame(
            {
                "seed": [7, 42],
                "capacity_rate": [0.10, 0.20],
                "low_signal_floor_fraction": [0.0, 0.5],
            }
        ),
        aggregate_noise_raw=pd.DataFrame(
            {
                "scenario": ["severe_signal_loss", "policy_restricted"],
                "noise_scale": [0.0, 1.0],
                "noise_seed": [7, 42],
            }
        ),
        aggregate_noise_summary=aggregate_noise_summary,
        threshold_sensitivity=threshold_sensitivity,
        recovery_feature_ablation_raw=pd.DataFrame(
            {
                "scenario": ["severe_signal_loss", "policy_restricted"],
                "variant": [
                    "engagement_aggregate_only",
                    "combined_privacy_safe_aggregates",
                ],
                "seed": [7, 42],
            }
        ),
        model_sensitivity_raw=pd.DataFrame(
            {
                "model": ["logistic_regression", "hist_gradient_boosting"],
                "experiment": [
                    "full_signal_raw_baseline",
                    "severe_signal_loss_baseline",
                ],
                "seed": [7, 42],
            }
        ),
        validation_checks=pd.DataFrame(
            [
                {
                    "check": "example required check",
                    "required": True,
                    "status": "PASS",
                },
                {
                    "check": "example informational check",
                    "required": False,
                    "status": "PASS",
                },
            ]
        ),
    )

    assert "**Required-check status:** PASS" in card
    assert "| Synthetic household-service events" in card
    assert "## Multi-Seed Recovery Results" in card
    assert "## Aggregate-Noise Checkpoint" in card
    assert "## Cohort-Threshold Checkpoints" in card
    assert "Recovery feature ablation" in card
    assert "Model sensitivity diagnostic" in card
    assert "[Limitations](limitations.md)" in card
