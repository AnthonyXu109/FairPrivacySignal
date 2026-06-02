from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import heldout_context_shift


def _context_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "community_id": "C1",
                "household_id": "H1",
                "service_category": "food_assistance",
                "income_band": "low",
                "urbanicity": "urban",
                "median_income": 60000.0,
                "unemployment_rate": 0.10,
                "broadband_access": 0.80,
                "food_access_risk": 0.30,
                "health_need_score": 0.40,
                "housing_pressure": 0.35,
                "underserved_score": 0.45,
                "relevant": 1,
                "low_signal": True,
            },
            {
                "community_id": "C2",
                "household_id": "H2",
                "service_category": "housing_support",
                "income_band": "high",
                "urbanicity": "rural",
                "median_income": 40000.0,
                "unemployment_rate": 0.34,
                "broadband_access": 0.36,
                "food_access_risk": 0.93,
                "health_need_score": 0.97,
                "housing_pressure": 0.94,
                "underserved_score": 0.98,
                "relevant": 0,
                "low_signal": False,
            },
        ]
    )


def _raw_results() -> pd.DataFrame:
    rows = []
    for seed in [7, 42]:
        for scenario, metadata in heldout_context_shift.SCENARIOS.items():
            for shift_level, shift_metadata in heldout_context_shift.SHIFT_LEVELS.items():
                for variant in heldout_context_shift.VARIANTS:
                    delta = (
                        0.01 + 0.002 * shift_metadata["strength"]
                        if variant == heldout_context_shift.AGGREGATE_VARIANT
                        else 0.0
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "scenario_display_name": metadata["display_name"],
                            "shift_level": shift_level,
                            "shift_display_name": shift_metadata["display_name"],
                            "shift_strength": shift_metadata["strength"],
                            "variant": variant,
                            "overall_auc": 0.60 + delta,
                            "overall_ndcg_at_3": 0.50 + delta,
                            "low_signal_ndcg_at_3": 0.40 + delta,
                            "not_low_signal_ndcg_at_3": 0.55 + delta,
                            "fallback_event_share": 0.05,
                            "bucket_migration_share": (
                                shift_metadata["strength"]
                                * heldout_context_shift.MAX_BUCKET_MIGRATION_SHARE
                            ),
                            "num_test_events": 100,
                            "num_test_households": 20,
                        }
                    )
    return pd.DataFrame(rows)


def test_apply_holdout_context_shift_preserves_labels_and_clips_bounds() -> None:
    events = _context_events()
    shifted = heldout_context_shift.apply_holdout_context_shift(events, strength=1.0)
    unchanged = heldout_context_shift.apply_holdout_context_shift(events, strength=0.0)

    assert unchanged[list(heldout_context_shift.CONTEXT_SHIFT_RULES)].equals(
        events[list(heldout_context_shift.CONTEXT_SHIFT_RULES)]
    )
    assert shifted[["community_id", "household_id", "relevant", "low_signal"]].equals(
        events[["community_id", "household_id", "relevant", "low_signal"]]
    )
    assert shifted.loc[0, "median_income"] < events.loc[0, "median_income"]
    assert shifted.loc[0, "broadband_access"] < events.loc[0, "broadband_access"]
    assert shifted.loc[0, "unemployment_rate"] > events.loc[0, "unemployment_rate"]
    assert shifted.loc[1, "unemployment_rate"] == pytest.approx(0.35)
    assert shifted.loc[1, "underserved_score"] == pytest.approx(1.0)
    assert not unchanged["context_bucket_migrated"].any()
    assert shifted["context_bucket_migrated"].any()
    migrated = shifted[shifted["context_bucket_migrated"]]
    assert not migrated["income_band"].equals(
        events.loc[migrated.index, "income_band"]
    )


def test_apply_holdout_context_shift_rejects_invalid_strength() -> None:
    with pytest.raises(ValueError, match="strength"):
        heldout_context_shift.apply_holdout_context_shift(
            _context_events(),
            strength=1.1,
        )


def test_run_diagnostic_scores_all_scenario_shift_variant_combinations(
    monkeypatch,
) -> None:
    events = _context_events()
    observed = []

    def fake_apply_signal_loss(received_events, scenario):
        assert received_events is events
        return received_events.assign(scenario=scenario)

    def fake_split(received_events):
        return received_events.iloc[:1].copy(), received_events.iloc[1:].copy()

    def fake_add_privacy_safe(received_events, reference_events, seed):
        assert seed == 23
        assert len(reference_events) == 1
        return received_events.assign(cohort_suppressed=False)

    def fake_fit(train, variant):
        return variant

    def fake_score(model, test, **kwargs):
        observed.append(
            (
                kwargs["scenario"],
                kwargs["shift_level"],
                kwargs["variant"],
            )
        )
        return {
            "scenario": kwargs["scenario"],
            "shift_level": kwargs["shift_level"],
            "variant": kwargs["variant"],
        }

    monkeypatch.setattr(heldout_context_shift, "apply_signal_loss", fake_apply_signal_loss)
    monkeypatch.setattr(heldout_context_shift, "split_household_events", fake_split)
    monkeypatch.setattr(
        heldout_context_shift,
        "add_privacy_safe_features",
        fake_add_privacy_safe,
    )
    monkeypatch.setattr(heldout_context_shift, "fit_variant_model", fake_fit)
    monkeypatch.setattr(heldout_context_shift, "score_shifted_holdout", fake_score)

    results = heldout_context_shift.run_heldout_context_shift(events, seed=23)

    assert len(results) == 12
    assert len(observed) == 12


def test_summary_plot_and_markdown_render(tmp_path: Path) -> None:
    paired = heldout_context_shift.build_paired_recovery(_raw_results())
    summary = heldout_context_shift.build_summary(paired)
    figure_path = tmp_path / "heldout_context_shift.png"
    markdown_path = tmp_path / "heldout_context_shift.md"

    heldout_context_shift.plot_heldout_context_shift(summary, figure_path)
    heldout_context_shift.write_markdown_summary(summary, markdown_path)

    for scenario in heldout_context_shift.SCENARIOS:
        ordered = (
            summary[summary["scenario"] == scenario]
            .set_index("shift_level")
            .loc[list(heldout_context_shift.SHIFT_LEVELS)]
        )
        assert ordered["overall_recovery_mean"].tolist() == pytest.approx(
            [0.010, 0.011, 0.012]
        )
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert "synthetic covariate-drift proxy" in markdown_path.read_text()
