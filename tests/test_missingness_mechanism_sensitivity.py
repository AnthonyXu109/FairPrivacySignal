from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import missingness_mechanism_sensitivity as sensitivity


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "broadband_access": [0.1, 0.2, 0.8, 0.9] * 5,
            "underserved_score": [0.9, 0.8, 0.2, 0.1] * 5,
            "consent_behavioral": [False, False, True, True] * 5,
            "historical_service_engagement_count": [0, 1, 8, 12] * 5,
        }
    )


def test_matched_masks_preserve_quantity_and_change_incidence() -> None:
    events = _events()
    masks = {
        mechanism: sensitivity.build_matched_availability_mask(
            events,
            mechanism,
            target_share=0.50,
            seed=42,
        )
        for mechanism in sensitivity.MECHANISMS
    }

    assert {mask.mean() for mask in masks.values()} == {0.50}
    assert not (masks["uniform_random"] == masks["observed_context"]).all()
    assert not (masks["uniform_random"] == masks["signal_dependent"]).all()


def test_summary_plot_and_markdown_render(tmp_path: Path) -> None:
    rows = []
    for mechanism, metadata in sensitivity.MECHANISMS.items():
        for seed in [7, 42]:
            for variant in sensitivity.VARIANTS:
                recovery = 0.0 if variant == sensitivity.BASELINE_VARIANT else 0.02
                rows.append(
                    {
                        "seed": seed,
                        "mechanism": mechanism,
                        "mechanism_display_name": metadata["display_name"],
                        "mechanism_short_name": metadata["short_name"],
                        "variant": variant,
                        "variant_display_name": sensitivity.VARIANTS[variant],
                        "behavioral_available_share": 0.56,
                        "low_signal_available_share": 0.45,
                        "not_low_signal_available_share": 0.62,
                        "availability_gap_not_low_minus_low": 0.17,
                        "overall_ndcg_at_3": 0.50 + recovery,
                        "low_signal_ndcg_at_3": 0.40 + recovery,
                    }
                )

    summary = sensitivity.build_summary(pd.DataFrame(rows))
    figure_path = tmp_path / "mechanisms.png"
    markdown_path = tmp_path / "mechanisms.md"
    sensitivity.plot_mechanism_sensitivity(summary, figure_path)
    sensitivity.write_markdown_summary(summary, markdown_path)

    aggregate = summary[summary["variant"] == sensitivity.AGGREGATE_VARIANT]
    assert aggregate["overall_recovery_mean"].tolist() == pytest.approx(
        [0.02, 0.02, 0.02]
    )
    assert figure_path.exists()
    assert "signal quantity from incidence" in markdown_path.read_text()
