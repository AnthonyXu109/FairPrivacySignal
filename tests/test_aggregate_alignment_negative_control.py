from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import aggregate_alignment_negative_control as negative_control


def test_service_permutation_rotates_categories_and_preserves_counts() -> None:
    reference = pd.DataFrame(
        {
            "service_category": ["a", "a", "b", "b", "c", "c"],
            "value": range(6),
        }
    )

    permuted = negative_control.permute_reference_service_categories(reference)

    assert permuted["service_category"].tolist() == ["b", "b", "c", "c", "a", "a"]
    assert sorted(permuted["service_category"]) == sorted(reference["service_category"])
    assert permuted["value"].equals(reference["value"])


def test_summary_plot_and_markdown_render(tmp_path: Path) -> None:
    rows = []
    for scenario, metadata in negative_control.SCENARIOS.items():
        for seed in [7, 42]:
            for variant, variant_metadata in negative_control.VARIANTS.items():
                recovery = {
                    negative_control.BASELINE_VARIANT: 0.0,
                    negative_control.ALIGNED_VARIANT: 0.02,
                    negative_control.PERMUTED_VARIANT: 0.004,
                }[variant]
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "scenario_display_name": metadata["display_name"],
                        "variant": variant,
                        "variant_display_name": variant_metadata["display_name"],
                        "overall_ndcg_at_3": 0.50 + recovery,
                        "low_signal_ndcg_at_3": 0.40 + recovery,
                    }
                )
    summary = negative_control.build_summary(pd.DataFrame(rows))
    figure_path = tmp_path / "negative_control.png"
    markdown_path = tmp_path / "negative_control.md"

    negative_control.plot_negative_control(summary, figure_path)
    negative_control.write_markdown_summary(summary, markdown_path)

    aligned = summary[summary["variant"] == negative_control.ALIGNED_VARIANT]
    assert aligned["overall_recovery_mean"].tolist() == pytest.approx([0.02, 0.02])
    assert figure_path.exists()
    assert "structural negative control" in markdown_path.read_text()
