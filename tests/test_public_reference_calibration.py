from pathlib import Path

import pandas as pd
import pytest

from fairprivacysignal import public_reference_calibration


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "median_household_income",
                "display_name": "Median household income",
                "synthetic_column": "median_income",
                "aggregation": "population_weighted_mean",
                "reference_value": 80000,
                "unit": "usd",
                "source_name": "U.S. Census Bureau QuickFacts",
                "source_period": "2020-2024",
                "source_url": "https://www.census.gov/quickfacts/example",
                "retrieved_on": "2026-05-31",
                "notes": "test target",
            },
            {
                "metric": "broadband_subscription_share",
                "display_name": "Broadband subscription share",
                "synthetic_column": "broadband_access",
                "aggregation": "population_weighted_mean",
                "reference_value": 0.90,
                "unit": "share",
                "source_name": "U.S. Census Bureau QuickFacts",
                "source_period": "2020-2024",
                "source_url": "https://www.census.gov/quickfacts/example",
                "retrieved_on": "2026-05-31",
                "notes": "test target",
            },
            {
                "metric": "unemployment_rate",
                "display_name": "Unemployment rate",
                "synthetic_column": "unemployment_rate",
                "aggregation": "population_weighted_mean",
                "reference_value": 0.052,
                "unit": "share",
                "source_name": "U.S. Census Bureau ACS 5-Year Data Profile DP03",
                "source_period": "2020-2024",
                "source_url": "https://data.census.gov/table/example",
                "retrieved_on": "2026-06-01",
                "notes": "test target",
            },
        ]
    )


def _communities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population": [1, 3],
            "median_income": [50000, 70000],
            "broadband_access": [0.60, 1.00],
            "unemployment_rate": [0.04, 0.08],
        }
    )


def test_population_weighted_mean() -> None:
    assert public_reference_calibration.population_weighted_mean(
        [10, 20],
        [1, 3],
    ) == pytest.approx(17.5)


def test_load_public_reference_targets_accepts_documented_census_sources(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "public_reference_targets.csv"
    _targets().to_csv(targets_path, index=False)

    targets = public_reference_calibration.load_public_reference_targets(targets_path)

    assert set(targets["metric"]) == {
        "median_household_income",
        "broadband_subscription_share",
        "unemployment_rate",
    }


def test_load_public_reference_targets_rejects_undocumented_source(
    tmp_path: Path,
) -> None:
    targets_path = tmp_path / "public_reference_targets.csv"
    targets = _targets()
    targets.loc[targets["metric"] == "unemployment_rate", "source_url"] = (
        "https://example.com/table"
    )
    targets.to_csv(targets_path, index=False)

    with pytest.raises(ValueError, match="documented Census sources"):
        public_reference_calibration.load_public_reference_targets(targets_path)


def test_build_public_reference_comparison() -> None:
    comparison = public_reference_calibration.build_public_reference_comparison(
        _communities(),
        _targets(),
    ).set_index("metric")

    assert comparison.loc["median_household_income", "synthetic_value"] == pytest.approx(
        65000
    )
    assert comparison.loc[
        "broadband_subscription_share",
        "synthetic_as_share_of_reference",
    ] == pytest.approx(0.9 / 0.9)
    assert comparison.loc["unemployment_rate", "synthetic_value"] == pytest.approx(
        0.07
    )


def test_plot_public_reference_comparison(tmp_path: Path) -> None:
    out_path = tmp_path / "public_reference_calibration.png"
    comparison = public_reference_calibration.build_public_reference_comparison(
        _communities(),
        _targets(),
    )

    public_reference_calibration.plot_public_reference_comparison(comparison, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
