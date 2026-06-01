from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def add_privacy_safe_features(
    events: pd.DataFrame,
    min_cohort_size: int = 50,
    dp_noise_scale: float = 1.0,
    seed: int = 42,
    reference_events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add privacy-safe aggregate/contextual features.

    This module intentionally avoids real personal data and does not use employer-
    specific information. It demonstrates general privacy-preserving feature patterns:
    cohort aggregation, k-thresholding, and DP-style noise on aggregate statistics.

    Note: The DP-style noise here is educational and not a formal production DP guarantee.
    """
    rng = np.random.default_rng(seed)
    df = events.copy()
    reference = events if reference_events is None else reference_events

    cohort_cols = [
        "service_category",
        "urbanicity",
        "income_band",
        "age_group",
    ]

    cohort_stats = (
        reference.groupby(cohort_cols, observed=False)
        .agg(
            cohort_size=("household_id", "nunique"),
            cohort_avg_engagement=("historical_service_engagement_count", "mean"),
            cohort_avg_underserved=("underserved_score", "mean"),
            cohort_avg_food_risk=("food_access_risk", "mean"),
            cohort_avg_health_need=("health_need_score", "mean"),
            cohort_avg_housing_pressure=("housing_pressure", "mean"),
        )
        .reset_index()
    )

    noisy_cols = [
        "cohort_avg_engagement",
        "cohort_avg_underserved",
        "cohort_avg_food_risk",
        "cohort_avg_health_need",
        "cohort_avg_housing_pressure",
    ]

    for col in noisy_cols:
        cohort_stats[f"privacy_safe_{col}"] = cohort_stats[col] + rng.laplace(
            loc=0.0,
            scale=dp_noise_scale / np.sqrt(cohort_stats["cohort_size"].clip(lower=1)),
            size=len(cohort_stats),
        )

    # k-thresholding: suppress aggregates for small cohorts.
    safe_mask = cohort_stats["cohort_size"] >= min_cohort_size

    for col in noisy_cols:
        safe_col = f"privacy_safe_{col}"
        cohort_stats.loc[~safe_mask, safe_col] = np.nan

    keep_cols = cohort_cols + ["cohort_size"] + [
        f"privacy_safe_{col}" for col in noisy_cols
    ]

    df = df.merge(cohort_stats[keep_cols], on=cohort_cols, how="left")

    # Fill suppressed or missing cohort aggregates with broad service-level aggregates.
    service_defaults = (
        reference.groupby("service_category", observed=False)
        .agg(
            default_engagement=("historical_service_engagement_count", "mean"),
            default_underserved=("underserved_score", "mean"),
            default_food_risk=("food_access_risk", "mean"),
            default_health_need=("health_need_score", "mean"),
            default_housing_pressure=("housing_pressure", "mean"),
        )
        .reset_index()
    )

    df = df.merge(service_defaults, on="service_category", how="left")

    fill_map = {
        "privacy_safe_cohort_avg_engagement": "default_engagement",
        "privacy_safe_cohort_avg_underserved": "default_underserved",
        "privacy_safe_cohort_avg_food_risk": "default_food_risk",
        "privacy_safe_cohort_avg_health_need": "default_health_need",
        "privacy_safe_cohort_avg_housing_pressure": "default_housing_pressure",
    }

    for safe_col, default_col in fill_map.items():
        df[safe_col] = df[safe_col].fillna(df[default_col])

    df["cohort_size"] = df["cohort_size"].fillna(0).astype(int)
    df["cohort_suppressed"] = df["cohort_size"] < min_cohort_size

    # This feature represents a privacy-safer substitute for raw behavioral history.
    df["privacy_safe_engagement_signal"] = df[
        "privacy_safe_cohort_avg_engagement"
    ]

    drop_cols = list(fill_map.values())
    df = df.drop(columns=drop_cols)

    return df


def summarize_privacy_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "num_events": len(df),
                "avg_cohort_size": df["cohort_size"].mean(),
                "suppressed_cohort_share": df["cohort_suppressed"].mean(),
                "avg_privacy_safe_engagement_signal": df[
                    "privacy_safe_engagement_signal"
                ].mean(),
                "avg_privacy_safe_underserved": df[
                    "privacy_safe_cohort_avg_underserved"
                ].mean(),
            }
        ]
    )


def main() -> None:
    data_dir = Path("data/synthetic")
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_dir / "synthetic_outreach_events.csv")

    transformed = add_privacy_safe_features(
        events,
        min_cohort_size=50,
        dp_noise_scale=1.0,
        seed=42,
    )

    summary = summarize_privacy_features(transformed)
    out_path = out_dir / "privacy_safe_feature_summary.csv"
    summary.to_csv(out_path, index=False)

    print("Privacy-safe feature summary:")
    print(summary.round(4).to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
