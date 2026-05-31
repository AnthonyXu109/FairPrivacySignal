from pathlib import Path

import numpy as np
import pandas as pd

from fairprivacysignal.policy_rules import (
    DEFAULT_POLICY_RULES,
    behavioral_availability_mask,
    privacy_exposure_score,
)


SCENARIOS = DEFAULT_POLICY_RULES["behavioral_signal_scenarios"]


def apply_signal_loss(events: pd.DataFrame, scenario: str) -> pd.DataFrame:
    df = events.copy()

    behavioral_available = behavioral_availability_mask(df, scenario)

    df["scenario"] = scenario
    df["behavioral_available"] = behavioral_available
    service_signal_col = (
        "historical_service_engagement_count"
        if "historical_service_engagement_count" in df.columns
        else "historical_engagement_count"
    )

    df["available_historical_service_engagement_count"] = np.where(
        behavioral_available,
        df[service_signal_col],
        0,
    )

    df["available_historical_engagement_count"] = np.where(
        behavioral_available,
        df["historical_engagement_count"],
        0,
    )

    # A simple interpretable privacy exposure proxy.
    # Higher means more individual-level signal is available to the model.
    df["privacy_exposure_score"] = privacy_exposure_score(
        df,
        behavioral_available,
    )

    return df


def summarize_scenario(df: pd.DataFrame) -> dict:
    low_signal = df[df["low_signal"] == True]
    not_low_signal = df[df["low_signal"] == False]

    low_relevance = low_signal["relevant"].mean()
    not_low_relevance = not_low_signal["relevant"].mean()

    return {
        "scenario": df["scenario"].iloc[0],
        "num_events": len(df),
        "behavioral_available_share": df["behavioral_available"].mean(),
        "avg_available_historical_service_engagement": df[
            "available_historical_service_engagement_count"
        ].mean(),
        "avg_privacy_exposure_score": df["privacy_exposure_score"].mean(),
        "relevance_rate_low_signal": low_relevance,
        "relevance_rate_not_low_signal": not_low_relevance,
        "low_signal_relevance_gap": not_low_relevance - low_relevance,
    }


def main() -> None:
    data_dir = Path("data/synthetic")
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(data_dir / "synthetic_outreach_events.csv")

    summaries = []
    for scenario in SCENARIOS:
        scenario_df = apply_signal_loss(events, scenario)
        summaries.append(summarize_scenario(scenario_df))

    summary = pd.DataFrame(summaries)
    summary_path = out_dir / "signal_loss_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Signal loss summary:")
    print(summary.round(4).to_string(index=False))
    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
