from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = {
    "full_signal": {
        "description": "All synthetic individual-level features are available.",
    },
    "consent_restricted": {
        "description": "Behavioral history is removed when household consent is false.",
    },
    "policy_restricted": {
        "description": "Behavioral history is removed for non-consented or sensitive-cohort households.",
    },
    "severe_signal_loss": {
        "description": "Individual behavioral history is removed for all households.",
    },
}


def apply_signal_loss(events: pd.DataFrame, scenario: str) -> pd.DataFrame:
    df = events.copy()

    if scenario == "full_signal":
        behavioral_available = np.ones(len(df), dtype=bool)

    elif scenario == "consent_restricted":
        behavioral_available = df["consent_behavioral"].astype(bool).to_numpy()

    elif scenario == "policy_restricted":
        behavioral_available = (
            df["consent_behavioral"].astype(bool)
            & ~df["sensitive_cohort"].astype(bool)
        ).to_numpy()

    elif scenario == "severe_signal_loss":
        behavioral_available = np.zeros(len(df), dtype=bool)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

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
    df["privacy_exposure_score"] = (
        0.45 * df["behavioral_available"].astype(float)
        + 0.15 * df["consent_behavioral"].astype(float)
        + 0.15 * (~df["sensitive_cohort"].astype(bool)).astype(float)
        + 0.15 * (df["age_group"].notna()).astype(float)
        + 0.10 * (df["income_band"].notna()).astype(float)
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
