"""
Synthetic public-service outreach data generator for FairPrivacySignal.

This module creates synthetic data for a privacy-preserving and fairness-aware
ranking/matching demo. It does not use real personal data or employer-specific
confidential information.

The synthetic scenario:
- Communities have different public-service needs.
- Households belong to communities and may have different consent states.
- Public services target different needs, such as food assistance, preventive
  health outreach, housing support, job training, and education support.
- Outreach events simulate whether a service is relevant to a household.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


SERVICE_CATEGORIES = [
    "food_assistance",
    "preventive_health",
    "housing_support",
    "job_training",
    "education_support",
    "transportation_support",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_communities(
    n_communities: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic community-level context."""
    community_ids = [f"C{i:04d}" for i in range(n_communities)]

    urbanicity = rng.choice(
        ["urban", "suburban", "rural"],
        size=n_communities,
        p=[0.45, 0.35, 0.20],
    )

    median_income = rng.normal(65000, 18000, n_communities).clip(22000, 140000)
    unemployment_rate = rng.beta(2.0, 12.0, n_communities).clip(0.01, 0.30)
    broadband_access = rng.beta(8.0, 2.0, n_communities).clip(0.35, 0.99)
    food_access_risk = rng.beta(2.2, 5.0, n_communities).clip(0.02, 0.95)
    health_need_score = rng.beta(2.5, 3.5, n_communities).clip(0.02, 0.98)
    housing_pressure = rng.beta(2.0, 4.0, n_communities).clip(0.02, 0.95)

    # Composite underserved score. This is synthetic and not intended to classify
    # any real community.
    underserved_score = (
        0.30 * (1.0 - (median_income - 22000) / (140000 - 22000))
        + 0.25 * unemployment_rate / 0.30
        + 0.20 * (1.0 - broadband_access)
        + 0.15 * food_access_risk
        + 0.10 * housing_pressure
    ).clip(0, 1)

    population = rng.integers(800, 35000, n_communities)

    return pd.DataFrame(
        {
            "community_id": community_ids,
            "urbanicity": urbanicity,
            "population": population,
            "median_income": median_income.round(0).astype(int),
            "unemployment_rate": unemployment_rate.round(4),
            "broadband_access": broadband_access.round(4),
            "food_access_risk": food_access_risk.round(4),
            "health_need_score": health_need_score.round(4),
            "housing_pressure": housing_pressure.round(4),
            "underserved_score": underserved_score.round(4),
        }
    )


def generate_households(
    n_households: int,
    communities: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic household records."""
    community_probs = communities["population"].to_numpy(dtype=float)
    community_probs = community_probs / community_probs.sum()

    community_ids = rng.choice(
        communities["community_id"].to_numpy(),
        size=n_households,
        p=community_probs,
    )

    community_lookup = communities.set_index("community_id")
    underserved = community_lookup.loc[community_ids, "underserved_score"].to_numpy()
    broadband = community_lookup.loc[community_ids, "broadband_access"].to_numpy()
    unemployment = community_lookup.loc[community_ids, "unemployment_rate"].to_numpy()

    household_ids = [f"H{i:07d}" for i in range(n_households)]

    age_group = rng.choice(
        ["youth", "adult", "senior"],
        size=n_households,
        p=[0.18, 0.62, 0.20],
    )

    income_band = rng.choice(
        ["low", "middle", "high"],
        size=n_households,
        p=[0.32, 0.50, 0.18],
    )

    # Synthetic consent state: households in low-broadband or high-underserved
    # contexts may have fewer digital traces and more missing history.
    consent_probability = (0.78 - 0.20 * underserved + 0.08 * broadband).clip(0.35, 0.95)
    consent_behavioral = rng.binomial(1, consent_probability).astype(bool)

    internet_access_probability = (0.55 + 0.40 * broadband - 0.15 * underserved).clip(
        0.20, 0.98
    )
    has_internet_access = rng.binomial(1, internet_access_probability).astype(bool)

    language_access_need = rng.binomial(1, (0.05 + 0.18 * underserved).clip(0, 0.45)).astype(bool)
    disability_proxy = rng.binomial(1, 0.08 + 0.10 * (age_group == "senior")).astype(bool)

    # Sensitive cohort is used only for policy-rule simulation.
    sensitive_cohort = (age_group == "youth") | disability_proxy

    # Low-signal households have fewer historical events.
    low_signal_probability = (
        0.15
        + 0.35 * underserved
        + 0.20 * (~has_internet_access)
        + 0.15 * (~consent_behavioral)
    ).clip(0, 0.90)
    low_signal = rng.binomial(1, low_signal_probability).astype(bool)

    historical_engagement_count = rng.poisson(
        lam=np.where(low_signal, 1.2, 8.0)
    ).clip(0, 50)

    employment_need = rng.binomial(
        1,
        (0.10 + 1.8 * unemployment + 0.15 * (income_band == "low")).clip(0, 0.80),
    ).astype(bool)

    return pd.DataFrame(
        {
            "household_id": household_ids,
            "community_id": community_ids,
            "age_group": age_group,
            "income_band": income_band,
            "consent_behavioral": consent_behavioral,
            "has_internet_access": has_internet_access,
            "language_access_need": language_access_need,
            "disability_proxy": disability_proxy,
            "sensitive_cohort": sensitive_cohort,
            "low_signal": low_signal,
            "historical_engagement_count": historical_engagement_count,
            "employment_need": employment_need,
        }
    )


def generate_services(rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic public services."""
    rows = []
    for i, category in enumerate(SERVICE_CATEGORIES):
        rows.append(
            {
                "service_id": f"S{i:03d}",
                "service_category": category,
                "provider_type": rng.choice(["public_agency", "nonprofit", "clinic", "school_partner"]),
                "monthly_capacity": int(rng.integers(500, 5000)),
                "requires_individual_history": category in {"preventive_health", "job_training"},
                "sensitive_service": category in {"preventive_health", "housing_support"},
            }
        )
    return pd.DataFrame(rows)


def _service_need_score(
    merged: pd.DataFrame,
    service_category: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetic latent relevance score by service category.

    Behavioral engagement is intentionally important in this synthetic benchmark
    so that signal-loss scenarios meaningfully affect ranking utility. Privacy-safe
    aggregate features are then expected to recover part of the lost signal.
    """
    income_low = (merged["income_band"] == "low").astype(float).to_numpy()
    senior = (merged["age_group"] == "senior").astype(float).to_numpy()
    youth = (merged["age_group"] == "youth").astype(float).to_numpy()
    employment_need = merged["employment_need"].astype(float).to_numpy()
    language_need = merged["language_access_need"].astype(float).to_numpy()
    disability = merged["disability_proxy"].astype(float).to_numpy()

    food_risk = merged["food_access_risk"].to_numpy()
    health_need = merged["health_need_score"].to_numpy()
    housing_pressure = merged["housing_pressure"].to_numpy()
    underserved = merged["underserved_score"].to_numpy()

    # Individual-level behavioral signal. Signal-loss simulations remove or suppress
    # this feature, creating the privacy-utility tradeoff studied by the project.
    engagement = np.log1p(merged["historical_engagement_count"].to_numpy()) / np.log(51)

    noise = rng.normal(0, 0.25, len(merged))

    if service_category == "food_assistance":
        score = (
            1.0 * income_low
            + 0.9 * food_risk
            + 0.5 * underserved
            + 0.8 * engagement
            + 0.2 * language_need
        )
    elif service_category == "preventive_health":
        score = (
            0.9 * health_need
            + 0.5 * senior
            + 0.3 * disability
            + 1.2 * engagement
        )
    elif service_category == "housing_support":
        score = (
            1.0 * housing_pressure
            + 0.8 * income_low
            + 0.4 * underserved
            + 0.7 * engagement
        )
    elif service_category == "job_training":
        score = (
            1.0 * employment_need
            + 0.6 * income_low
            + 1.4 * engagement
        )
    elif service_category == "education_support":
        score = (
            0.8 * youth
            + 0.6 * language_need
            + 0.4 * underserved
            + 0.8 * engagement
        )
    elif service_category == "transportation_support":
        score = (
            0.7 * senior
            + 0.7 * disability
            + 0.5 * underserved
            + 0.4 * food_risk
            + 0.7 * engagement
        )
    else:
        score = np.zeros(len(merged))

    return score + noise


def generate_outreach_events(
    households: pd.DataFrame,
    communities: pd.DataFrame,
    services: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate household-service candidate pairs and synthetic relevance labels."""
    households_with_context = households.merge(communities, on="community_id", how="left")

    all_rows = []
    for _, service in services.iterrows():
        merged = households_with_context.copy()
        merged["service_id"] = service["service_id"]
        merged["service_category"] = service["service_category"]

        latent_score = _service_need_score(merged, service["service_category"], rng)

        # Create a service-specific behavioral signal. Unlike household-level engagement,
        # this can change the relative ranking of services within the same household.
        service_affinity = sigmoid(latent_score - 0.6 + rng.normal(0, 0.6, len(merged)))
        base_rate = np.where(merged["low_signal"].astype(bool).to_numpy(), 0.8, 3.8)
        historical_service_engagement_count = rng.poisson(
            lam=base_rate * (0.25 + 2.5 * service_affinity)
        ).clip(0, 50)

        service_engagement = np.log1p(historical_service_engagement_count) / np.log(51)

        # Low-signal households are intentionally harder to rank when raw behavioral
        # signals are removed. This creates the fairness/utility problem the project studies.
        low_signal_penalty = -0.35 * merged["low_signal"].astype(float).to_numpy()

        relevance_probability = sigmoid(
            -1.7
            + 0.45 * latent_score
            + 1.8 * service_engagement
            + low_signal_penalty
        )
        relevant = rng.binomial(1, relevance_probability).astype(int)

        # Outreach success is a noisier downstream proxy.
        success_probability = (0.15 + 0.65 * relevance_probability).clip(0, 0.95)
        outreach_success = rng.binomial(1, success_probability).astype(int)

        event = merged[
            [
                "household_id",
                "community_id",
                "service_id",
                "service_category",
                "age_group",
                "income_band",
                "consent_behavioral",
                "has_internet_access",
                "language_access_need",
                "disability_proxy",
                "sensitive_cohort",
                "low_signal",
                "historical_engagement_count",
                "employment_need",
                "urbanicity",
                "median_income",
                "unemployment_rate",
                "broadband_access",
                "food_access_risk",
                "health_need_score",
                "housing_pressure",
                "underserved_score",
            ]
        ].copy()

        event["historical_service_engagement_count"] = historical_service_engagement_count
        event["latent_relevance_score"] = latent_score.round(4)
        event["relevance_probability"] = relevance_probability.round(4)
        event["relevant"] = relevant
        event["outreach_success"] = outreach_success
        all_rows.append(event)

    events = pd.concat(all_rows, ignore_index=True)
    events["event_id"] = [f"E{i:08d}" for i in range(len(events))]

    return events[
        ["event_id"]
        + [col for col in events.columns if col != "event_id"]
    ]


def generate_all(
    n_communities: int = 120,
    n_households: int = 10000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    communities = generate_communities(n_communities=n_communities, rng=rng)
    households = generate_households(
        n_households=n_households,
        communities=communities,
        rng=rng,
    )
    services = generate_services(rng=rng)
    events = generate_outreach_events(
        households=households,
        communities=communities,
        services=services,
        rng=rng,
    )

    return communities, households, services, events


def write_outputs(
    out_dir: Path,
    communities: pd.DataFrame,
    households: pd.DataFrame,
    services: pd.DataFrame,
    events: pd.DataFrame,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "communities": out_dir / "synthetic_communities.csv",
        "households": out_dir / "synthetic_households.csv",
        "services": out_dir / "synthetic_services.csv",
        "events": out_dir / "synthetic_outreach_events.csv",
    }

    communities.to_csv(outputs["communities"], index=False)
    households.to_csv(outputs["households"], index=False)
    services.to_csv(outputs["services"], index=False)
    events.to_csv(outputs["events"], index=False)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--n-communities", type=int, default=120)
    parser.add_argument("--n-households", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    communities, households, services, events = generate_all(
        n_communities=args.n_communities,
        n_households=args.n_households,
        seed=args.seed,
    )

    outputs = write_outputs(
        out_dir=args.out,
        communities=communities,
        households=households,
        services=services,
        events=events,
    )

    print("Generated synthetic public-service outreach data:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")

    print("\nShapes:")
    print(f"- communities: {communities.shape}")
    print(f"- households: {households.shape}")
    print(f"- services: {services.shape}")
    print(f"- events: {events.shape}")


if __name__ == "__main__":
    main()
