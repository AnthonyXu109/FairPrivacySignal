from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    data_dir = Path("data/synthetic")
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    communities = pd.read_csv(data_dir / "synthetic_communities.csv")
    households = pd.read_csv(data_dir / "synthetic_households.csv")
    events = pd.read_csv(data_dir / "synthetic_outreach_events.csv")

    # 1. Distribution of underserved score
    plt.figure(figsize=(7, 4))
    plt.hist(communities["underserved_score"], bins=20)
    plt.xlabel("Underserved score")
    plt.ylabel("Number of communities")
    plt.title("Synthetic communities vary in underserved status")
    plt.tight_layout()
    plt.savefig(out_dir / "underserved_score_distribution.png", dpi=160)
    plt.close()

    # 2. Low-signal share by underserved score bucket
    households_with_context = households.merge(
        communities[["community_id", "underserved_score"]],
        on="community_id",
        how="left",
    )
    households_with_context["underserved_bucket"] = pd.qcut(
        households_with_context["underserved_score"],
        q=4,
        labels=["lowest", "low", "high", "highest"],
    )

    low_signal_by_bucket = (
        households_with_context
        .groupby("underserved_bucket", observed=False)["low_signal"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(7, 4))
    plt.bar(
        low_signal_by_bucket["underserved_bucket"].astype(str),
        low_signal_by_bucket["low_signal"],
    )
    plt.xlabel("Community underserved score bucket")
    plt.ylabel("Share of low-signal households")
    plt.title("Low-signal households concentrate in underserved communities")
    plt.tight_layout()
    plt.savefig(out_dir / "low_signal_by_underserved_bucket.png", dpi=160)
    plt.close()

    # 3. Relevance rate by service category and low-signal status
    relevance = (
        events
        .groupby(["service_category", "low_signal"], observed=False)["relevant"]
        .mean()
        .reset_index()
    )

    pivot = relevance.pivot(
        index="service_category",
        columns="low_signal",
        values="relevant",
    ).rename(columns={False: "not_low_signal", True: "low_signal"})

    pivot.plot(kind="bar", figsize=(9, 4))
    plt.xlabel("Service category")
    plt.ylabel("Synthetic relevance rate")
    plt.title("Service relevance differs by signal availability")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "relevance_by_service_and_signal_status.png", dpi=160)
    plt.close()

    print("Wrote sanity check figures to docs/assets:")
    for path in sorted(out_dir.glob("*.png")):
        print(f"- {path}")


if __name__ == "__main__":
    main()
