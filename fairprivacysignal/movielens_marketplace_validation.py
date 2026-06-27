from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

from fairprivacysignal.public_data_visuals import write_recovery_profile_svg


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data/raw/movielens")
TABLE_DIR = Path("outputs/tables")
ASSET_DIR = Path("docs/assets")
REPORT_PATH = Path("docs/movielens_marketplace_validation.md")
FIGURE_PATH = ASSET_DIR / "movielens_marketplace_validation.svg"
PROFILE_FIGURE_PATH = ASSET_DIR / "movielens_marketplace_recovery_profile.svg"

K = 10


def ensure_movielens_data(data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "ml-latest-small"
    ratings_path = extracted / "ratings.csv"
    if ratings_path.exists():
        return extracted

    archive = data_dir / "ml-latest-small.zip"
    if not archive.exists():
        print(f"Downloading MovieLens latest-small from {DATA_URL}")
        urlretrieve(DATA_URL, archive)

    with ZipFile(archive) as zip_file:
        zip_file.extractall(data_dir)
    return extracted


def split_genres(value: str) -> List[str]:
    if not isinstance(value, str) or value == "(no genres listed)":
        return ["unknown"]
    return [genre.strip() for genre in value.split("|") if genre.strip()]


def load_movielens_frames(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = ensure_movielens_data(data_dir)
    ratings = pd.read_csv(root / "ratings.csv")
    movies = pd.read_csv(root / "movies.csv")
    movies["genre_list"] = movies["genres"].map(split_genres)
    return ratings, movies[["movieId", "title", "genre_list"]]


def split_user_history(ratings: pd.DataFrame, test_fraction: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for _, group in ratings.sort_values(["userId", "timestamp"]).groupby("userId"):
        n_rows = len(group)
        n_test = max(1, int(round(n_rows * test_fraction)))
        n_test = min(n_test, max(1, n_rows - 5))
        train_parts.append(group.iloc[: n_rows - n_test])
        test_parts.append(group.iloc[n_rows - n_test :])

    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def explode_genres(frame: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(movies, on="movieId", how="left").explode("genre_list")


def _mean_lookup(series: pd.Series) -> Dict[object, float]:
    return {key: float(value) for key, value in series.items()}


def _multi_mean_lookup(series: pd.Series) -> Dict[Tuple[object, object], float]:
    return {(key[0], key[1]): float(value) for key, value in series.items()}


def mean_for_genres(
    genres: Iterable[str],
    lookup: Dict[object, float],
    fallback: float,
) -> float:
    values = [lookup.get(genre, np.nan) for genre in genres]
    values = [value for value in values if not pd.isna(value)]
    return float(np.mean(values)) if values else float(fallback)


def user_mean_for_genres(
    user_id: int,
    genres: Iterable[str],
    lookup: Dict[Tuple[object, object], float],
    fallback_lookup: Dict[object, float],
    fallback: float,
) -> float:
    values = [lookup.get((user_id, genre), np.nan) for genre in genres]
    values = [value for value in values if not pd.isna(value)]
    if values:
        return float(np.mean(values))
    return mean_for_genres(genres, fallback_lookup, fallback)


def cohort_mean_for_genres(
    low_signal: bool,
    genres: Iterable[str],
    lookup: Dict[Tuple[object, object], float],
    fallback_lookup: Dict[object, float],
    fallback: float,
) -> float:
    values = [lookup.get((low_signal, genre), np.nan) for genre in genres]
    values = [value for value in values if not pd.isna(value)]
    if values:
        return float(np.mean(values))
    return mean_for_genres(genres, fallback_lookup, fallback)


def average_ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = K) -> float:
    values = []
    for _, group in df.groupby("userId"):
        ranked = group.sort_values(score_col, ascending=False)
        gains = ranked["relevant"].to_numpy(dtype=float)[:k]
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))

        ideal = np.sort(group["relevant"].to_numpy(dtype=float))[::-1][:k]
        ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        idcg = float(np.sum(ideal * ideal_discounts))
        if idcg > 0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else float("nan")


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    truth = y_true.to_numpy(dtype=int)
    scores = y_score.to_numpy(dtype=float)
    positives = truth == 1
    negatives = truth == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[positives].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def build_candidate_events(
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    negatives_per_user: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_movies = np.array(sorted(movies["movieId"].unique()))
    seen_by_user = ratings_by_user(train, test)

    candidate_parts = []
    holdout = test.copy()
    holdout["candidate_source"] = "heldout_rating"
    holdout["relevant"] = (holdout["rating"] >= 4.0).astype(int)
    candidate_parts.append(holdout)

    negative_rows = []
    for user_id, seen_movies in seen_by_user.items():
        available = np.setdiff1d(all_movies, np.array(list(seen_movies)), assume_unique=False)
        if len(available) == 0:
            continue
        sample_size = min(negatives_per_user, len(available))
        sampled = rng.choice(available, size=sample_size, replace=False)
        negative_rows.extend(
            {
                "userId": user_id,
                "movieId": int(movie_id),
                "rating": np.nan,
                "timestamp": np.nan,
                "candidate_source": "sampled_unrated_negative",
                "relevant": 0,
            }
            for movie_id in sampled
        )

    candidate_parts.append(pd.DataFrame(negative_rows))
    candidates = pd.concat(candidate_parts, ignore_index=True)
    return candidates.merge(movies, on="movieId", how="left")


def ratings_by_user(train: pd.DataFrame, test: pd.DataFrame) -> Dict[int, set[int]]:
    combined = pd.concat([train[["userId", "movieId"]], test[["userId", "movieId"]]])
    return {
        int(user_id): set(group["movieId"].astype(int))
        for user_id, group in combined.groupby("userId")
    }


def prepare_scored_events(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    train, test = split_user_history(ratings)
    global_mean = float(train["rating"].mean())

    user_counts = train.groupby("userId").size().rename("train_rating_count")
    low_signal_cutoff = float(user_counts.quantile(0.25))
    low_signal_users = set(user_counts[user_counts <= low_signal_cutoff].index)

    movie_stats = train.groupby("movieId")["rating"].agg(["mean", "count"])
    movie_mean = _mean_lookup(movie_stats["mean"])
    movie_count = _mean_lookup(movie_stats["count"])
    max_log_count = max(np.log1p(list(movie_count.values()))) if movie_count else 1.0

    train_with_genres = explode_genres(train, movies)
    genre_mean = _mean_lookup(train_with_genres.groupby("genre_list")["rating"].mean())
    user_genre_mean = _multi_mean_lookup(
        train_with_genres.groupby(["userId", "genre_list"])["rating"].mean()
    )

    train_with_genres["low_signal"] = train_with_genres["userId"].isin(low_signal_users)
    cohort_genre_mean = _multi_mean_lookup(
        train_with_genres.groupby(["low_signal", "genre_list"])["rating"].mean()
    )

    scored = build_candidate_events(train, test, movies)
    scored["train_rating_count"] = scored["userId"].map(user_counts).fillna(0).astype(int)
    scored["low_signal"] = scored["userId"].isin(low_signal_users)
    scored["movie_mean_rating"] = scored["movieId"].map(movie_mean).fillna(global_mean)
    scored["movie_popularity_component"] = scored["movieId"].map(
        lambda movie_id: global_mean
        + 0.30 * ((np.log1p(movie_count.get(movie_id, 0.0)) / max_log_count) - 0.5)
    )
    scored["genre_global_rating"] = scored["genre_list"].map(
        lambda genres: mean_for_genres(genres, genre_mean, global_mean)
    )
    scored["user_genre_rating"] = scored.apply(
        lambda row: user_mean_for_genres(
            int(row["userId"]),
            row["genre_list"],
            user_genre_mean,
            genre_mean,
            global_mean,
        ),
        axis=1,
    )
    scored["cohort_genre_rating"] = scored.apply(
        lambda row: cohort_mean_for_genres(
            bool(row["low_signal"]),
            row["genre_list"],
            cohort_genre_mean,
            genre_mean,
            global_mean,
        ),
        axis=1,
    )

    scored["full_signal_score"] = (
        0.60 * scored["user_genre_rating"]
        + 0.25 * scored["movie_mean_rating"]
        + 0.15 * scored["movie_popularity_component"]
    )
    scored["no_history_score"] = (
        0.65 * scored["movie_mean_rating"]
        + 0.25 * scored["genre_global_rating"]
        + 0.10 * scored["movie_popularity_component"]
    )
    scored["cohort_recovery_score"] = (
        0.55 * scored["cohort_genre_rating"]
        + 0.30 * scored["movie_mean_rating"]
        + 0.15 * scored["movie_popularity_component"]
    )
    scored["policy_aware_score"] = np.where(
        scored["low_signal"],
        scored["cohort_recovery_score"],
        scored["full_signal_score"],
    )
    return scored


def summarize_method(
    scored: pd.DataFrame,
    method: str,
    score_col: str,
    individual_history_exposure: float,
) -> Dict[str, float | str]:
    low = scored[scored["low_signal"]]
    not_low = scored[~scored["low_signal"]]
    return {
        "sector": "marketplace",
        "dataset": "MovieLens latest-small",
        "method": method,
        "overall_auc": safe_auc(scored["relevant"], scored[score_col]),
        "overall_ndcg_at_10": average_ndcg_at_k(scored, score_col),
        "low_signal_ndcg_at_10": average_ndcg_at_k(low, score_col),
        "not_low_signal_ndcg_at_10": average_ndcg_at_k(not_low, score_col),
        "individual_history_exposure": individual_history_exposure,
        "num_users": int(scored["userId"].nunique()),
        "num_candidate_events": int(len(scored)),
        "low_signal_user_share": float(scored.groupby("userId")["low_signal"].first().mean()),
    }


def summarize_results(scored: pd.DataFrame) -> pd.DataFrame:
    policy_exposure = float((~scored["low_signal"]).mean())
    rows = [
        summarize_method(scored, "Full signal oracle", "full_signal_score", 1.0),
        summarize_method(scored, "No user-history baseline", "no_history_score", 0.0),
        summarize_method(scored, "Cohort aggregate recovery", "cohort_recovery_score", 0.0),
        summarize_method(scored, "Policy-aware partial recovery", "policy_aware_score", policy_exposure),
    ]
    summary = pd.DataFrame(rows)

    full = float(summary.loc[summary["method"] == "Full signal oracle", "overall_ndcg_at_10"].iloc[0])
    loss = float(summary.loc[summary["method"] == "No user-history baseline", "overall_ndcg_at_10"].iloc[0])
    denominator = full - loss
    summary["full_signal_gap_closed"] = np.nan
    if denominator > 0:
        summary["full_signal_gap_closed"] = (
            summary["overall_ndcg_at_10"] - loss
        ) / denominator
    summary.loc[summary["method"].eq("Full signal oracle"), "full_signal_gap_closed"] = 1.0
    summary.loc[summary["method"].eq("No user-history baseline"), "full_signal_gap_closed"] = 0.0
    return summary


def write_svg(summary: pd.DataFrame, path: Path = FIGURE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = summary["method"].tolist()
    overall = summary["overall_ndcg_at_10"].tolist()
    low_signal = summary["low_signal_ndcg_at_10"].tolist()
    max_value = max(overall + low_signal + [0.01])

    width = 940
    height = 550
    left = 92
    top = 70
    plot_width = 760
    plot_height = 310
    group_width = plot_width / len(methods)
    bar_width = 34

    def y_pos(value: float) -> float:
        return top + plot_height - (value / max_value) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">MovieLens public-data marketplace validation</text>',
        '<text x="40" y="60" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Ranking held-out ratings under user-history signal loss; higher NDCG@10 is better.</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]

    for tick in np.linspace(0.0, max_value, 5):
        y = y_pos(float(tick))
        lines.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#4b5563">{tick:.2f}</text>')

    colors = {"overall": "#2563eb", "low": "#f97316"}
    for idx, method in enumerate(methods):
        center = left + group_width * idx + group_width / 2
        for offset, value, key in [
            (-bar_width / 2 - 4, overall[idx], "overall"),
            (bar_width / 2 + 4, low_signal[idx], "low"),
        ]:
            x = center + offset - bar_width / 2
            y = y_pos(float(value))
            h = top + plot_height - y
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" rx="3" fill="{colors[key]}"/>')
            lines.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#111827">{value:.3f}</text>')
        label = method.replace(" ", "\n")
        for line_idx, label_line in enumerate(label.split("\n")):
            lines.append(f'<text x="{center:.1f}" y="{top + plot_height + 22 + 14 * line_idx}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{label_line}</text>')

    legend_y = top + plot_height + 102
    lines.extend(
        [
            f'<rect x="{left}" y="{legend_y}" width="14" height="14" fill="{colors["overall"]}" rx="2"/>',
            f'<text x="{left + 22}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Overall users</text>',
            f'<rect x="{left + 140}" y="{legend_y}" width="14" height="14" fill="{colors["low"]}" rx="2"/>',
            f'<text x="{left + 162}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Low-signal users</text>',
            f'<text x="{left}" y="{legend_y + 42}" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">Raw MovieLens data is downloaded at runtime and is not redistributed in this repository.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{100.0 * value:.1f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def write_report(summary: pd.DataFrame, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    display = summary.copy()
    for column in [
        "overall_auc",
        "overall_ndcg_at_10",
        "low_signal_ndcg_at_10",
        "not_low_signal_ndcg_at_10",
        "individual_history_exposure",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    display["full_signal_gap_closed"] = summary["full_signal_gap_closed"].map(format_percent)

    table = markdown_table(
        display[
        [
            "method",
            "overall_ndcg_at_10",
            "low_signal_ndcg_at_10",
            "full_signal_gap_closed",
            "individual_history_exposure",
        ]
        ]
    )

    cohort = summary[summary["method"] == "Cohort aggregate recovery"].iloc[0]
    policy = summary[summary["method"] == "Policy-aware partial recovery"].iloc[0]

    content = f"""# MovieLens Marketplace Validation

This public-data pilot adapts the FairPrivacySignal signal-loss pattern to a
marketplace recommendation setting using MovieLens latest-small. Users rank held-out
movies against sampled unrated candidate movies; their prior rating history is treated
as the behavioral signal that may be unavailable under privacy, retention, or consent
constraints.

The raw MovieLens files are downloaded at runtime from GroupLens and are not
redistributed in this repository. The pilot uses only public ratings and movie genres.

![MovieLens public-data marketplace validation](assets/movielens_marketplace_validation.svg)

![MovieLens recovery profile](assets/movielens_marketplace_recovery_profile.svg)

## Task

- **Ranked candidate:** held-out rated movies plus sampled unrated candidate movies
- **Restricted historical signal:** user-specific genre preference inferred from prior ratings
- **Permitted context:** movie-level rating aggregates, movie popularity, and genre-level aggregates
- **Low-signal group:** users in the bottom quartile of training-history volume
- **Metric:** NDCG@10, with binary relevance defined as held-out rating >= 4

## Results

{table}

The aggregate-only recovery path closes {format_percent(float(cohort["full_signal_gap_closed"]))}
of the full-signal NDCG@10 gap without using individual user-history features at
scoring time. The policy-aware partial path keeps user-history signal for higher-history
users and substitutes cohort aggregates for low-signal users, closing
{format_percent(float(policy["full_signal_gap_closed"]))} of the same gap in this pilot.

## Interpretation

This is an external public-data validation of the system shape, not a production
marketplace deployment. It shows that the repository's core pattern can be instantiated
outside the original synthetic public-service scenario: define a restricted behavioral
signal, suppress it at ranking time, substitute train-fitted aggregates, and measure
overall and low-signal ranking effects. Because MovieLens is a ratings dataset rather
than a privacy-policy log, the availability policy is simulated for evaluation. The
sampled unrated candidates are treated as implicit negatives, a standard recommender
evaluation shortcut but not proof that a user would dislike every sampled item.
"""
    path.write_text(content, encoding="utf-8")


def run_validation() -> pd.DataFrame:
    ratings, movies = load_movielens_frames()
    scored = prepare_scored_events(ratings, movies)
    summary = summarize_results(scored)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(TABLE_DIR / "movielens_marketplace_scored_events.csv", index=False)
    summary.to_csv(TABLE_DIR / "movielens_marketplace_validation_summary.csv", index=False)
    write_svg(summary)
    write_recovery_profile_svg(
        summary,
        PROFILE_FIGURE_PATH,
        title="MovieLens Marketplace Recovery Profile",
        subtitle="Held-out movie ranking under user-history signal loss.",
        metric_col="overall_ndcg_at_10",
        low_signal_col="low_signal_ndcg_at_10",
        exposure_col="individual_history_exposure",
        low_signal_label="Low-signal NDCG@10",
    )
    write_report(summary)
    return summary


def main() -> None:
    summary = run_validation()
    print("MovieLens marketplace validation:")
    print(summary.round(4).to_string(index=False))
    print(f"\nWrote: {REPORT_PATH}")
    print(f"Wrote: {FIGURE_PATH}")
    print(f"Wrote: {PROFILE_FIGURE_PATH}")


if __name__ == "__main__":
    main()
