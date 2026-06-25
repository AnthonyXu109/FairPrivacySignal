from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
DATA_DIR = Path("data/raw/education")
TABLE_DIR = Path("outputs/tables")
ASSET_DIR = Path("docs/assets")
REPORT_PATH = Path("docs/education_student_performance_validation.md")
FIGURE_PATH = ASSET_DIR / "education_student_performance_validation.svg"

K = 50
RECONSTRUCTION_FEATURES = [
    "subject",
    "school",
    "studytime",
    "failures",
    "absences",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "higher",
    "internet",
    "goout",
    "health",
]


def ensure_student_data(data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "student"
    if (extracted / "student-mat.csv").exists() and (extracted / "student-por.csv").exists():
        return extracted

    archive = data_dir / "student.zip"
    if not archive.exists():
        print(f"Downloading UCI Student Performance data from {DATA_URL}")
        urlretrieve(DATA_URL, archive)

    extracted.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        zip_file.extractall(extracted)
    return extracted


def load_student_frame(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    root = ensure_student_data(data_dir)
    frames = []
    for filename, subject in [("student-mat.csv", "math"), ("student-por.csv", "portuguese")]:
        frame = pd.read_csv(root / filename, sep=";")
        frame["subject"] = subject
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["record_id"] = np.arange(len(data))
    return data


def split_train_test(data: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for _, group in data.groupby(["subject", "school"]):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        split_at = max(1, int(round(len(shuffled) * 0.70)))
        split_at = min(split_at, len(shuffled) - 1)
        train_parts.append(shuffled.iloc[:split_at])
        test_parts.append(shuffled.iloc[split_at:])
    return pd.concat(train_parts), pd.concat(test_parts)


def minmax(series: pd.Series) -> pd.Series:
    low = float(series.min())
    high = float(series.max())
    if high == low:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - low) / (high - low)


def design_matrices(
    train: pd.DataFrame,
    apply_to: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    combined = pd.concat(
        [train[feature_columns], apply_to[feature_columns]],
        keys=["train", "apply"],
    )
    categorical = [
        column for column in feature_columns if combined[column].dtype == "object"
    ]
    encoded = pd.get_dummies(combined, columns=categorical, dtype=float).fillna(0.0)
    train_matrix = encoded.xs("train").to_numpy(dtype=float)
    apply_matrix = encoded.xs("apply").to_numpy(dtype=float)

    means = train_matrix.mean(axis=0)
    stds = train_matrix.std(axis=0)
    stds[stds == 0] = 1.0
    train_matrix = (train_matrix - means) / stds
    apply_matrix = (apply_matrix - means) / stds
    return (
        np.column_stack([np.ones(len(train_matrix)), train_matrix]),
        np.column_stack([np.ones(len(apply_matrix)), apply_matrix]),
    )


def reconstruct_prior_grade_risk(
    train: pd.DataFrame,
    apply_to: pd.DataFrame,
    alpha: float = 10.0,
) -> np.ndarray:
    train_matrix, apply_matrix = design_matrices(
        train, apply_to, RECONSTRUCTION_FEATURES
    )
    target = 1.0 - (train[["G1", "G2"]].mean(axis=1).to_numpy(dtype=float) / 20.0)
    penalty = np.eye(train_matrix.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = (
        np.linalg.pinv(train_matrix.T @ train_matrix + penalty)
        @ train_matrix.T
        @ target
    )
    return np.clip(apply_matrix @ coefficients, 0.0, 1.0)


def cohort_lookup(train: pd.DataFrame) -> dict[tuple[object, ...], float]:
    reference = train.copy()
    reference["prior_grade_risk"] = 1.0 - (reference[["G1", "G2"]].mean(axis=1) / 20.0)
    reference["studytime_bucket"] = reference["studytime"].clip(upper=3)
    grouped = reference.groupby(
        ["subject", "school", "studytime_bucket", "failures"]
    )["prior_grade_risk"].mean()
    return {tuple(key): float(value) for key, value in grouped.items()}


def fallback_lookup(train: pd.DataFrame) -> dict[tuple[object, ...], float]:
    reference = train.copy()
    reference["prior_grade_risk"] = 1.0 - (reference[["G1", "G2"]].mean(axis=1) / 20.0)
    grouped = reference.groupby(["subject", "school"])["prior_grade_risk"].mean()
    return {tuple(key): float(value) for key, value in grouped.items()}


def score_events(data: pd.DataFrame) -> pd.DataFrame:
    train, test = split_train_test(data)
    cohort = cohort_lookup(train)
    fallback = fallback_lookup(train)
    global_prior_risk = float(1.0 - (train[["G1", "G2"]].mean(axis=1) / 20.0).mean())

    scored = test.copy()
    scored["studytime_bucket"] = scored["studytime"].clip(upper=3)
    scored["needs_support"] = (scored["G3"] < 10).astype(int)
    scored["ranking_group"] = scored["subject"] + "_" + scored["school"]
    scored["low_signal"] = scored["absences"] <= scored["absences"].median()

    scored["prior_grade_risk"] = 1.0 - (scored[["G1", "G2"]].mean(axis=1) / 20.0)
    scored["absence_risk"] = minmax(scored["absences"].clip(upper=30))
    scored["failure_risk"] = scored["failures"].clip(upper=3) / 3.0
    scored["studytime_risk"] = 1.0 - ((scored["studytime"] - 1) / 3.0)
    scored["family_context_risk"] = 1.0 - (
        scored[["Medu", "Fedu"]].mean(axis=1) / 4.0
    )

    scored["cohort_prior_risk"] = scored.apply(
        lambda row: cohort.get(
            (
                row["subject"],
                row["school"],
                row["studytime_bucket"],
                row["failures"],
            ),
            fallback.get((row["subject"], row["school"]), global_prior_risk),
        ),
        axis=1,
    )
    scored["reconstructed_prior_risk"] = reconstruct_prior_grade_risk(train, scored)

    scored["full_signal_score"] = (
        0.55 * scored["prior_grade_risk"]
        + 0.225 * scored["failure_risk"]
        + 0.225 * scored["absence_risk"]
    )
    scored["no_prior_score"] = (
        0.35 * scored["failure_risk"]
        + 0.25 * scored["absence_risk"]
        + 0.20 * scored["studytime_risk"]
        + 0.20 * scored["family_context_risk"]
    )
    scored["signal_recovery_score"] = (
        0.50 * scored["reconstructed_prior_risk"]
        + 0.05 * scored["cohort_prior_risk"]
        + 0.225 * scored["failure_risk"]
        + 0.125 * scored["absence_risk"]
        + 0.10 * scored["studytime_risk"]
    )
    scored["policy_aware_score"] = np.where(
        scored["low_signal"],
        scored["signal_recovery_score"],
        scored["full_signal_score"],
    )
    return scored


def average_ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = K) -> float:
    values = []
    for _, group in df.groupby("ranking_group"):
        ranked = group.sort_values(score_col, ascending=False)
        gains = ranked["needs_support"].to_numpy(dtype=float)[:k]
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))

        ideal = np.sort(group["needs_support"].to_numpy(dtype=float))[::-1][:k]
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


def summarize_method(
    scored: pd.DataFrame,
    method: str,
    score_col: str,
    prior_grade_exposure: float,
) -> dict[str, float | str]:
    low = scored[scored["low_signal"]]
    not_low = scored[~scored["low_signal"]]
    return {
        "sector": "education",
        "dataset": "UCI Student Performance",
        "method": method,
        "overall_auc": safe_auc(scored["needs_support"], scored[score_col]),
        "overall_ndcg_at_50": average_ndcg_at_k(scored, score_col),
        "low_signal_ndcg_at_50": average_ndcg_at_k(low, score_col),
        "not_low_signal_ndcg_at_50": average_ndcg_at_k(not_low, score_col),
        "prior_grade_exposure": prior_grade_exposure,
        "num_student_records": int(len(scored)),
        "low_signal_record_share": float(scored["low_signal"].mean()),
    }


def summarize_results(scored: pd.DataFrame) -> pd.DataFrame:
    policy_exposure = float((~scored["low_signal"]).mean())
    rows = [
        summarize_method(scored, "Full prior-grade signal", "full_signal_score", 1.0),
        summarize_method(scored, "No prior-grade baseline", "no_prior_score", 0.0),
        summarize_method(scored, "Train-fitted signal recovery", "signal_recovery_score", 0.0),
        summarize_method(scored, "Policy-aware partial recovery", "policy_aware_score", policy_exposure),
    ]
    summary = pd.DataFrame(rows)
    full = float(summary.loc[summary["method"] == "Full prior-grade signal", "overall_ndcg_at_50"].iloc[0])
    loss = float(summary.loc[summary["method"] == "No prior-grade baseline", "overall_ndcg_at_50"].iloc[0])
    denominator = full - loss
    summary["full_signal_gap_closed"] = np.nan
    if denominator > 0:
        summary["full_signal_gap_closed"] = (
            summary["overall_ndcg_at_50"] - loss
        ) / denominator
    summary.loc[summary["method"].eq("Full prior-grade signal"), "full_signal_gap_closed"] = 1.0
    summary.loc[summary["method"].eq("No prior-grade baseline"), "full_signal_gap_closed"] = 0.0
    return summary


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


def write_svg(summary: pd.DataFrame, path: Path = FIGURE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = summary["method"].tolist()
    overall = summary["overall_ndcg_at_50"].tolist()
    low_signal = summary["low_signal_ndcg_at_50"].tolist()
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
        '<text x="40" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Education public-data validation</text>',
        '<text x="40" y="60" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Ranking students for support when prior-grade signals are suppressed; higher NDCG@50 is better.</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]

    for tick in np.linspace(0.0, max_value, 5):
        y = y_pos(float(tick))
        lines.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#4b5563">{tick:.2f}</text>')

    colors = {"overall": "#0f766e", "low": "#7c3aed"}
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
        for line_idx, label_line in enumerate(method.replace(" ", "\n").split("\n")):
            lines.append(f'<text x="{center:.1f}" y="{top + plot_height + 22 + 14 * line_idx}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{label_line}</text>')

    legend_y = top + plot_height + 102
    lines.extend(
        [
            f'<rect x="{left}" y="{legend_y}" width="14" height="14" fill="{colors["overall"]}" rx="2"/>',
            f'<text x="{left + 22}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Overall student records</text>',
            f'<rect x="{left + 210}" y="{legend_y}" width="14" height="14" fill="{colors["low"]}" rx="2"/>',
            f'<text x="{left + 232}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Low-absence records</text>',
            f'<text x="{left}" y="{legend_y + 42}" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">Raw UCI data is downloaded at runtime and is not redistributed in this repository.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary: pd.DataFrame, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = summary.copy()
    for column in [
        "overall_auc",
        "overall_ndcg_at_50",
        "low_signal_ndcg_at_50",
        "not_low_signal_ndcg_at_50",
        "prior_grade_exposure",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    display["full_signal_gap_closed"] = summary["full_signal_gap_closed"].map(format_percent)

    table = markdown_table(
        display[
            [
                "method",
                "overall_ndcg_at_50",
                "low_signal_ndcg_at_50",
                "full_signal_gap_closed",
                "prior_grade_exposure",
            ]
        ]
    )

    recovery = summary[summary["method"] == "Train-fitted signal recovery"].iloc[0]
    policy = summary[summary["method"] == "Policy-aware partial recovery"].iloc[0]

    content = f"""# Education Student Performance Validation

This public-data pilot adapts the FairPrivacySignal signal-loss pattern to an
education support setting using the UCI Student Performance dataset. Student-course
records are ranked for support need; prior period grades are treated as historical
academic signals that may be unavailable under privacy, retention, or consent limits.

The raw UCI files are downloaded at runtime and are not redistributed in this repository.

![Education public-data validation](assets/education_student_performance_validation.svg)

## Task

- **Ranked candidate:** student-course records for support triage
- **Restricted historical signal:** prior grades `G1` and `G2`
- **Permitted context:** school, subject, study time, absences, prior failures, and family context
- **Low-signal group:** students with below-median absences, where obvious administrative warning signals are weaker
- **Metric:** NDCG@50, with binary relevance defined as final grade `G3 < 10`

## Results

{table}

The train-fitted recovery path closes {format_percent(float(recovery["full_signal_gap_closed"]))}
of the full-signal NDCG@50 gap without using individual prior-grade features at
scoring time. The policy-aware partial path keeps prior-grade signal for higher-signal
records and substitutes the recovered prior-grade risk for low-signal records, closing
{format_percent(float(policy["full_signal_gap_closed"]))} of the same gap in this pilot.

## Interpretation

This is an external public-data validation of the system shape, not a school deployment.
It shows how the method can be instantiated in an education-support workflow: define a
restricted historical academic signal, suppress it at scoring time, substitute a
train-fitted reconstruction with a cohort stabilizer, and measure support-ranking
recovery. The dataset is small and does not contain a real privacy-policy event, so
the availability policy is simulated for evaluation.
"""
    path.write_text(content, encoding="utf-8")


def run_validation() -> pd.DataFrame:
    data = load_student_frame()
    scored = score_events(data)
    summary = summarize_results(scored)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(TABLE_DIR / "education_student_performance_scored_events.csv", index=False)
    summary.to_csv(TABLE_DIR / "education_student_performance_validation_summary.csv", index=False)
    write_svg(summary)
    write_report(summary)
    return summary


def main() -> None:
    summary = run_validation()
    print("Education student performance validation:")
    print(summary.round(4).to_string(index=False))
    print(f"\nWrote: {REPORT_PATH}")
    print(f"Wrote: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
