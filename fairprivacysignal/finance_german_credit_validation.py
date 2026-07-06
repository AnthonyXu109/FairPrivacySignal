from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from fairprivacysignal.public_data_visuals import (
    write_gallery_card_svg,
    write_recovery_profile_svg,
)


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATA_DIR = Path("data/raw/finance_german")
TABLE_DIR = Path("outputs/tables")
ASSET_DIR = Path("docs/assets")
REPORT_PATH = Path("docs/finance_german_credit_validation.md")
FIGURE_PATH = ASSET_DIR / "finance_german_credit_validation.svg"
PROFILE_FIGURE_PATH = ASSET_DIR / "finance_german_credit_recovery_profile.svg"
GALLERY_FIGURE_PATH = ASSET_DIR / "finance_german_credit_gallery.svg"

K = 100
COLUMN_NAMES = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "class",
]
PERMITTED_RECONSTRUCTION_FEATURES = [
    "duration",
    "purpose",
    "credit_amount",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
]


def ensure_german_credit_data(data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "german.data"
    if not data_path.exists():
        print(f"Downloading UCI German Credit data from {DATA_URL}")
        urlretrieve(DATA_URL, data_path)
    return data_path


def load_credit_frame(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    data_path = ensure_german_credit_data(data_dir)
    frame = pd.read_csv(data_path, sep=" ", header=None, names=COLUMN_NAMES)
    frame["application_id"] = np.arange(len(frame))
    return frame


def split_train_test(
    data: pd.DataFrame,
    train_share: float = 0.70,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = data.sample(frac=1.0, random_state=seed)
    split_at = int(round(len(shuffled) * train_share))
    return shuffled.iloc[:split_at].copy(), shuffled.iloc[split_at:].copy()


def minmax(series: pd.Series) -> pd.Series:
    low = float(series.min())
    high = float(series.max())
    if high == low:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - low) / (high - low)


def add_signal_features(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["needs_review"] = (scored["class"] == 2).astype(int)
    scored["checking_risk"] = scored["checking_status"].map(
        {"A11": 1.0, "A12": 0.65, "A13": 0.35, "A14": 0.0}
    )
    scored["history_risk"] = scored["credit_history"].map(
        {"A30": 1.0, "A31": 0.80, "A32": 0.45, "A33": 0.25, "A34": 0.05}
    )
    scored["savings_risk"] = scored["savings_status"].map(
        {"A61": 1.0, "A62": 0.75, "A63": 0.45, "A64": 0.20, "A65": 0.50}
    )
    scored["restricted_history_signal"] = (
        0.45 * scored["checking_risk"]
        + 0.35 * scored["history_risk"]
        + 0.20 * scored["savings_risk"]
    )
    scored["amount_risk"] = minmax(scored["credit_amount"].clip(upper=12000))
    scored["duration_risk"] = minmax(scored["duration"].clip(upper=60))
    scored["age_risk"] = 1.0 - minmax(scored["age"].clip(lower=18, upper=75))
    scored["installment_risk"] = (scored["installment_rate"] - 1.0) / 3.0
    scored["thin_file"] = scored["existing_credits"] <= 1
    scored["context_score"] = (
        0.35 * scored["amount_risk"]
        + 0.25 * scored["duration_risk"]
        + 0.20 * scored["installment_risk"]
        + 0.20 * scored["age_risk"]
    )
    return scored


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


def reconstruct_history_signal(
    train: pd.DataFrame,
    apply_to: pd.DataFrame,
    alpha: float = 20.0,
) -> np.ndarray:
    train_matrix, apply_matrix = design_matrices(
        train, apply_to, PERMITTED_RECONSTRUCTION_FEATURES
    )
    target = train["restricted_history_signal"].to_numpy(dtype=float)
    penalty = np.eye(train_matrix.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = (
        np.linalg.pinv(train_matrix.T @ train_matrix + penalty)
        @ train_matrix.T
        @ target
    )
    return np.clip(apply_matrix @ coefficients, 0.0, 1.0)


def cohort_signal(train: pd.DataFrame, apply_to: pd.DataFrame) -> pd.Series:
    grouped = train.groupby(["purpose", "housing", "employment"])[
        "restricted_history_signal"
    ].mean()
    fallback = train.groupby("purpose")["restricted_history_signal"].mean()
    global_mean = float(train["restricted_history_signal"].mean())

    return apply_to.apply(
        lambda row: grouped.get(
            (row["purpose"], row["housing"], row["employment"]),
            fallback.get(row["purpose"], global_mean),
        ),
        axis=1,
    )


def score_applications(data: pd.DataFrame) -> pd.DataFrame:
    raw_train, raw_test = split_train_test(data)
    train = add_signal_features(raw_train)
    scored = add_signal_features(raw_test)

    scored["cohort_history_signal"] = cohort_signal(train, scored)
    scored["reconstructed_history_signal"] = reconstruct_history_signal(train, scored)
    recovered_signal = (
        0.90 * scored["reconstructed_history_signal"]
        + 0.10 * scored["cohort_history_signal"]
    )

    scored["full_signal_score"] = (
        scored["context_score"] + scored["restricted_history_signal"]
    )
    scored["no_history_score"] = scored["context_score"]
    scored["signal_recovery_score"] = scored["context_score"] + recovered_signal
    scored["policy_aware_score"] = np.where(
        scored["thin_file"],
        scored["signal_recovery_score"],
        scored["full_signal_score"],
    )
    return scored


def ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = K) -> float:
    ranked = df.sort_values(score_col, ascending=False)
    gains = ranked["needs_review"].to_numpy(dtype=float)[: min(k, len(ranked))]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(df["needs_review"].to_numpy(dtype=float))[::-1][: len(gains)]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else float("nan")


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
    history_exposure: float,
) -> dict[str, float | str]:
    thin_file = scored[scored["thin_file"]]
    not_thin_file = scored[~scored["thin_file"]]
    return {
        "sector": "financial_access",
        "dataset": "UCI German Credit",
        "method": method,
        "overall_auc": safe_auc(scored["needs_review"], scored[score_col]),
        "overall_ndcg_at_100": ndcg_at_k(scored, score_col),
        "thin_file_ndcg_at_100": ndcg_at_k(thin_file, score_col),
        "not_thin_file_ndcg_at_100": ndcg_at_k(not_thin_file, score_col),
        "history_signal_exposure": history_exposure,
        "num_applications": int(len(scored)),
        "thin_file_share": float(scored["thin_file"].mean()),
    }


def summarize_results(scored: pd.DataFrame) -> pd.DataFrame:
    policy_exposure = float((~scored["thin_file"]).mean())
    rows = [
        summarize_method(scored, "Full financial-history signal", "full_signal_score", 1.0),
        summarize_method(scored, "No history baseline", "no_history_score", 0.0),
        summarize_method(scored, "Train-fitted signal recovery", "signal_recovery_score", 0.0),
        summarize_method(scored, "Policy-aware partial recovery", "policy_aware_score", policy_exposure),
    ]
    summary = pd.DataFrame(rows)
    full = float(
        summary.loc[
            summary["method"] == "Full financial-history signal",
            "overall_ndcg_at_100",
        ].iloc[0]
    )
    loss = float(
        summary.loc[
            summary["method"] == "No history baseline",
            "overall_ndcg_at_100",
        ].iloc[0]
    )
    denominator = full - loss
    summary["full_signal_gap_closed"] = np.nan
    if denominator > 0:
        summary["full_signal_gap_closed"] = (
            summary["overall_ndcg_at_100"] - loss
        ) / denominator
    summary.loc[
        summary["method"].eq("Full financial-history signal"),
        "full_signal_gap_closed",
    ] = 1.0
    summary.loc[
        summary["method"].eq("No history baseline"),
        "full_signal_gap_closed",
    ] = 0.0
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
    overall = summary["overall_ndcg_at_100"].tolist()
    thin_file = summary["thin_file_ndcg_at_100"].tolist()
    max_value = max(overall + thin_file + [0.01])

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
        '<text x="40" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Financial-access public-data validation</text>',
        '<text x="40" y="60" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Ranking credit applications for review when financial-history signals are suppressed; higher NDCG@100 is better.</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]

    for tick in np.linspace(0.0, max_value, 5):
        y = y_pos(float(tick))
        lines.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#4b5563">{tick:.2f}</text>')

    colors = {"overall": "#2563eb", "thin": "#b45309"}
    for idx, method in enumerate(methods):
        center = left + group_width * idx + group_width / 2
        for offset, value, key in [
            (-bar_width / 2 - 4, overall[idx], "overall"),
            (bar_width / 2 + 4, thin_file[idx], "thin"),
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
            f'<text x="{left + 22}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Overall applications</text>',
            f'<rect x="{left + 210}" y="{legend_y}" width="14" height="14" fill="{colors["thin"]}" rx="2"/>',
            f'<text x="{left + 232}" y="{legend_y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#374151">Thin-file applications</text>',
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
        "overall_ndcg_at_100",
        "thin_file_ndcg_at_100",
        "not_thin_file_ndcg_at_100",
        "history_signal_exposure",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    display["full_signal_gap_closed"] = summary["full_signal_gap_closed"].map(format_percent)

    table = markdown_table(
        display[
            [
                "method",
                "overall_ndcg_at_100",
                "thin_file_ndcg_at_100",
                "full_signal_gap_closed",
                "history_signal_exposure",
            ]
        ]
    )

    recovery = summary[summary["method"] == "Train-fitted signal recovery"].iloc[0]
    policy = summary[summary["method"] == "Policy-aware partial recovery"].iloc[0]

    content = f"""# Finance German Credit Validation

This public-data pilot adapts the FairPrivacySignal signal-loss pattern to a
financial-access review setting using the UCI German Credit dataset. Credit
applications are ranked for manual review or assistance; checking-account status,
credit-history status, and savings status are treated as historical financial
signals that may be unavailable or minimized under a privacy-preserving workflow.

The raw UCI file is downloaded at runtime and is not redistributed in this repository.

![Financial-access public-data validation](assets/finance_german_credit_validation.svg)

![Financial-access recovery profile](assets/finance_german_credit_recovery_profile.svg)

## Task

- **Ranked candidate:** credit applications for review triage
- **Restricted historical signal:** checking-account status, credit-history status, and savings status
- **Permitted context:** amount, duration, purpose, employment, housing, property, installment rate, age, and other application context
- **Low-signal group:** thin-file applications with one or fewer existing credits
- **Metric:** NDCG@100, with binary relevance defined as the dataset's higher-risk credit class

## Results

{table}

The train-fitted recovery path closes {format_percent(float(recovery["full_signal_gap_closed"]))}
of the full-signal NDCG@100 gap without exposing the restricted financial-history
features at scoring time. The policy-aware partial path keeps those history
signals for non-thin-file applications while substituting recovered signal for
thin-file applications, closing {format_percent(float(policy["full_signal_gap_closed"]))}
of the same gap in this pilot.

## Interpretation

This is an external public-data validation of the system shape, not a credit
decisioning deployment. It shows how the method can be instantiated in a
financial-access workflow: define historical financial signals, suppress them at
scoring time, substitute a train-fitted reconstruction with a cohort stabilizer,
and measure review-ranking recovery. The dataset is old, compact, and does not
contain a real privacy-policy event, so the availability policy is simulated for
evaluation.
"""
    path.write_text(content, encoding="utf-8")


def run_validation() -> pd.DataFrame:
    data = load_credit_frame()
    scored = score_applications(data)
    summary = summarize_results(scored)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(TABLE_DIR / "finance_german_credit_scored_applications.csv", index=False)
    summary.to_csv(TABLE_DIR / "finance_german_credit_validation_summary.csv", index=False)
    write_svg(summary)
    write_recovery_profile_svg(
        summary,
        PROFILE_FIGURE_PATH,
        title="Financial-Access Recovery Profile",
        subtitle="Credit-application review under financial-history signal loss.",
        metric_col="overall_ndcg_at_100",
        low_signal_col="thin_file_ndcg_at_100",
        exposure_col="history_signal_exposure",
        low_signal_label="Thin-file NDCG@100",
    )
    write_gallery_card_svg(
        summary,
        GALLERY_FIGURE_PATH,
        title="Financial Access",
        subtitle="Credit-review ranking under financial-history signal loss.",
        metric_col="overall_ndcg_at_100",
        exposure_col="history_signal_exposure",
    )
    write_report(summary)
    return summary


def main() -> None:
    summary = run_validation()
    print("Finance German Credit validation:")
    print(summary.round(4).to_string(index=False))
    print(f"\nWrote: {REPORT_PATH}")
    print(f"Wrote: {FIGURE_PATH}")
    print(f"Wrote: {PROFILE_FIGURE_PATH}")
    print(f"Wrote: {GALLERY_FIGURE_PATH}")


if __name__ == "__main__":
    main()
