from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _short_method_label(method: str) -> str:
    lowered = method.lower()
    if lowered.startswith("full"):
        return "Full"
    if lowered.startswith("no ") or lowered.startswith("summary") or lowered.startswith("context"):
        return "Baseline"
    if "policy-aware" in lowered:
        return "Policy-aware"
    if "cohort" in lowered:
        return "Aggregate"
    if "train-fitted" in lowered:
        return "Recovered"
    return method.split()[0]


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.0f}%"


def write_recovery_profile_svg(
    summary: pd.DataFrame,
    path: Path,
    title: str,
    subtitle: str,
    metric_col: str,
    low_signal_col: str,
    exposure_col: str,
    low_signal_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = summary.copy()
    rows["short_label"] = rows["method"].map(_short_method_label)
    baseline_low = float(rows.loc[rows["full_signal_gap_closed"].eq(0.0), low_signal_col].iloc[0])
    rows["low_signal_lift"] = rows[low_signal_col].astype(float) - baseline_low

    width = 1280
    height = 700
    left_x = 64
    left_y = 108
    left_w = 620
    row_gap = 64
    axis_x0 = left_x + 136
    axis_x1 = left_x + left_w - 124
    value_x = axis_x1 + 24
    exposure_x = 742
    exposure_y = 114
    exposure_w = 468
    exposure_h = 222
    lift_x = 742
    lift_y = 392
    lift_w = 468
    lift_h = 128

    colors = {
        "Full": "#047857",
        "Baseline": "#6b7280",
        "Aggregate": "#0f766e",
        "Recovered": "#2563eb",
        "Policy-aware": "#be123c",
    }

    def gap_x(value: float) -> float:
        clipped = max(0.0, min(1.0, float(value)))
        return axis_x0 + clipped * (axis_x1 - axis_x0)

    def scatter_x(value: float) -> float:
        clipped = max(0.0, min(1.0, float(value)))
        return exposure_x + 34 + clipped * (exposure_w - 78)

    def scatter_y(value: float) -> float:
        clipped = max(0.0, min(1.0, float(value)))
        return exposure_y + exposure_h - 34 - clipped * (exposure_h - 72)

    max_lift = max(0.01, float(rows["low_signal_lift"].abs().max()))

    def lift_x_pos(value: float) -> float:
        center = lift_x + 116
        scale = (lift_w - 150) / (2.0 * max_lift)
        return center + float(value) * scale

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="40" y="38" font-family="Arial, sans-serif" font-size="23" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="40" y="64" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{escape(subtitle)}</text>',
        '<text x="64" y="96" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Recovery ladder</text>',
        '<text x="542" y="96" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">Utility vs exposure</text>',
        f'<line x1="{axis_x0}" y1="{left_y + 242}" x2="{axis_x1}" y2="{left_y + 242}" stroke="#d1d5db" stroke-width="1"/>',
    ]

    for pct in [0.0, 0.5, 1.0]:
        x = gap_x(pct)
        lines.append(f'<line x1="{x:.1f}" y1="{left_y + 226}" x2="{x:.1f}" y2="{left_y + 250}" stroke="#d1d5db" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{left_y + 270}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{int(pct * 100)}%</text>')

    for idx, row in rows.iterrows():
        y = left_y + idx * row_gap
        label = str(row["short_label"])
        color = colors.get(label, "#2563eb")
        gap = float(row["full_signal_gap_closed"])
        x = gap_x(gap)
        metric = float(row[metric_col])
        exposure = float(row[exposure_col])
        lines.extend(
            [
                f'<text x="{left_x}" y="{y + 5}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{escape(label)}</text>',
                f'<line x1="{axis_x0}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{color}" stroke-width="7" stroke-linecap="round" opacity="0.28"/>',
                f'<circle cx="{x:.1f}" cy="{y}" r="9" fill="{color}"/>',
                f'<text x="{value_x}" y="{y + 5}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{_format_percent(gap)} closed</text>',
                f'<text x="{value_x}" y="{y + 22}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">metric {metric:.3f}, exposure {_format_percent(exposure)}</text>',
            ]
        )

    lines.extend(
        [
            f'<rect x="{exposure_x}" y="{exposure_y}" width="{exposure_w}" height="{exposure_h}" rx="8" fill="#ffffff" stroke="#e5e7eb"/>',
            f'<text x="{exposure_x + 18}" y="{exposure_y + 26}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">Gap closed vs restricted-signal exposure</text>',
            f'<line x1="{exposure_x + 34}" y1="{exposure_y + exposure_h - 34}" x2="{exposure_x + exposure_w - 30}" y2="{exposure_y + exposure_h - 34}" stroke="#9ca3af"/>',
            f'<line x1="{exposure_x + 34}" y1="{exposure_y + 42}" x2="{exposure_x + 34}" y2="{exposure_y + exposure_h - 34}" stroke="#9ca3af"/>',
            f'<text x="{exposure_x + exposure_w - 30}" y="{exposure_y + exposure_h - 12}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">gap closed</text>',
            f'<text x="{exposure_x + 14}" y="{exposure_y + 52}" transform="rotate(-90 {exposure_x + 14} {exposure_y + 52})" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">exposure</text>',
        ]
    )
    for pct in [0.0, 0.5, 1.0]:
        x = scatter_x(pct)
        y = scatter_y(pct)
        lines.append(f'<line x1="{x:.1f}" y1="{exposure_y + exposure_h - 34}" x2="{x:.1f}" y2="{exposure_y + exposure_h - 28}" stroke="#9ca3af"/>')
        lines.append(f'<text x="{x:.1f}" y="{exposure_y + exposure_h - 16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#6b7280">{int(pct * 100)}</text>')
        lines.append(f'<line x1="{exposure_x + 28}" y1="{y:.1f}" x2="{exposure_x + 34}" y2="{y:.1f}" stroke="#9ca3af"/>')
        lines.append(f'<text x="{exposure_x + 23}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#6b7280">{int(pct * 100)}</text>')

    for _, row in rows.iterrows():
        label = str(row["short_label"])
        color = colors.get(label, "#2563eb")
        x = scatter_x(float(row["full_signal_gap_closed"]))
        y = scatter_y(float(row[exposure_col]))
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" opacity="0.88"/>')
        lines.append(f'<text x="{x + 11:.1f}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#374151">{escape(label)}</text>')

    lines.extend(
        [
            f'<rect x="{lift_x}" y="{lift_y}" width="{lift_w}" height="{lift_h}" rx="8" fill="#ffffff" stroke="#e5e7eb"/>',
            f'<text x="{lift_x + 18}" y="{lift_y + 26}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{escape(low_signal_label)} lift vs baseline</text>',
            f'<line x1="{lift_x + 116}" y1="{lift_y + 40}" x2="{lift_x + 116}" y2="{lift_y + lift_h - 22}" stroke="#d1d5db" stroke-width="1"/>',
        ]
    )
    lift_rows = rows[~rows["full_signal_gap_closed"].eq(0.0)]
    for draw_idx, (_, row) in enumerate(lift_rows.iterrows()):
        y = lift_y + 52 + draw_idx * 24
        label = str(row["short_label"])
        color = colors.get(label, "#2563eb")
        value = float(row["low_signal_lift"])
        x0 = lift_x + 116
        x1 = lift_x_pos(value)
        bar_x = min(x0, x1)
        bar_w = abs(x1 - x0)
        lines.extend(
            [
                f'<text x="{lift_x + 18}" y="{y + 5}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{escape(label)}</text>',
                f'<rect x="{bar_x:.1f}" y="{y - 8}" width="{bar_w:.1f}" height="14" rx="3" fill="{color}" opacity="0.82"/>',
                f'<text x="{max(x0, x1) + 8:.1f}" y="{y + 5}" font-family="Arial, sans-serif" font-size="11" fill="#374151">{value:+.3f}</text>',
            ]
        )

    lines.extend(
        [
            '<text x="64" y="636" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">Gap closed compares each method with the no-signal baseline and full-signal reference for this public-data pilot.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
