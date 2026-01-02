"""Export WTCS radar charts to image files for PowerPoint.

Usage examples:
- SVG (recommended for PPT vector quality):
  python -m tri_analysis.export_wtcs_radar --year 2025 --athlete-name "Morgan Pearson" --format svg

- PNG:
  python -m tri_analysis.export_wtcs_radar --year 2025 --athlete-id 12345 --format png

Notes:
- Uses the same WTCS radar scoring as Streamlit (see tri_analysis.wtcs_radar).
- SVG/PNG export defaults to a Matplotlib backend to avoid Kaleido hangs.
- HTML export uses Plotly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tri_analysis.database import get_engine
from tri_analysis.wtcs_performance import WTCSFilters, fetch_wtcs_us_dataset
from tri_analysis.wtcs_radar import compute_wtcs_radar_profile


def _build_radar_figure(radar_df: pd.DataFrame, title: str) -> "px.Figure":
    radar_plot = radar_df.copy()
    radar_plot["value"] = pd.to_numeric(radar_plot["value"], errors="coerce")

    fig = px.line_polar(
        radar_plot,
        r="value",
        theta="category",
        line_close=True,
        title=title,
    )
    fig.update_traces(
        connectgaps=False,
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.22)",
        line=dict(color="rgba(31, 119, 180, 0.95)", width=3),
        marker=dict(size=7, color="rgba(31, 119, 180, 1.0)"),
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[1, 10], tickmode="array", tickvals=[1, 3, 5, 7, 9, 10], showline=True),
            angularaxis=dict(direction="clockwise"),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _export_radar_matplotlib(radar_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Write a radar chart using Matplotlib (SVG/PNG), suitable for PowerPoint."""
    df = radar_df.copy()
    categories = df["category"].astype(str).tolist()
    values = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=float)

    n = len(categories)
    if n == 0:
        raise ValueError("No radar categories to plot")

    # Angles for axes
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    values_closed = np.concatenate([values, values[:1]])

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2.0)  # start at top

    # Grid / ticks
    ax.set_ylim(1, 10)
    ax.set_yticks([1, 3, 5, 7, 9, 10])
    ax.set_yticklabels(["1", "3", "5", "7", "9", "10"], fontsize=9)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11)

    # Plot line; fill only if all points present
    color = "#1f77b4"
    ax.plot(angles_closed, values_closed, color=color, linewidth=3)
    if not np.isnan(values).any():
        ax.fill(angles_closed, values_closed, color=color, alpha=0.22)
    ax.scatter(angles, values, color=color, s=35, zorder=3)

    ax.set_title(title, va="bottom", fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".svg":
        fig.savefig(out_path, format="svg", bbox_inches="tight")
    elif suffix == ".png":
        fig.savefig(out_path, format="png", bbox_inches="tight")
    else:
        raise ValueError(f"Unsupported matplotlib output format: {suffix}")
    plt.close(fig)


def _resolve_athlete_id(engine, year: int, athlete_name: str) -> int:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    # Search full WTCS field across genders; then exact-match by name.
    # Use full WTCS field (no country filter) for name resolution.
    filters = WTCSFilters(start_date=start, end_date=end, gender=None, para_filter=False, country_codes=[], min_events=1)
    df = fetch_wtcs_us_dataset(engine, filters)
    if df.empty:
        raise ValueError(f"No WTCS rows found for year={year} (check DB / filters)")

    names = df[["athlete_id", "full_name"]].dropna().drop_duplicates()
    exact = names[names["full_name"].astype(str) == athlete_name]
    if len(exact) == 1:
        return int(exact.iloc[0]["athlete_id"])

    # Fallback: case-insensitive contains
    contains = names[names["full_name"].astype(str).str.contains(athlete_name, case=False, na=False)]
    contains = contains.sort_values("full_name")
    if len(contains) == 1:
        return int(contains.iloc[0]["athlete_id"])
    if len(contains) == 0:
        raise ValueError(f"Athlete name not found: '{athlete_name}'")

    options = ", ".join([f"{r.athlete_id}:{r.full_name}" for r in contains.itertuples(index=False)])
    raise ValueError(f"Ambiguous athlete-name match. Use --athlete-id or a more specific name. Matches: {options}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export WTCS radar chart for an athlete to SVG/PNG for PowerPoint")
    parser.add_argument("--year", type=int, required=True, help="Season year (uses YYYY-01-01..YYYY-12-31)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--athlete-id", type=int, help="Athlete id")
    group.add_argument("--athlete-name", type=str, help="Athlete full name (exact match preferred)")

    parser.add_argument("--format", choices=["svg", "png", "html"], default="svg", help="Output format")
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help="Export backend for svg/png. HTML always uses plotly.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output path (default: outputs/wtcs_radar_...)")

    args = parser.parse_args(argv)

    engine = get_engine()

    athlete_id = args.athlete_id
    if athlete_id is None:
        athlete_id = _resolve_athlete_id(engine, args.year, args.athlete_name)

    start = f"{args.year}-01-01"
    end = f"{args.year}-12-31"

    profile = compute_wtcs_radar_profile(
        engine,
        athlete_id=int(athlete_id),
        season_start=start,
        season_end=end,
        para_filter=False,
    )

    radar_df = pd.DataFrame(
        [
            {"category": "Swim", "value": profile.swim, "n": profile.n_swim},
            {"category": "Bike", "value": profile.bike, "n": profile.n_bike},
            {"category": "Run", "value": profile.run, "n": profile.n_run},
            {"category": "Transitions", "value": profile.transitions, "n": profile.n_transitions},
        ]
    )

    title = f"WTCS Strengths & Weakness vs. The Field -- {profile.full_name} -- {args.year}"

    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"wtcs_radar_{args.year}_{int(athlete_id)}.{args.format}"

    if args.format in {"svg", "png"}:
        if args.backend == "matplotlib":
            _export_radar_matplotlib(radar_df, title=title, out_path=out_path)
        else:
            fig = _build_radar_figure(radar_df, title=title)
            fig.write_image(str(out_path), scale=2)
    elif args.format == "html":
        fig = _build_radar_figure(radar_df, title=title)
        fig.write_html(str(out_path), include_plotlyjs="cdn")

    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
