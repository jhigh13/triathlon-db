#!/usr/bin/env python
"""
venue_analysis.py — Historical race analysis for a given venue.

Queries the triathlon-db PostgreSQL database and produces a PowerPoint report
with swim/bike/run split analysis, pack dynamics, and weather history.

Usage:
    python scripts/venue_analysis.py --venue Yokohama --years 8
    python scripts/venue_analysis.py --venue Yokohama --years 8 --gender men
    python scripts/venue_analysis.py --venue Yokohama --years 8 --output report.pptx
"""

import argparse
import io
import os
import sys
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Allow running from repo root or scripts/ subdirectory
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "ppt files")
USAT_LOGO_PATH = os.path.join(REPO_ROOT, "docs", "power_bi_files", "USA_Triathlon_Logo.jpg")

# ── Brand colours (matching USAT reference deck) ──────────────────────────────
NAVY = RGBColor(0x00, 0x20, 0x60)       # #002060
RED = RGBColor(0xC0, 0x00, 0x00)        # #C00000
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
MID_GRAY = RGBColor(0xBF, 0xBF, 0xBF)
DARK_GRAY = RGBColor(0x26, 0x26, 0x26)
LIGHT_NAVY = RGBColor(0x00, 0x47, 0xAB)  # accent blue for charts

# Chart colour palette
C_NAVY = "#002060"
C_RED = "#C00000"
C_LIGHT_BLUE = "#4472C4"
C_LIGHT_RED = "#FF6B6B"
C_GRAY = "#808080"
C_ORANGE = "#E76F51"
C_GREEN = "#2A9D8F"
C_GOLD = "#F4A261"
C_VIOLET = "#9B5DE5"

FONT = "Arial"
SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# Matplotlib global style
plt.rcParams.update({
    "font.family": "Arial",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_engine():
    load_dotenv(override=True)
    uri = os.environ.get(
        "DB_URI",
        "postgresql+psycopg://postgres:Bc020406%21@localhost:5432/triathlon_results",
    )
    return create_engine(uri)


def parse_time_to_seconds(t) -> float | None:
    """Convert 'H:MM:SS', 'MM:SS', or raw-second strings to float seconds."""
    if t is None or str(t).strip() in ("", "None", "nan"):
        return None
    t = str(t).strip()
    parts = t.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(t)
    except (ValueError, IndexError):
        return None


def seconds_to_mmss(s: float | None, always_hours: bool = False) -> str:
    if s is None:
        return "—"
    s = int(round(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0 or always_hours:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def query_venue_events(engine, venue: str, years_back: int, gender_filter: str) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=years_back * 365)
    gender_clauses = []
    if gender_filter in ("men", "both"):
        gender_clauses.append("(e.prog_name ILIKE '%Elite Men%' AND e.prog_name NOT ILIKE '%Women%')")
    if gender_filter in ("women", "both"):
        gender_clauses.append("e.prog_name ILIKE '%Elite Women%'")
    gender_sql = " OR ".join(gender_clauses)

    sql = text(f"""
        SELECT
            e.event_id, e.prog_id,
            e.event_name, e.event_date, e.event_venue,
            e.cat_name, e.prog_name, e.prog_distance_category,
            e.swim_distance, e.bike_distance, e.run_distance,
            e.swim_laps, e.bike_laps, e.run_laps,
            e.temperature_air, e.temperature_water,
            e.wind, e.wetsuit, e.weather,
            e.wind_speed_kmh, e.wind_gust_kmh,
            e.apparent_temp, e.precipitation_mm
        FROM events e
        WHERE e.event_venue ILIKE :venue_pat
          AND e.event_date >= :cutoff
          AND e.event_date <= :today
          AND ({gender_sql})
          AND (e.is_para IS NULL OR e.is_para IS NOT TRUE)
        ORDER BY e.event_date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={
            "venue_pat": f"%{venue}%",
            "cutoff": cutoff,
            "today": date.today(),
        })
    return df


def query_splits(engine, event_id: int, prog_id: int) -> dict:
    sql = text("""
        SELECT swimtime, biketime, runtime, total_time
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid AND finish_status = 'FINISH'
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"eid": event_id, "pid": prog_id})

    result = {}
    for col, key in [("swimtime", "swim"), ("biketime", "bike"), ("runtime", "run"), ("total_time", "total")]:
        secs = df[col].apply(parse_time_to_seconds).dropna()
        result[f"{key}_fastest"] = float(secs.min()) if len(secs) else None
        result[f"{key}_avg"] = float(secs.mean()) if len(secs) else None
        result[f"{key}_n"] = len(secs)
    return result


def query_winner(engine, event_id: int, prog_id: int) -> dict:
    sql = text("""
        SELECT athlete_full_name, total_time
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid AND finish_position = 1
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"eid": event_id, "pid": prog_id}).fetchone()
    if row:
        return {"winner_name": row[0], "winner_time": row[1]}
    return {"winner_name": None, "winner_time": None}


def query_swim_exit_group(engine, event_id: int, prog_id: int) -> int | None:
    sql = text("""
        SELECT COUNT(*) FROM position_metrics
        WHERE event_id = :eid AND prog_id = :pid AND behindswim <= 15
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"eid": event_id, "pid": prog_id}).fetchone()
    return int(row[0]) if row and row[0] else None


def query_bike_pack(engine, event_id: int, prog_id: int) -> dict:
    sql = text("""
        SELECT pack_id, pack_size FROM wtcs_pack_membership
        WHERE event_id = :eid AND prog_id = :pid AND checkpoint = 'bike'
        ORDER BY elapsed_sec ASC LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"eid": event_id, "pid": prog_id}).fetchone()
    if row and row[1]:
        return {"lead_bike_pack": row[1], "source": "wtcs_pack"}

    sql2 = text("""
        SELECT COUNT(*) FROM position_metrics
        WHERE event_id = :eid AND prog_id = :pid AND behindbike <= 10
    """)
    with engine.connect() as conn:
        row2 = conn.execute(sql2, {"eid": event_id, "pid": prog_id}).fetchone()
    n = int(row2[0]) if row2 and row2[0] else None
    return {"lead_bike_pack": n, "source": "position_metrics_fallback"}


def query_field_size(engine, event_id: int, prog_id: int) -> int | None:
    # Try precomputed n_finishers first
    sql = text("""
        SELECT n_finishers FROM position_metrics
        WHERE event_id = :eid AND prog_id = :pid AND n_finishers IS NOT NULL LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"eid": event_id, "pid": prog_id}).fetchone()
    if row and row[0]:
        return int(row[0])
    # Fallback: count finishers directly
    sql2 = text("""
        SELECT COUNT(*) FROM race_results
        WHERE event_id = :eid AND prog_id = :pid AND finish_status = 'FINISH'
    """)
    with engine.connect() as conn:
        row2 = conn.execute(sql2, {"eid": event_id, "pid": prog_id}).fetchone()
    return int(row2[0]) if row2 and row2[0] else None


def query_position_times(engine, event_id: int, prog_id: int) -> dict:
    """Return total_time in seconds for positions 1, 2, 3, 5, 10, 20."""
    positions = [1, 2, 3, 5, 10, 20]
    sql = text("""
        SELECT finish_position, total_time, athlete_full_name
        FROM race_results
        WHERE event_id = :eid AND prog_id = :pid
          AND finish_position = ANY(:positions)
          AND finish_status = 'FINISH'
        ORDER BY finish_position
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"eid": event_id, "pid": prog_id, "positions": positions})

    result = {}
    winner_secs = None
    for _, row in df.iterrows():
        pos = int(row.finish_position)
        secs = parse_time_to_seconds(row.total_time)
        result[pos] = {"secs": secs, "name": row.athlete_full_name, "time_str": row.total_time}
        if pos == 1:
            winner_secs = secs

    # Compute gaps from winner
    for pos in positions:
        if pos in result and winner_secs and result[pos]["secs"]:
            result[pos]["gap"] = result[pos]["secs"] - winner_secs
        elif pos in result:
            result[pos]["gap"] = None

    return result


def collect_race_data(engine, events_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, ev in events_df.iterrows():
        eid, pid = int(ev.event_id), int(ev.prog_id)
        splits = query_splits(engine, eid, pid)
        if splits.get("total_n", 0) == 0:
            print(f"  Skipping {ev.event_date} {ev.prog_name} — no results (cancelled or unavailable)")
            continue
        winner = query_winner(engine, eid, pid)
        swim_group = query_swim_exit_group(engine, eid, pid)
        bike = query_bike_pack(engine, eid, pid)
        field_size = splits.get("total_n") or query_field_size(engine, eid, pid) or None
        pos_times = query_position_times(engine, eid, pid)

        rows.append({
            "event_id": eid,
            "prog_id": pid,
            "year": pd.to_datetime(ev.event_date).year,
            "date": ev.event_date,
            "event_name": ev.event_name,
            "cat_name": ev.cat_name or "",
            "prog_name": ev.prog_name or "",
            "prog_distance_category": ev.prog_distance_category or "",
            "swim_km": ev.swim_distance,
            "bike_km": ev.bike_distance,
            "run_km": ev.run_distance,
            "bike_laps": ev.bike_laps,
            "run_laps": ev.run_laps,
            "field_size": field_size,
            "swim_exit_group": swim_group,
            "lead_bike_pack": bike["lead_bike_pack"],
            "bike_pack_source": bike["source"],
            "temp_air": ev.temperature_air,
            "temp_water": ev.temperature_water,
            "wind_raw": ev.wind,
            "wind_kmh": ev.wind_speed_kmh,
            "wetsuit": ev.wetsuit,
            "weather": ev.weather,
            "winner_name": winner["winner_name"],
            "winner_time": winner["winner_time"],
            "pos_times": pos_times,
            **splits,
        })
    return rows


# ── PPTX helpers ───────────────────────────────────────────────────────────────

def _set_cell(cell, text_val: str, bold=False, font_size=10, color=None,
              align=PP_ALIGN.CENTER, bg_color=None, italic=False):
    """Set text, font, and background of a table cell."""
    tf = cell.text_frame
    tf.word_wrap = False
    para = tf.paragraphs[0]
    para.alignment = align
    # Clear any existing runs
    for run in para.runs:
        run.text = ""
    run = para.runs[0] if para.runs else para.add_run()
    run.text = str(text_val)
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = FONT
    run.font.color.rgb = color or DARK_GRAY

    if bg_color:
        from pptx.oxml.ns import qn
        from lxml import etree
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        # Remove existing fills
        for existing in tcPr.findall(qn("a:solidFill")):
            tcPr.remove(existing)
        solidFill = etree.SubElement(tcPr, qn("a:solidFill"))
        srgbClr = etree.SubElement(solidFill, qn("a:srgbClr"))
        srgbClr.set("val", str(bg_color))


def _add_textbox(slide, text: str, left, top, width, height,
                 font_size=12, bold=False, color=None, align=PP_ALIGN.LEFT,
                 italic=False):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = FONT
    run.font.color.rgb = color or DARK_GRAY
    return txb


def add_slide_chrome(slide, title: str, subtitle: str = "", show_logo: bool = True):
    """Add the standard chrome: navy header bar, red accent line, USAT logo."""
    # Navy header bar
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    # Title text in header
    _add_textbox(slide, title, Inches(0.25), Inches(0.05), Inches(10.5), Inches(0.65),
                 font_size=28, bold=True, color=WHITE)

    if subtitle:
        _add_textbox(slide, subtitle, Inches(0.25), Inches(0.68), Inches(10.5), Inches(0.35),
                     font_size=12, color=RGBColor(0xAA, 0xBB, 0xDD))

    # Red accent line
    accent = slide.shapes.add_shape(1, Inches(0), Inches(1.05), SLIDE_W, Inches(0.07))
    accent.fill.solid()
    accent.fill.fore_color.rgb = RED
    accent.line.fill.background()

    # USAT logo top-right
    if show_logo and os.path.exists(USAT_LOGO_PATH):
        slide.shapes.add_picture(USAT_LOGO_PATH, Inches(11.9), Inches(0.03), Inches(1.35), Inches(1.0))

    # Thin navy footer strip
    footer = slide.shapes.add_shape(1, Inches(0), Inches(7.35), SLIDE_W, Inches(0.15))
    footer.fill.solid()
    footer.fill.fore_color.rgb = NAVY
    footer.line.fill.background()


def fig_to_image(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Slide builders ─────────────────────────────────────────────────────────────

def add_title_slide(prs: Presentation, venue: str, years_back: int):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Full navy background
    bg = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    # Red diagonal accent bar at right
    accent = slide.shapes.add_shape(1, Inches(12.5), Inches(0), Inches(0.83), SLIDE_H)
    accent.fill.solid()
    accent.fill.fore_color.rgb = RED
    accent.line.fill.background()

    # White horizontal rule
    rule = slide.shapes.add_shape(1, Inches(0.5), Inches(4.1), Inches(11.8), Inches(0.06))
    rule.fill.solid()
    rule.fill.fore_color.rgb = WHITE
    rule.line.fill.background()

    # USAT logo centered
    if os.path.exists(USAT_LOGO_PATH):
        slide.shapes.add_picture(USAT_LOGO_PATH, Inches(0.6), Inches(0.4), Inches(2.1), Inches(1.6))

    # Main title
    _add_textbox(slide, f"{venue.upper()} RACE HISTORY",
                 Inches(0.5), Inches(2.0), Inches(11.8), Inches(1.4),
                 font_size=54, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Subtitle
    _add_textbox(slide, f"Elite Analysis  ·  Past {years_back} Years  ·  {date.today().strftime('%B %Y')}",
                 Inches(0.5), Inches(4.3), Inches(11.8), Inches(0.6),
                 font_size=18, color=RGBColor(0xAA, 0xBB, 0xDD), align=PP_ALIGN.CENTER)

    _add_textbox(slide, "USA Triathlon High Performance",
                 Inches(0.5), Inches(5.0), Inches(11.8), Inches(0.5),
                 font_size=14, color=RGBColor(0x80, 0x90, 0xB0), align=PP_ALIGN.CENTER)


def add_section_divider(prs: Presentation, gender: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    red_bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(0.5), SLIDE_H)
    red_bar.fill.solid()
    red_bar.fill.fore_color.rgb = RED
    red_bar.line.fill.background()

    rule = slide.shapes.add_shape(1, Inches(0.8), Inches(4.2), Inches(11.5), Inches(0.06))
    rule.fill.solid()
    rule.fill.fore_color.rgb = WHITE
    rule.line.fill.background()

    icon = "♂" if gender.lower() == "men" else "♀"
    _add_textbox(slide, f"ELITE {gender.upper()}  {icon}",
                 Inches(0.8), Inches(2.3), Inches(11.5), Inches(1.8),
                 font_size=60, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    if os.path.exists(USAT_LOGO_PATH):
        slide.shapes.add_picture(USAT_LOGO_PATH, Inches(11.0), Inches(6.3), Inches(2.0), Inches(1.1))


def add_overview_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Race Overview  —  Elite {gender.title()}", venue)

    # Column definitions: (header, width_inches, align)
    cols = [
        ("Year",        0.65,  PP_ALIGN.CENTER),
        ("Date",        0.95,  PP_ALIGN.CENTER),
        ("Category",    2.0,   PP_ALIGN.LEFT),
        ("Field",       0.6,   PP_ALIGN.CENTER),
        ("Winner",      2.1,   PP_ALIGN.LEFT),
        ("Winner Time", 1.0,   PP_ALIGN.CENTER),
        ("Air Temp",    0.85,  PP_ALIGN.CENTER),
        ("Water Temp",  0.9,   PP_ALIGN.CENTER),
        ("Swim km",     0.75,  PP_ALIGN.CENTER),
        ("Bike km",     0.75,  PP_ALIGN.CENTER),
        ("Run km",      0.75,  PP_ALIGN.CENTER),
    ]
    n_rows = len(rows) + 1
    n_cols = len(cols)
    row_h = min(0.42, 5.6 / n_rows)
    total_w = sum(c[1] for c in cols)
    scale = 12.73 / total_w  # fit within slide width
    scaled_widths = [c[1] * scale for c in cols]

    tbl_shape = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(0.3), Inches(1.2),
        Inches(12.73), Inches(row_h * n_rows)
    )
    tbl = tbl_shape.table
    for i, w in enumerate(scaled_widths):
        tbl.columns[i].width = Inches(w)

    # Header row
    for ci, (col_name, _, align) in enumerate(cols):
        _set_cell(tbl.cell(0, ci), col_name, bold=True, font_size=10,
                  color=WHITE, align=PP_ALIGN.CENTER, bg_color=NAVY)

    def _fmt_dist(val) -> str:
        """Format a distance value stored in either km or metres."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        # Values > 100 are assumed to be stored in metres; convert to km
        if val > 100:
            val = val / 1000.0
        return f"{val:.2f}" if val < 2 else f"{val:.1f}"

    def _dims_equal(a, b) -> bool:
        """Compare two distance tuples treating NaN as equal."""
        for x, y in zip(a, b):
            x_nan = x is None or (isinstance(x, float) and np.isnan(x))
            y_nan = y is None or (isinstance(y, float) and np.isnan(y))
            if x_nan and y_nan:
                continue
            if x_nan != y_nan or x != y:
                return False
        return True

    prev_dims = None
    footnote_needed = False
    for ri, row in enumerate(rows, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE

        current_dims = (row["swim_km"], row["bike_km"], row["run_km"])
        course_flag = ""
        if prev_dims is not None and not _dims_equal(prev_dims, current_dims):
            course_flag = " *"
            footnote_needed = True
        prev_dims = current_dims

        cat = (row["cat_name"] or row["prog_distance_category"] or "—").replace("World Triathlon Championship Series", "WTCS")
        winner_last = ""
        if row["winner_name"]:
            parts = row["winner_name"].split()
            winner_last = parts[-1] if parts else row["winner_name"]

        air   = str(row.get("temp_air")   or "—").strip()
        water = str(row.get("temp_water") or "—").strip()

        vals = [
            str(row["year"]),
            pd.to_datetime(row["date"]).strftime("%b %d"),
            cat + course_flag,
            str(row["field_size"]) if row["field_size"] else "—",
            winner_last,
            seconds_to_mmss(parse_time_to_seconds(row["winner_time"])),
            air,
            water,
            _fmt_dist(row["swim_km"]),
            _fmt_dist(row["bike_km"]),
            _fmt_dist(row["run_km"]),
        ]
        for ci, (v, (_, _, align)) in enumerate(zip(vals, cols)):
            _set_cell(tbl.cell(ri, ci), v, font_size=9, align=align, bg_color=bg)

    if footnote_needed:
        _add_textbox(slide, "* Course distance changed vs. prior year",
                     Inches(0.3), Inches(7.1), Inches(6), Inches(0.28),
                     font_size=8, color=MID_GRAY, italic=True)


def add_swim_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Swim Exit Dynamics  —  Elite {gender.title()}", venue)

    years = [str(r["year"]) for r in rows]
    group_sizes = [r["swim_exit_group"] or 0 for r in rows]
    fastest = [r["swim_fastest"] for r in rows]
    avg_swim = [r["swim_avg"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.patch.set_facecolor("white")

    # Left: swim exit group size
    ax1 = axes[0]
    bar_colors = [C_NAVY if g >= 20 else C_RED for g in group_sizes]
    bars = ax1.bar(years, group_sizes, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, group_sizes):
        if val:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(val), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
    ax1.set_title("Athletes Within 15 sec of Swim Leader", fontsize=11, pad=8)
    ax1.set_ylabel("# Athletes", fontsize=10)
    ax1.set_ylim(0, max(group_sizes + [1]) * 1.3)
    ax1.tick_params(axis="x", rotation=45, labelsize=9)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.set_facecolor("white")

    # Right: fastest vs avg swim time
    ax2 = axes[1]
    x = np.arange(len(years))
    w = 0.35
    fast_min = [f / 60 if f else 0 for f in fastest]
    avg_min  = [a / 60 if a else 0 for a in avg_swim]
    bars_f = ax2.bar(x - w / 2, fast_min, w, label="Fastest", color=C_NAVY, edgecolor="white")
    bars_a = ax2.bar(x + w / 2, avg_min,  w, label="Average", color=C_LIGHT_BLUE, edgecolor="white", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, rotation=45, fontsize=9)
    ax2.set_title("Fastest vs. Average Swim Time", fontsize=11, pad=8)
    ax2.set_ylabel("Time (min:sec)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_facecolor("white")
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v)}:{int(round((v % 1) * 60)):02d}")
    )
    for bar, secs in zip(bars_f, fastest):
        if secs:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     seconds_to_mmss(secs), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar, secs in zip(bars_a, avg_swim):
        if secs:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     seconds_to_mmss(secs), ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout(pad=1.5)
    slide.shapes.add_picture(fig_to_image(fig), Inches(0.3), Inches(1.18), Inches(12.73), Inches(5.0))


def add_bike_pack_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Bike Pack Dynamics  —  Elite {gender.title()}", venue)

    years = [str(r["year"]) for r in rows]
    pack_sizes = [r["lead_bike_pack"] or 0 for r in rows]
    bar_colors = [C_NAVY if p >= 15 else C_RED for p in pack_sizes]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor("white")
    bars = ax.bar(years, pack_sizes, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.55)
    for bar, val in zip(bars, pack_sizes):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333333")

    ax.axhline(15, color=C_GRAY, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title("Lead Pack Size Exiting T2 (Off Bike)", fontsize=13, pad=10)
    ax.set_ylabel("Athletes in Lead Pack", fontsize=11)
    ax.set_ylim(0, max(pack_sizes + [1]) * 1.35)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_facecolor("white")

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=C_NAVY, label="Large pack (≥15 athletes)"),
        Patch(facecolor=C_RED,  label="Breakaway / small lead group (<15)"),
        plt.Line2D([0], [0], color=C_GRAY, linestyle="--", label="15-athlete threshold"),
    ]
    ax.legend(handles=legend_els, fontsize=10, loc="upper right")
    plt.tight_layout(pad=1.5)

    slide.shapes.add_picture(fig_to_image(fig), Inches(1.2), Inches(1.18), Inches(10.9), Inches(5.0))

    # Fallback footnote
    if any(r["bike_pack_source"] == "position_metrics_fallback" for r in rows):
        _add_textbox(slide,
                     "* Years without pack membership data use fallback: athletes within 10 sec of bike leader",
                     Inches(0.3), Inches(6.9), Inches(10), Inches(0.4),
                     font_size=8, color=MID_GRAY, italic=True)


def add_run_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Run Split Analysis  —  Elite {gender.title()}", venue)

    years = [str(r["year"]) for r in rows]
    fastest = [r["run_fastest"] for r in rows]
    avg_run  = [r["run_avg"]     for r in rows]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor("white")
    x = np.arange(len(years))
    w = 0.35
    fast_min = [f / 60 if f else 0 for f in fastest]
    avg_min  = [a / 60 if a else 0 for a in avg_run]

    bars_f = ax.bar(x - w / 2, fast_min, w, label="Fastest", color=C_NAVY, edgecolor="white")
    bars_a = ax.bar(x + w / 2, avg_min,  w, label="Average", color=C_LIGHT_BLUE, edgecolor="white", alpha=0.8)

    for bar, secs in zip(bars_f, fastest):
        if secs:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    seconds_to_mmss(secs), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, secs in zip(bars_a, avg_run):
        if secs:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    seconds_to_mmss(secs), ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, fontsize=10)
    ax.set_title("Fastest vs. Average Run Split by Year", fontsize=13, pad=10)
    ax.set_ylabel("Time (min:sec)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_facecolor("white")

    all_vals = [v for v in fast_min + avg_min if v]
    if all_vals:
        ax.set_ylim(min(all_vals) * 0.97, max(all_vals) * 1.08)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v)}:{int(round((v % 1) * 60)):02d}")
    )
    plt.tight_layout(pad=1.5)
    slide.shapes.add_picture(fig_to_image(fig), Inches(1.2), Inches(1.18), Inches(10.9), Inches(5.0))


def add_position_times_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    """Line chart + table: times for 1st, 2nd, 3rd, 5th, 10th, 20th place per year."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Key Position Times  —  Elite {gender.title()}", venue)

    positions = [1, 2, 3, 5, 10, 20]
    pos_labels = {1: "1st", 2: "2nd", 3: "3rd", 5: "5th", 10: "10th", 20: "20th"}
    colors_pos = {1: C_NAVY, 2: C_RED, 3: C_ORANGE, 5: C_LIGHT_BLUE, 10: C_GREEN, 20: C_GRAY}

    years = [str(r["year"]) for r in rows]

    # ── Left: line chart of gap from winner (seconds) ──
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 3.4))
    fig.patch.set_facecolor("white")

    for pos in [2, 3, 5, 10, 20]:
        gaps = []
        for row in rows:
            pt = row["pos_times"].get(pos)
            gaps.append(pt["gap"] if pt and pt.get("gap") is not None else np.nan)
        ax_left.plot(years, gaps, marker="o", label=pos_labels[pos],
                     color=colors_pos[pos], linewidth=2, markersize=6)
        # Label endpoint
        last_valid = [(y, g) for y, g in zip(years, gaps) if not np.isnan(g)]
        if last_valid:
            ly, lg = last_valid[-1]
            ax_left.annotate(f"{int(round(lg))}s", xy=(ly, lg),
                             xytext=(4, 0), textcoords="offset points",
                             fontsize=7.5, color=colors_pos[pos], fontweight="bold")

    ax_left.set_title("Gap from Winner (seconds)", fontsize=11, pad=8)
    ax_left.set_ylabel("Seconds behind 1st place", fontsize=10)
    ax_left.tick_params(axis="x", rotation=45, labelsize=9)
    ax_left.set_facecolor("white")
    ax_left.legend(fontsize=9, loc="upper left")
    ax_left.axhline(0, color=C_NAVY, linewidth=0.8, alpha=0.4)

    # ── Right: bar chart of average gaps per position ──
    avg_gaps = []
    for pos in [2, 3, 5, 10, 20]:
        all_gaps = []
        for row in rows:
            pt = row["pos_times"].get(pos)
            if pt and pt.get("gap") is not None:
                all_gaps.append(pt["gap"])
        avg_gaps.append(np.mean(all_gaps) if all_gaps else 0)

    bar_labels = [pos_labels[p] for p in [2, 3, 5, 10, 20]]
    bar_clrs = [colors_pos[p] for p in [2, 3, 5, 10, 20]]
    bars = ax_right.bar(bar_labels, avg_gaps, color=bar_clrs, edgecolor="white", width=0.55)
    for bar, val in zip(bars, avg_gaps):
        if val:
            ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                          f"{int(round(val))}s", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax_right.set_title("Avg Gap from 1st Place (all years)", fontsize=11, pad=8)
    ax_right.set_ylabel("Seconds behind winner", fontsize=10)
    ax_right.set_facecolor("white")
    ax_right.tick_params(axis="x", labelsize=10)

    plt.tight_layout(pad=1.2)
    slide.shapes.add_picture(fig_to_image(fig), Inches(0.3), Inches(1.18), Inches(12.73), Inches(3.3))

    # ── Summary table: actual times per position per year ──
    tbl_rows = 1 + len(rows)
    tbl_cols = 1 + len(positions)
    tbl_top = Inches(4.58)
    row_h = min(0.38, 2.7 / tbl_rows)
    tbl_h = Inches(row_h * tbl_rows)
    tbl_shape = slide.shapes.add_table(tbl_rows, tbl_cols,
                                       Inches(0.3), tbl_top, Inches(12.73), tbl_h)
    tbl = tbl_shape.table
    col_widths = [Inches(0.7)] + [Inches(12.03 / len(positions))] * len(positions)
    for ci, w in enumerate(col_widths):
        tbl.columns[ci].width = w

    _set_cell(tbl.cell(0, 0), "Year", bold=True, color=WHITE, bg_color=NAVY)
    for ci, pos in enumerate(positions, start=1):
        _set_cell(tbl.cell(0, ci), pos_labels[pos], bold=True, color=WHITE, bg_color=NAVY)

    for ri, row in enumerate(rows, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        _set_cell(tbl.cell(ri, 0), str(row["year"]), bold=True, color=DARK_GRAY, bg_color=bg)
        for ci, pos in enumerate(positions, start=1):
            pt = row["pos_times"].get(pos)
            if pt and pt.get("secs"):
                time_str = seconds_to_mmss(pt["secs"])
                gap_str = f"+{int(round(pt['gap']))}s" if pt.get("gap") else ""
                val = f"{time_str}\n{gap_str}" if gap_str else time_str
            else:
                val = "—"
            _set_cell(tbl.cell(ri, ci), val, font_size=8.5, bg_color=bg)


def add_weather_slide(prs: Presentation, rows: list[dict], gender: str, venue: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_chrome(slide, f"Race Conditions  —  Elite {gender.title()}", venue)

    cols = [
        ("Year",       0.65, PP_ALIGN.CENTER),
        ("Date",       0.95, PP_ALIGN.CENTER),
        ("Air Temp",   1.1,  PP_ALIGN.CENTER),
        ("Water Temp", 1.1,  PP_ALIGN.CENTER),
        ("Wind (km/h)", 1.1, PP_ALIGN.CENTER),
        ("Wetsuit",    1.0,  PP_ALIGN.CENTER),
        ("Conditions", 6.82, PP_ALIGN.LEFT),
    ]
    n_rows = len(rows) + 1
    tbl_shape = slide.shapes.add_table(
        n_rows, len(cols),
        Inches(0.3), Inches(1.2),
        Inches(12.73), Inches(min(5.8, 0.48 * n_rows + 0.48))
    )
    tbl = tbl_shape.table
    total_w = sum(c[1] for c in cols)
    scale = 12.73 / total_w
    for ci, (_, w, _) in enumerate(cols):
        tbl.columns[ci].width = Inches(w * scale)

    for ci, (name, _, align) in enumerate(cols):
        _set_cell(tbl.cell(0, ci), name, bold=True, color=WHITE, align=PP_ALIGN.CENTER, bg_color=NAVY)

    for ri, row in enumerate(rows, start=1):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        wind_val = (f"{row['wind_kmh']:.0f}" if row.get("wind_kmh") else (row.get("wind_raw") or "—"))
        air    = str(row.get("temp_air")   or "—").strip()
        water  = str(row.get("temp_water") or "—").strip()
        wetsuit = str(row.get("wetsuit")   or "—").strip()
        weather = str(row.get("weather")   or "—").strip()
        if len(weather) > 90:
            weather = weather[:87] + "..."

        vals = [str(row["year"]), pd.to_datetime(row["date"]).strftime("%b %d"),
                air, water, str(wind_val), wetsuit, weather]
        for ci, (v, (_, _, align)) in enumerate(zip(vals, cols)):
            _set_cell(tbl.cell(ri, ci), v, font_size=9.5, align=align, bg_color=bg)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate venue historical race analysis PPT")
    parser.add_argument("--venue", required=True, help="Venue name (e.g. 'Yokohama')")
    parser.add_argument("--years", type=int, default=8, help="Years to look back (default: 8)")
    parser.add_argument("--gender", choices=["men", "women", "both"], default="both")
    parser.add_argument("--output", default=None, help="Output .pptx filename (optional)")
    args = parser.parse_args()

    fname = args.output or f"{args.venue.replace(' ', '_')}_{date.today()}.pptx"
    if not os.path.isabs(fname):
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        fname = os.path.join(DEFAULT_OUTPUT_DIR, fname)

    print(f"Connecting to database...")
    engine = get_engine()

    print(f"Querying events at '{args.venue}' (past {args.years} years)...")
    events_df = query_venue_events(engine, args.venue, args.years, args.gender)
    if events_df.empty:
        print(f"No events found for venue '{args.venue}'.")
        sys.exit(1)

    print(f"Found {len(events_df)} program(s):")
    for _, r in events_df.iterrows():
        print(f"  {r.event_date}  {r.event_name}  ({r.prog_name})")

    men_df   = events_df[events_df.prog_name.str.contains("Elite Men",   case=False, na=False) &
                         ~events_df.prog_name.str.contains("Women",      case=False, na=False)]
    women_df = events_df[events_df.prog_name.str.contains("Elite Women", case=False, na=False)]

    print("\nCollecting race data...")
    men_data   = collect_race_data(engine, men_df)   if not men_df.empty   else []
    women_data = collect_race_data(engine, women_df) if not women_df.empty else []

    print("Building PowerPoint...")
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    add_title_slide(prs, args.venue, args.years)

    gender_sections = []
    if men_data   and args.gender in ("men",   "both"): gender_sections.append(("Men",   men_data))
    if women_data and args.gender in ("women", "both"): gender_sections.append(("Women", women_data))

    for gender_label, data in gender_sections:
        print(f"  Building {gender_label} slides ({len(data)} races)...")
        add_section_divider(prs, gender_label)
        add_overview_slide(prs, data, gender_label, args.venue)
        add_swim_slide(prs, data, gender_label, args.venue)
        add_bike_pack_slide(prs, data, gender_label, args.venue)
        add_run_slide(prs, data, gender_label, args.venue)
        add_position_times_slide(prs, data, gender_label, args.venue)
        add_weather_slide(prs, data, gender_label, args.venue)

    prs.save(fname)
    print(f"\nDone! Saved to: {fname}")


if __name__ == "__main__":
    main()
