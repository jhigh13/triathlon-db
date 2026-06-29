"""
Weekly US athlete rankings email.

Usage:
    # Full run: refresh → preview to you → sync Supabase → send to all after 1 hour
    python scripts/weekly_rankings_email.py

    # Skip API refresh, just query DB and send
    python scripts/weekly_rankings_email.py --no-refresh

    # Just send the email from already-synced data (no fetch, no sync)
    python scripts/weekly_rankings_email.py --send-only --delay 0

    # Print email to stdout instead of sending
    python scripts/weekly_rankings_email.py --dry-run
"""

import argparse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import sys
from pathlib import Path

from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv

# Load .env before any tri_analysis imports (config.py also calls load_dotenv,
# but being explicit here ensures env is ready for email vars too)
load_dotenv(override=True)

# Allow running from repo root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))
from sqlalchemy import text

from tri_analysis.database import get_engine
from tri_analysis.ranking_import import import_rankings

# Categories to include in the email
EMAIL_CATEGORIES = {
    11: "Olympic Qualification — Men",
    12: "Olympic Qualification — Women",
    13: "World Rankings — Men",
    14: "World Rankings — Women",
    15: "WTCS Rankings — Men",
    16: "WTCS Rankings — Women",
}
WTCS_CATEGORIES = {15, 16}
OLYMPIC_CATEGORIES = {11, 12}

COUNTRY = "United States"


def fetch_us_rankings(engine) -> dict[int, list[dict]]:
    """
    Query the DB for current US athlete rankings across the four email categories.
    Returns dict keyed by ranking_cat_id, each value a list of row dicts.
    """
    query = text("""
        WITH current_dates AS (
            SELECT ranking_cat_id, MAX(retrieved_at) AS latest_date
            FROM athlete_rankings
            WHERE ranking_cat_id = ANY(:cat_ids)
            GROUP BY ranking_cat_id
        ),
        prev_dates AS (
            -- Pick the most recent snapshot at least ~one week before latest,
            -- so stray midweek snapshots don't collapse "change" to ~0.
            SELECT ar.ranking_cat_id,
                   MAX(ar.retrieved_at) AS prev_date
            FROM athlete_rankings ar
            JOIN current_dates cd ON ar.ranking_cat_id = cd.ranking_cat_id
            WHERE ar.retrieved_at <= cd.latest_date - INTERVAL '6 days'
            GROUP BY ar.ranking_cat_id
        ),
        current_ranks AS (
             SELECT DISTINCT ON (ar.ranking_cat_id, ar.athlete_id)
                   ar.ranking_cat_id,
                   ar.athlete_id,
                   ar.athlete_name,
                 a.country,
                   ar.rank_position,
                   ar.total_points,
                   ar.retrieved_at,
                   ar.events_current_period,
                   ar.events_previous_period
            FROM athlete_rankings ar
            JOIN current_dates cd
              ON ar.ranking_cat_id = cd.ranking_cat_id
             AND ar.retrieved_at   = cd.latest_date
            JOIN athlete a ON ar.athlete_id = a.athlete_id
            WHERE a.country = :country
             ORDER BY ar.ranking_cat_id, ar.athlete_id, ar.rank_position
        ),
        prev_ranks AS (
             SELECT DISTINCT ON (ar.ranking_cat_id, ar.athlete_id)
                   ar.ranking_cat_id,
                 ar.athlete_id,
                   ar.rank_position AS prev_rank_position
            FROM athlete_rankings ar
            JOIN prev_dates pd
              ON ar.ranking_cat_id = pd.ranking_cat_id
             AND ar.retrieved_at   = pd.prev_date
             ORDER BY ar.ranking_cat_id, ar.athlete_id, ar.rank_position
        )
        SELECT
            cr.ranking_cat_id,
            cr.athlete_id,
            cr.athlete_name,
             cr.country,
            cr.rank_position,
            cr.total_points,
            cr.retrieved_at,
            cr.events_current_period,
            cr.events_previous_period,
            pr.prev_rank_position
        FROM current_ranks cr
        LEFT JOIN prev_ranks pr
          ON cr.ranking_cat_id = pr.ranking_cat_id
                 AND cr.athlete_id     = pr.athlete_id
        ORDER BY cr.ranking_cat_id, cr.rank_position
    """)

    cat_ids = list(EMAIL_CATEGORIES.keys())
    rows = []
    with engine.connect() as conn:
        result = conn.execute(query, {"cat_ids": cat_ids, "country": COUNTRY})
        rows = result.mappings().all()

    # Group by category
    by_cat: dict[int, list[dict]] = {cat_id: [] for cat_id in EMAIL_CATEGORIES}
    for row in rows:
        by_cat[row["ranking_cat_id"]].append(dict(row))
    return by_cat


def fetch_and_store_breakdowns(engine, by_cat: dict[int, list[dict]]) -> None:
    """
    For each US athlete in each category, fetch the per-event ranking breakdown
    from the World Triathlon API and upsert to athlete_ranking_breakdown.
    """
    from tri_analysis.api_handling import fetch_athlete_ranking_breakdown

    today = date.today()
    upsert_sql = text("""
        INSERT INTO athlete_ranking_breakdown (
            athlete_id, ranking_cat_id, event_id, event_title,
            event_finish_date, points, period, position, included, retrieved_at
        ) VALUES (
            :athlete_id, :ranking_cat_id, :event_id, :event_title,
            :event_finish_date, :points, :period, :position, :included, :retrieved_at
        )
        ON CONFLICT (athlete_id, ranking_cat_id, event_id) DO UPDATE SET
            event_title       = EXCLUDED.event_title,
            event_finish_date = EXCLUDED.event_finish_date,
            points            = EXCLUDED.points,
            period            = EXCLUDED.period,
            position          = EXCLUDED.position,
            included          = EXCLUDED.included,
            retrieved_at      = EXCLUDED.retrieved_at
    """)

    total_stored = 0
    for cat_id, athletes in by_cat.items():
        for athlete in athletes:
            athlete_id = athlete.get("athlete_id")
            if not athlete_id:
                continue
            breakdown = fetch_athlete_ranking_breakdown(cat_id, athlete_id)
            if not breakdown:
                continue
            records = []
            for b in breakdown:
                event_finish_date = None
                if b.get("event_finish_date"):
                    try:
                        event_finish_date = pd.to_datetime(b["event_finish_date"]).date()
                    except Exception:
                        pass
                records.append({
                    "athlete_id":        athlete_id,
                    "ranking_cat_id":    cat_id,
                    "event_id":          b["event_id"],
                    "event_title":       b["event_title"],
                    "event_finish_date": event_finish_date,
                    "points":            b["points"],
                    "period":            b["period"],
                    "position":          b["position"],
                    "included":          b["included"],
                    "retrieved_at":      today,
                })
            if records:
                with engine.begin() as conn:
                    conn.execute(upsert_sql, records)
                total_stored += len(records)
    print(f"Stored {total_stored} ranking breakdown records.")


def store_all_breakdowns(engine) -> None:
    """
    Fetch and store per-event ranking breakdowns for ALL ranked athletes
    across the 4 email categories. Used to populate the public dashboard.
    ~1350 API calls, takes ~15 minutes with rate-limit delays.
    """
    import time
    from tri_analysis.api_handling import fetch_athlete_ranking_breakdown

    today = date.today()
    upsert_sql = text("""
        INSERT INTO athlete_ranking_breakdown (
            athlete_id, ranking_cat_id, event_id, event_title,
            event_finish_date, points, period, position, included, retrieved_at
        ) VALUES (
            :athlete_id, :ranking_cat_id, :event_id, :event_title,
            :event_finish_date, :points, :period, :position, :included, :retrieved_at
        )
        ON CONFLICT (athlete_id, ranking_cat_id, event_id) DO UPDATE SET
            event_title       = EXCLUDED.event_title,
            event_finish_date = EXCLUDED.event_finish_date,
            points            = EXCLUDED.points,
            period            = EXCLUDED.period,
            position          = EXCLUDED.position,
            included          = EXCLUDED.included,
            retrieved_at      = EXCLUDED.retrieved_at
    """)

    # Get all athlete_ids from the latest snapshot per category
    athlete_query = text("""
        WITH current_dates AS (
            SELECT ranking_cat_id, MAX(retrieved_at) AS latest_date
            FROM athlete_rankings WHERE ranking_cat_id = ANY(:cat_ids)
            GROUP BY ranking_cat_id
        )
        SELECT DISTINCT ar.ranking_cat_id, ar.athlete_id
        FROM athlete_rankings ar
        JOIN current_dates cd
          ON ar.ranking_cat_id = cd.ranking_cat_id AND ar.retrieved_at = cd.latest_date
        ORDER BY ar.ranking_cat_id, ar.athlete_id
    """)

    cat_ids = list(EMAIL_CATEGORIES.keys())
    with engine.connect() as conn:
        pairs = conn.execute(athlete_query, {"cat_ids": cat_ids}).fetchall()

    print(f"Fetching breakdowns for {len(pairs)} athlete-category pairs...")
    total_stored = 0
    for i, (cat_id, athlete_id) in enumerate(pairs):
        breakdown = fetch_athlete_ranking_breakdown(cat_id, athlete_id)
        if breakdown:
            records = []
            for b in breakdown:
                event_finish_date = None
                if b.get("event_finish_date"):
                    try:
                        event_finish_date = pd.to_datetime(b["event_finish_date"]).date()
                    except Exception:
                        pass
                records.append({
                    "athlete_id":        athlete_id,
                    "ranking_cat_id":    cat_id,
                    "event_id":          b["event_id"],
                    "event_title":       b["event_title"],
                    "event_finish_date": event_finish_date,
                    "points":            b["points"],
                    "period":            b["period"],
                    "position":          b["position"],
                    "included":          b["included"],
                    "retrieved_at":      today,
                })
            if records:
                with engine.begin() as conn:
                    conn.execute(upsert_sql, records)
                total_stored += len(records)
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(pairs)} athletes processed ({total_stored} records stored)")
        time.sleep(0.3)

    print(f"Stored {total_stored} total breakdown records for all athletes.")


def fetch_at_risk(
    engine,
    by_cat: dict[int, list[dict]],
    breakdown_engine=None,
) -> dict[tuple, dict]:
    """
    Simulate per-athlete World Triathlon score under the official drop-off model
    (events get 1/3 points at 12 months, 0 points at 24 months) and report how
    much score is at risk over the next 2 weeks, 1 month, and 3 months.

    Mirrors the dashboard logic in app/webapp/app.py so the email and the live
    rankings page show the same numbers.

    breakdown_engine: if provided, use this engine for the breakdown query
        (e.g. the TRIATHLON_DATABASE_URL Supabase engine).

    Returns dict keyed by (athlete_id, ranking_cat_id) ->
        {"at_risk_2w": float, "at_risk_1m": float, "at_risk_3m": float,
         "drop_2w": int, "drop_1m": int, "drop_3m": int}
    """
    bd_engine = breakdown_engine if breakdown_engine is not None else engine
    athlete_ids = [
        a["athlete_id"]
        for athletes in by_cat.values()
        for a in athletes
        if a.get("athlete_id")
    ]
    cat_ids = list(by_cat.keys())
    if not athlete_ids or not cat_ids:
        return {}

    # Pull all events for the target athletes, restricted to the latest
    # breakdown snapshot per ranking category (matches dashboard behavior).
    query = text("""
        WITH latest_bd AS (
            SELECT ranking_cat_id, MAX(retrieved_at) AS latest
            FROM athlete_ranking_breakdown
            WHERE ranking_cat_id = ANY(:cat_ids)
            GROUP BY ranking_cat_id
        )
        SELECT abd.athlete_id, abd.ranking_cat_id, abd.event_id,
               abd.event_finish_date, abd.points, abd.period, abd.included
        FROM athlete_ranking_breakdown abd
        JOIN latest_bd lb
          ON abd.ranking_cat_id = lb.ranking_cat_id
         AND abd.retrieved_at   = lb.latest
        WHERE abd.athlete_id = ANY(:athlete_ids)
    """)

    today = date.today()
    cutoff_2w = today + timedelta(days=14)
    cutoff_1m = today + timedelta(days=30)
    cutoff_3m = today + timedelta(days=91)

    def _event_value_at(ev: dict, as_of: date) -> float:
        pts = float(ev.get("points") or 0.0)
        efd = ev.get("event_finish_date")
        period = ev.get("period", 1)
        if not efd:
            return pts / 3.0 if period == 2 else pts
        one_year = efd + timedelta(days=365)
        two_years = efd + timedelta(days=730)
        if as_of >= two_years:
            return 0.0
        if as_of >= one_year:
            return pts / 3.0
        return pts

    def _score_at(events: list[dict], as_of: date, event_cap: int) -> float:
        values = [_event_value_at(ev, as_of) for ev in events]
        positive_values = [v for v in values if v > 0]
        if not positive_values:
            return 0.0
        positive_values.sort(reverse=True)
        if event_cap <= 0:
            return sum(positive_values)
        return sum(positive_values[:event_cap])

    with bd_engine.connect() as conn:
        rows = conn.execute(
            query, {"cat_ids": cat_ids, "athlete_ids": athlete_ids}
        ).mappings().all()

    # Group events by (athlete_id, cat_id), dedupe by event_id within group.
    athlete_events: dict[tuple, list[dict]] = {}
    seen_events: set[tuple] = set()
    for row in rows:
        key = (row["athlete_id"], row["ranking_cat_id"])
        event_id = row.get("event_id")
        if event_id is not None:
            dedupe_key = (key, int(event_id))
            if dedupe_key in seen_events:
                continue
            seen_events.add(dedupe_key)
        athlete_events.setdefault(key, []).append({
            "event_finish_date": row.get("event_finish_date"),
            "points": float(row.get("points") or 0.0),
            "period": row.get("period", 1),
            "included": row.get("included", True),
        })

    at_risk: dict[tuple, dict] = {}
    for key, events in athlete_events.items():
        included_count = sum(1 for ev in events if ev.get("included", True))
        event_cap = included_count if included_count > 0 else len(events)
        current_score = _score_at(events, today, event_cap)
        risk_2w = max(0.0, current_score - _score_at(events, cutoff_2w, event_cap))
        risk_1m = max(0.0, current_score - _score_at(events, cutoff_1m, event_cap))
        risk_3m = max(0.0, current_score - _score_at(events, cutoff_3m, event_cap))
        at_risk[key] = {
            "at_risk_2w": risk_2w,
            "at_risk_1m": risk_1m,
            "at_risk_3m": risk_3m,
        }

    # --- Compute projected rank drops per window ---
    # For each category, load the full ranking list and simulate point loss.
    full_ranking_query = text("""
        WITH current_dates AS (
            SELECT ranking_cat_id, MAX(retrieved_at) AS latest_date
            FROM athlete_rankings WHERE ranking_cat_id = ANY(:cat_ids)
            GROUP BY ranking_cat_id
        )
        SELECT ar.ranking_cat_id, ar.athlete_id, ar.total_points
        FROM athlete_rankings ar
        JOIN current_dates cd
          ON ar.ranking_cat_id = cd.ranking_cat_id AND ar.retrieved_at = cd.latest_date
        ORDER BY ar.ranking_cat_id, ar.total_points DESC
    """)
    cat_ids = list(by_cat.keys())
    with engine.connect() as conn:
        full_rows = conn.execute(full_ranking_query, {"cat_ids": cat_ids}).mappings().all()

    # Group all athletes' points by category
    cat_points: dict[int, list[float]] = {}
    cat_athlete_pts: dict[tuple, float] = {}
    for r in full_rows:
        cid = r["ranking_cat_id"]
        cat_points.setdefault(cid, []).append(float(r["total_points"] or 0))
        cat_athlete_pts[(r["athlete_id"], cid)] = float(r["total_points"] or 0)

    for key, risk in at_risk.items():
        athlete_id, cat_id = key
        current_pts = cat_athlete_pts.get(key)
        all_pts = cat_points.get(cat_id, [])
        if current_pts is None or not all_pts:
            risk["drop_2w"] = 0
            risk["drop_1m"] = 0
            risk["drop_3m"] = 0
            continue
        current_rank = sum(1 for p in all_pts if p > current_pts) + 1
        for suffix in ("2w", "1m", "3m"):
            lost = risk[f"at_risk_{suffix}"]
            if lost:
                new_pts = current_pts - lost
                new_rank = sum(1 for p in all_pts if p > new_pts) + 1
                risk[f"drop_{suffix}"] = new_rank - current_rank
            else:
                risk[f"drop_{suffix}"] = 0

    return at_risk


def rank_change_html(current: int, prev: int | None) -> str:
    if prev is None:
        return '<span style="color:#888">&#8212;</span>'
    delta = prev - current  # positive = moved up
    if delta > 0:
        return f'<span style="color:#22a744">&#9650; {delta}</span>'
    elif delta < 0:
        return f'<span style="color:#d93025">&#9660; {abs(delta)}</span>'
    return '<span style="color:#888">&#8212;</span>'


def build_html(by_cat: dict[int, list[dict]], at_risk: dict[tuple, dict] | None = None) -> str:
    if at_risk is None:
        at_risk = {}
    sections_html = ""
    for cat_id, label in EMAIL_CATEGORIES.items():
        athletes = by_cat.get(cat_id, [])
        if not athletes:
            continue

        is_wtcs = cat_id in WTCS_CATEGORIES
        snapshot_date = athletes[0]["retrieved_at"].strftime("%B %d, %Y") if athletes else "—"

        rows_html = ""
        for row in athletes:
            change = rank_change_html(row["rank_position"], row["prev_rank_position"])
            pts = f"{int(row['total_points']):,}" if row["total_points"] else "—"

            if is_wtcs:
                curr = row["events_current_period"] if row["events_current_period"] is not None else 0
                prev_ev = row["events_previous_period"] if row["events_previous_period"] is not None else 0
                total_events = curr + prev_ev if (curr or prev_ev) else "—"
                country = row.get("country") or "—"
                rows_html += f"""
            <tr>
                <td style="padding:6px 12px;text-align:center;font-weight:bold">{row['rank_position']}</td>
                <td style="padding:6px 12px">{row['athlete_name']}</td>
                <td style="padding:6px 12px">{country}</td>
                <td style="padding:6px 12px;text-align:right">{pts}</td>
                <td style="padding:6px 12px;text-align:center">{total_events}</td>
                <td style="padding:6px 12px;text-align:center">{change}</td>
            </tr>"""
            else:
                curr = row["events_current_period"] if row["events_current_period"] is not None else "—"
                prev = row["events_previous_period"] if row["events_previous_period"] is not None else "—"
                country = row.get("country") or "—"
                risk = at_risk.get((row.get("athlete_id"), cat_id), {})
                r2w = f"{int(risk['at_risk_2w']):,}" if risk.get("at_risk_2w") else "—"
                r1m = f"{int(risk['at_risk_1m']):,}" if risk.get("at_risk_1m") else "—"
                r3m = f"{int(risk['at_risk_3m']):,}" if risk.get("at_risk_3m") else "—"
                d2w = f"&#9660; {risk['drop_2w']}" if risk.get("drop_2w") else "—"
                d1m = f"&#9660; {risk['drop_1m']}" if risk.get("drop_1m") else "—"
                d3m = f"&#9660; {risk['drop_3m']}" if risk.get("drop_3m") else "—"
                rows_html += f"""
            <tr>
                <td style="padding:6px 12px;text-align:center;font-weight:bold">{row['rank_position']}</td>
                <td style="padding:6px 12px">{row['athlete_name']}</td>
                <td style="padding:6px 12px">{country}</td>
                <td style="padding:6px 12px;text-align:right">{pts}</td>
                <td style="padding:6px 12px;text-align:center">{curr}</td>
                <td style="padding:6px 12px;text-align:center">{prev}</td>
                <td style="padding:6px 12px;text-align:center">{change}</td>
                <td style="padding:6px 12px;text-align:right;color:#d93025">{r2w}</td>
                <td style="padding:6px 12px;text-align:right;color:#d93025">{r1m}</td>
                <td style="padding:6px 12px;text-align:right;color:#d93025">{r3m}</td>
                <td style="padding:6px 12px;text-align:center;color:#d93025">{d2w}</td>
                <td style="padding:6px 12px;text-align:center;color:#d93025">{d1m}</td>
                <td style="padding:6px 12px;text-align:center;color:#d93025">{d3m}</td>
            </tr>"""

        if is_wtcs:
            header_html = """
                <tr style="background:#c8102e;color:white">
                    <th style="padding:8px 12px">Rank</th>
                    <th style="padding:8px 12px;text-align:left">Athlete</th>
                    <th style="padding:8px 12px;text-align:left">Country</th>
                    <th style="padding:8px 12px;text-align:right">Points</th>
                    <th style="padding:8px 12px">Events</th>
                    <th style="padding:8px 12px">Change</th>
                </tr>"""
        else:
            header_html = """
                <tr style="background:#c8102e;color:white">
                    <th style="padding:8px 12px">Rank</th>
                    <th style="padding:8px 12px;text-align:left">Athlete</th>
                    <th style="padding:8px 12px;text-align:left">Country</th>
                    <th style="padding:8px 12px;text-align:right">Points</th>
                    <th style="padding:8px 12px" title="Events scored in last 0-1 years">Curr Events</th>
                    <th style="padding:8px 12px" title="Events scored in last 1-2 years">Prev Events</th>
                    <th style="padding:8px 12px">Change</th>
                    <th style="padding:8px 12px;text-align:right" title="Points expiring in next 2 weeks">Risk 2W</th>
                    <th style="padding:8px 12px;text-align:right" title="Points expiring in next 1 month">Risk 1M</th>
                    <th style="padding:8px 12px;text-align:right" title="Points expiring in next 3 months">Risk 3M</th>
                    <th style="padding:8px 12px" title="Projected rank drop in 2 weeks">Drop 2W</th>
                    <th style="padding:8px 12px" title="Projected rank drop in 1 month">Drop 1M</th>
                    <th style="padding:8px 12px" title="Projected rank drop in 3 months">Drop 3M</th>
                </tr>"""

        sections_html += f"""
        <h2 style="margin:28px 0 4px;font-family:sans-serif;font-size:16px;color:#1a1a1a;
                   border-bottom:2px solid #c8102e;padding-bottom:6px">
            {label}
        </h2>
        <p style="margin:2px 0 8px;font-size:12px;color:#666;font-family:sans-serif">
            Rankings as of {snapshot_date}
        </p>
        <table style="border-collapse:collapse;width:100%;font-family:sans-serif;font-size:14px">
            <thead>{header_html}
            </thead>
            <tbody>{rows_html}
            </tbody>
        </table>"""

    if not sections_html:
        sections_html = "<p style='font-family:sans-serif'>No US athlete data found in database.</p>"

    dashboard_url = os.getenv("RANKINGS_DASHBOARD_URL", "")
    dashboard_top = ""
    dashboard_bottom = ""
    if dashboard_url:
        dashboard_top = f"""
        <div style="background:#f0f4f8;border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;margin:12px 0 20px;font-family:sans-serif">
            <p style="margin:0 0 8px;font-size:14px;color:#1a1a1a">
                <strong>&#128204; Interactive Dashboard Available</strong>
            </p>
            <p style="margin:0 0 10px;font-size:13px;color:#555;line-height:1.4">
                Search by athlete or country, view event breakdowns, and track weekly ranking changes.
            </p>
            <a href="{dashboard_url}/rankings" style="background:#c8102e;color:white;padding:10px 18px;
               border-radius:6px;text-decoration:none;font-size:14px;font-weight:bold;display:inline-block">
               Open Rankings Dashboard &#8594;</a>
        </div>"""
        dashboard_bottom = f"""
        <div style="text-align:center;margin:24px 0 8px;font-family:sans-serif">
            <a href="{dashboard_url}/rankings" style="background:#c8102e;color:white;padding:10px 24px;
               border-radius:6px;text-decoration:none;font-size:14px;font-weight:bold;display:inline-block">
               View Full Dashboard &#8594;</a>
            <p style="margin:8px 0 0;font-size:12px;color:#888">
                Filter by country, search athletes, and click any row for event-by-event breakdowns.
            </p>
        </div>"""

    return f"""
    <div style="max-width:960px;margin:auto">
        <h1 style="font-family:sans-serif;color:#c8102e;font-size:22px;margin-bottom:4px">
            USA Triathlon — Weekly Rankings
        </h1>
        {dashboard_top}
        {sections_html}
        {dashboard_bottom}
        <p style="font-size:11px;color:#aaa;font-family:sans-serif;margin-top:24px">
            Source: World Triathlon API &nbsp;|&nbsp; Generated automatically
        </p>
    </div>
    """


def send_email(subject: str, html_body: str, recipients: list[str] | None = None) -> None:
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")

    if recipients is None:
        recipients_raw = os.getenv("RANKINGS_EMAIL_RECIPIENTS", "")
        recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    if not smtp_user or not smtp_pass:
        print("[Email] SMTP_USER or SMTP_PASSWORD not set — printing email to stdout instead.")
        print(f"Subject: {subject}")
        print(html_body)
        return

    if not recipients:
        print("[Email] No recipients — skipping send.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipients, msg.as_string())

    print(f"[Email] Sent to {recipients}")


def _insert_batches(supa_engine, insert_sql: str, rows: list, table: str, batch_size: int = 500) -> None:
    """Insert rows in batches with retry. No conflict handling — caller ensures no duplicates."""
    import time as _time
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        for attempt in range(1, 4):
            try:
                with supa_engine.begin() as conn:
                    conn.execute(text(insert_sql), [dict(r._mapping) for r in batch])
                break
            except Exception as e:
                if attempt == 3:
                    print(f"[Supabase] {table} batch {i//batch_size} failed: {e}")
                    raise
                _time.sleep(2 ** attempt)


def prune_stale_breakdowns(engine, label: str = "DB") -> None:
    """
    Keep only each athlete's most recent breakdown snapshot per ranking category.

    When an event passes the 2-year window the World Triathlon API stops returning
    it, so a fresh fetch never overwrites the old row — it lingers as an "orphan"
    and keeps showing up (e.g. expired events still listed as Counted). The
    breakdown table is current-state only (rank history lives in athlete_rankings),
    so dropping anything older than the latest snapshot per (athlete, category)
    removes orphans without losing anything the dashboard needs.
    """
    delete_sql = text("""
        DELETE FROM athlete_ranking_breakdown b
        USING (
            SELECT athlete_id, ranking_cat_id, MAX(retrieved_at) AS latest
            FROM athlete_ranking_breakdown
            GROUP BY athlete_id, ranking_cat_id
        ) m
        WHERE b.athlete_id = m.athlete_id
          AND b.ranking_cat_id = m.ranking_cat_id
          AND b.retrieved_at < m.latest
    """)
    with engine.begin() as conn:
        result = conn.execute(delete_sql)
        deleted = result.rowcount or 0
    print(f"[{label}] Pruned {deleted:,} stale breakdown rows (orphaned expired events).")


def sync_to_supabase(local_engine) -> None:
    """Append new retrieved_at snapshots from local DB → Supabase (athlete_rankings + breakdown)."""
    supa_url = os.getenv("TRIATHLON_DATABASE_URL", "")
    if not supa_url:
        print("[Supabase] TRIATHLON_DATABASE_URL not set — skipping sync.")
        return

    from sqlalchemy import create_engine as _create_engine
    supa = _create_engine(supa_url, pool_pre_ping=True, pool_size=1, max_overflow=0)

    # --- athlete_rankings ---
    print("[Supabase] Syncing athlete_rankings...")
    with supa.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS athlete_rankings (
                athlete_id            INTEGER,
                athlete_name          TEXT,
                ranking_cat_name      TEXT,
                ranking_cat_id        INTEGER,
                rank_position         INTEGER,
                total_points          DOUBLE PRECISION,
                year                  INTEGER,
                retrieved_at          TIMESTAMP,
                events_current_period INTEGER,
                events_previous_period INTEGER
            )
        """))

    # Find dates already on Supabase — only upload snapshots not yet there
    with supa.connect() as conn:
        existing_dates = {
            r[0] for r in conn.execute(text(
                "SELECT DISTINCT retrieved_at::date FROM athlete_rankings"
            )).fetchall()
        }

    with local_engine.connect() as conn:
        local_dates = {
            r[0] for r in conn.execute(text(
                "SELECT DISTINCT retrieved_at::date FROM athlete_rankings"
            )).fetchall()
        }

    new_dates = local_dates - existing_dates
    if not new_dates:
        print("[Supabase] athlete_rankings: already up to date.")
    else:
        date_list = ", ".join(f"'{d}'" for d in sorted(new_dates))
        with local_engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT athlete_id, athlete_name, ranking_cat_name, ranking_cat_id,
                       rank_position, total_points, year, retrieved_at,
                       events_current_period, events_previous_period
                FROM athlete_rankings
                WHERE retrieved_at::date IN ({date_list})
            """)).fetchall()
        _insert_batches(supa, """
            INSERT INTO athlete_rankings
                (athlete_id, athlete_name, ranking_cat_name, ranking_cat_id,
                 rank_position, total_points, year, retrieved_at,
                 events_current_period, events_previous_period)
            VALUES
                (:athlete_id, :athlete_name, :ranking_cat_name, :ranking_cat_id,
                 :rank_position, :total_points, :year, :retrieved_at,
                 :events_current_period, :events_previous_period)
            ON CONFLICT (athlete_name, ranking_cat_name, year, retrieved_at) DO UPDATE SET
                athlete_id             = EXCLUDED.athlete_id,
                ranking_cat_id         = EXCLUDED.ranking_cat_id,
                rank_position          = EXCLUDED.rank_position,
                total_points           = EXCLUDED.total_points,
                events_current_period  = EXCLUDED.events_current_period,
                events_previous_period = EXCLUDED.events_previous_period
        """, rows, "athlete_rankings")
        print(f"[Supabase] athlete_rankings: {len(rows):,} rows added ({len(new_dates)} new date(s)).")

    # --- athlete_ranking_breakdown ---
    print("[Supabase] Syncing athlete_ranking_breakdown...")
    with supa.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS athlete_ranking_breakdown (
                athlete_id        INTEGER,
                ranking_cat_id    INTEGER,
                event_id          INTEGER,
                event_title       TEXT,
                event_finish_date DATE,
                points            DOUBLE PRECISION,
                period            INTEGER,
                position          INTEGER,
                retrieved_at      TIMESTAMP,
                included          BOOLEAN
            )
        """))

    # Prune local orphans first so they don't get pushed to Supabase.
    prune_stale_breakdowns(local_engine, label="Local")

    # Row-based sync: push the freshest row per (athlete_id, ranking_cat_id,
    # event_id) from local. The ON CONFLICT upsert makes this idempotent.
    # We deliberately do NOT date-diff here: the old logic only synced brand-new
    # retrieved_at dates, so a full-roster refresh that landed on a date already
    # partially present on Supabase (e.g. after the US-only sync) was silently
    # skipped, leaving non-US athletes frozen at an old snapshot.
    with local_engine.connect() as conn:
        rows2 = conn.execute(text("""
            SELECT DISTINCT ON (athlete_id, ranking_cat_id, event_id)
                   athlete_id, ranking_cat_id, event_id, event_title,
                   event_finish_date, points, period, position, retrieved_at, included
            FROM athlete_ranking_breakdown
            ORDER BY athlete_id, ranking_cat_id, event_id, retrieved_at DESC
        """)).fetchall()

    try:
        if not rows2:
            print("[Supabase] athlete_ranking_breakdown: nothing to sync.")
        else:
            _insert_batches(supa, """
                INSERT INTO athlete_ranking_breakdown
                    (athlete_id, ranking_cat_id, event_id, event_title,
                     event_finish_date, points, period, position, retrieved_at, included)
                VALUES
                    (:athlete_id, :ranking_cat_id, :event_id, :event_title,
                     :event_finish_date, :points, :period, :position, :retrieved_at, :included)
                ON CONFLICT (athlete_id, ranking_cat_id, event_id) DO UPDATE SET
                    event_title       = EXCLUDED.event_title,
                    event_finish_date = EXCLUDED.event_finish_date,
                    points            = EXCLUDED.points,
                    period            = EXCLUDED.period,
                    position          = EXCLUDED.position,
                    included          = EXCLUDED.included,
                    retrieved_at      = EXCLUDED.retrieved_at
            """, rows2, "athlete_ranking_breakdown")
            print(f"[Supabase] athlete_ranking_breakdown: {len(rows2):,} rows upserted (row-based).")
    finally:
        # Always prune Supabase orphans, even if the upsert failed partway —
        # the upsert only pushes local's clean newest-per-event rows, so it can
        # never create orphans, but a skipped prune leaves expired events behind.
        prune_stale_breakdowns(supa, label="Supabase")


def update_computed_rankings(local_engine) -> None:
    """Compute this week's rankings snapshot and sync to Supabase (no full recompute)."""
    from datetime import date as _date, timedelta as _timedelta
    import time as _time
    from tri_analysis.ranking_points import simulate_weekly_rankings

    # Find the most recent Sunday
    today = _date.today()
    days_since_sunday = today.isoweekday() % 7  # Mon=1..Sun=0 → Sun gives 0
    last_sunday = today - _timedelta(days=days_since_sunday)

    print(f"[Rankings] Computing snapshot for {last_sunday}...")
    n = simulate_weekly_rankings(local_engine, last_sunday, last_sunday)
    print(f"[Rankings] {n:,} rows written locally.")

    supa_url = os.getenv("TRIATHLON_DATABASE_URL", "")
    if not supa_url:
        print("[Supabase] TRIATHLON_DATABASE_URL not set — skipping sync.")
        return

    from sqlalchemy import create_engine as _create_engine
    supa = _create_engine(supa_url, pool_pre_ping=True, pool_size=1, max_overflow=0)

    with local_engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ranking_date, ranking_cat_id, athlete_id, rank_position,
                   total_points, events_current, events_previous
            FROM computed_weekly_rankings
            WHERE ranking_date = :d
            ORDER BY ranking_cat_id, rank_position
        """), {"d": last_sunday}).fetchall()

    if not rows:
        print("[Supabase] No rows to sync.")
        return

    print(f"[Supabase] Uploading {len(rows):,} rows for {last_sunday}...")
    BATCH = 500
    inserted = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        for attempt in range(1, 4):
            try:
                with supa.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO computed_weekly_rankings
                            (ranking_date, ranking_cat_id, athlete_id, rank_position,
                             total_points, events_current, events_previous)
                        VALUES
                            (:ranking_date, :ranking_cat_id, :athlete_id, :rank_position,
                             :total_points, :events_current, :events_previous)
                        ON CONFLICT (ranking_date, ranking_cat_id, athlete_id) DO UPDATE SET
                            rank_position   = EXCLUDED.rank_position,
                            total_points    = EXCLUDED.total_points,
                            events_current  = EXCLUDED.events_current,
                            events_previous = EXCLUDED.events_previous
                    """), [
                        {"ranking_date": r[0], "ranking_cat_id": r[1], "athlete_id": r[2],
                         "rank_position": r[3], "total_points": r[4], "events_current": r[5],
                         "events_previous": r[6]}
                        for r in batch
                    ])
                break
            except Exception as e:
                if attempt == 3:
                    print(f"[Supabase] computed_weekly_rankings batch failed: {e}")
                    raise
                _time.sleep(2 ** attempt)
        inserted += len(batch)

    print(f"[Supabase] computed_weekly_rankings: {inserted:,} rows synced.")


def main(refresh: bool = True, dry_run: bool = False, all_breakdowns: bool = True,
         preview_delay_minutes: int = 60, sync_only: bool = False,
         send_only: bool = False) -> None:
    import time as _time

    # --send-only: build and send the email straight from existing local data.
    # No API fetch, no breakdown refresh, no Supabase sync. Use when everything
    # has already been synced and you just want to (re)send the email.
    if send_only:
        refresh = False
        all_breakdowns = False
        print("Send-only mode: using existing DB data (no fetch, no sync).")

    # --sync-only: refresh data and push to Supabase, no email. Doubles as a
    # repair command: with all_breakdowns (default) it re-fetches every ranked
    # athlete's breakdown from the API, fixing stale non-US dashboard data.
    if sync_only:
        if refresh:
            print("Fetching rankings from World Triathlon API...")
            import_rankings(category_ids=list(EMAIL_CATEGORIES.keys()))
        engine = get_engine()
        if all_breakdowns:
            store_all_breakdowns(engine)
        sync_to_supabase(engine)
        update_computed_rankings(engine)
        return

    start_time = _time.time()

    # Step 1: refresh rankings from World Triathlon API
    if refresh:
        print("Fetching rankings from World Triathlon API...")
        import_rankings(category_ids=list(EMAIL_CATEGORIES.keys()))
    else:
        print("Skipping API refresh (--no-refresh).")

    engine = get_engine()

    # Step 1b: fetch breakdowns for ALL ranked athletes so the dashboard stays
    # current for every athlete, not just the US set. Skipped for --dry-run
    # (which only previews the email HTML) to keep that fast.
    did_all_breakdowns = False
    if all_breakdowns and not dry_run:
        store_all_breakdowns(engine)
        did_all_breakdowns = True

    # Step 2: query DB for US athletes
    by_cat = fetch_us_rankings(engine)

    total = sum(len(v) for v in by_cat.values())
    if total == 0:
        print("No US athletes found in rankings. Check athlete table country values.")
        return

    print(f"Found {total} US athlete ranking entries across {len(EMAIL_CATEGORIES)} categories.")
    for cat_id, label in EMAIL_CATEGORIES.items():
        print(f"  {label}: {len(by_cat.get(cat_id, []))} athletes")

    # Step 3: fetch per-event breakdown and compute points at risk.
    # If store_all_breakdowns already ran, the US athletes are already covered,
    # so skip the redundant US-only re-fetch. Send-only skips fetching entirely.
    if not did_all_breakdowns and not send_only:
        print("Fetching per-athlete ranking breakdowns...")
        fetch_and_store_breakdowns(engine, by_cat)

    # Read at-risk from the local engine: this week's rankings + breakdowns
    # were just refreshed above, but the Supabase sync hasn't run yet. Using
    # the local DB here ensures the email matches what the dashboard will
    # show once sync_to_supabase() completes below.
    at_risk = fetch_at_risk(engine, by_cat)

    # Step 4: build email
    snapshot_date = None
    for athletes in by_cat.values():
        if athletes:
            snapshot_date = athletes[0]["retrieved_at"]
            break
    subject = f"USA Triathlon Rankings — {snapshot_date.strftime('%B %d, %Y') if snapshot_date else 'Weekly Update'}"
    html_body = build_html(by_cat, at_risk)

    if dry_run:
        print(f"\nSubject: {subject}\n")
        print(html_body)
        return

    # Step 5: send preview to just yourself immediately
    preview_addr = os.getenv("SMTP_USER", "")
    if preview_addr:
        print(f"\n[Email] Sending preview to {preview_addr}...")
        send_email(f"[PREVIEW] {subject}", html_body, recipients=[preview_addr])
    else:
        print("[Email] SMTP_USER not set — skipping preview send.")

    # Step 6: sync local data to Supabase while we wait (skipped in send-only).
    if not send_only:
        print("\n[Supabase] Starting data sync...")
        try:
            sync_to_supabase(engine)
        except Exception as e:
            print(f"[Supabase] athlete_rankings sync failed (non-fatal): {e}")

        try:
            update_computed_rankings(engine)
        except Exception as e:
            print(f"[Supabase] computed_weekly_rankings sync failed (non-fatal): {e}")

    # Step 7: wait until preview_delay_minutes has elapsed from start
    elapsed = _time.time() - start_time
    wait_secs = max(0, preview_delay_minutes * 60 - elapsed)
    if wait_secs > 0:
        mins = int(wait_secs // 60)
        secs = int(wait_secs % 60)
        print(f"\n[Email] Waiting {mins}m {secs}s before sending to all recipients...")
        _time.sleep(wait_secs)

    # Step 8: send to the full contact list
    print("[Email] Sending to all recipients...")
    send_email(subject, html_body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send weekly US triathlon rankings email.")
    parser.add_argument("--no-refresh", action="store_true", help="Skip API fetch, use existing DB data.")
    parser.add_argument("--dry-run", action="store_true", help="Print email to stdout instead of sending.")
    parser.add_argument("--all-breakdowns", action="store_true",
                        help="(Default behavior; kept for back-compat) Fetch breakdowns for ALL athletes.")
    parser.add_argument("--us-breakdowns-only", action="store_true",
                        help="Only refresh US-athlete breakdowns (fast). Dashboard non-US athletes will go stale.")
    parser.add_argument("--delay", type=int, default=60, metavar="MINUTES",
                        help="Minutes to wait between preview and full send (default: 60).")
    parser.add_argument("--sync-only", action="store_true", help="Push local data to Supabase without sending any email.")
    parser.add_argument("--send-only", action="store_true",
                        help="Build and send the email from existing DB data only — no fetch, no Supabase sync.")
    args = parser.parse_args()

    main(refresh=not args.no_refresh, dry_run=args.dry_run,
         all_breakdowns=not args.us_breakdowns_only,
         preview_delay_minutes=args.delay, sync_only=args.sync_only,
         send_only=args.send_only)
