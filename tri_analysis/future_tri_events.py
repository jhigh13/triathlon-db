"""Fetch upcoming World Triathlon events and compute nomination dates.

Outputs a table with: event_id, event_name, event_date, event_venue, category_names, nomination_date (Tuesday 30 days prior).

Usage: Run as a script. It will:
- Fetch events from today forward filtered by category IDs provided.
- Compute nomination Tuesday for each event.
- Print a dataframe preview and save CSV to tri_analysis/outputs/future_tri_events.csv

Environment/Config:
- Uses API endpoints and headers from tri_analysis.config
- CATEGORY_IDS can be overridden via env var CATEGORY_IDS; defaults include the requested set.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import List
import pandas as pd
import requests

try:
    # Prefer explicit module path
    from tri_analysis.config import HEADERS, EVENT_LISTING_URL, CATEGORY_IDS
except Exception:  # fallback when executed within tri_analysis as CWD
    from config import HEADERS, EVENT_LISTING_URL, CATEGORY_IDS


# Default categories per user request: Continental Cups, World Cups, WTCS, Para Series
DEFAULT_CATEGORY_IDS = "340|341|342|623|352|624|348|349|449|350"


def nomination_tuesday(event_dt: date) -> date:
    """Return the Tuesday that is 30 days prior to the event date, rounding BACK to Tuesday.

    Steps: event_date - 30 days => if that day is Tuesday keep it, else go back to the previous Tuesday.
    """
    offset = event_dt - timedelta(days=30)
    # Python weekday: Monday=0 ... Sunday=6; Tuesday=1
    days_back = (offset.weekday() - 1) % 7  # how many days to go back to reach Tuesday
    return offset - timedelta(days=days_back)


def fetch_upcoming_events(category_ids: str) -> pd.DataFrame:
    """Fetch upcoming events (today -> +1 year) filtered by category IDs.

    Returns a DataFrame with event basics. Handles pagination.
    """
    today = date.today().isoformat()
    one_year = (date.today() + timedelta(days=365)).isoformat()
    params = {
        "per_page": 500,
        "order": "asc",
        "start_date": today,
        "end_date": one_year,
        "category_id": category_ids or DEFAULT_CATEGORY_IDS,
    }
    rows: List[dict] = []
    page = 1
    while True:
        params["page"] = page
        resp = requests.get(EVENT_LISTING_URL, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            break
        for ev in data:
            rows.append({
                "event_id": ev.get("event_id"),
                "event_name": ev.get("event_title"),
                "event_date": ev.get("event_date"),
                "event_venue": ev.get("event_venue"),
                "event_country": ev.get("event_country"),
                "event_categories": ", ".join([c.get("cat_name") for c in (ev.get("event_categories") or []) if c.get("cat_name")]),
            })
        if not resp.json().get("next_page_url"):
            break
        page += 1
    df = pd.DataFrame(rows)
    if not df.empty:
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    return df


def build_future_calendar(category_ids: str = None) -> pd.DataFrame:
    cats = category_ids or os.getenv("CATEGORY_IDS", DEFAULT_CATEGORY_IDS)
    events = fetch_upcoming_events(cats)
    if events.empty:
        return events
    events["nomination_date"] = events["event_date"].apply(nomination_tuesday)
    events.sort_values(["event_date", "event_name"], inplace=True)
    return events


def main():
    df = build_future_calendar()
    if df.empty:
        print("No upcoming events found for the specified categories.")
        return
    print(df[["event_name", "event_date", "nomination_date", "event_venue", "event_country", "event_categories"]].to_string(index=False))
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "future_tri_events.csv")
    tmp_path = os.path.join(out_dir, "future_tri_events.tmp.csv")
    try:
        df.to_csv(tmp_path, index=False)
        # Replace existing file atomically if possible
        try:
            os.replace(tmp_path, out_path)
        except PermissionError:
            # Fallback to timestamped file when the target is locked (e.g., opened in Excel)
            ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
            fallback = os.path.join(out_dir, f"future_tri_events_{ts}.csv")
            os.replace(tmp_path, fallback)
            print(f"Destination locked; wrote fallback: {fallback}")
        else:
            print(f"Saved: {out_path}")
    finally:
        # Ensure tmp removed if still present
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
