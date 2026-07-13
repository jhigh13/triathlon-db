#!/usr/bin/env python
"""
medal_contention.py -- How many athletes are actually medal-capable?

Answers: over the last N years, how many distinct athletes have won a medal
(podium, positions 1-3) at the sport's biggest races, and how are they
distributed by country? "Medalled at a WTCS" is the hardest, most defensible
proxy for "proven medal contention" -- it counts only athletes who have
actually stood on a top-level podium.

Scopes:
  wtcs    "World Triathlon Championship Series {City}" rounds PLUS the season
          "World Triathlon Championship Finals {City}" (the world-title race).
  majors  WTCS (as above) PLUS individual Olympic medals (Elite Men/Women).
          Excludes Mixed Relay (a team medal), T100, and regional/distance champs.

Usage:
    python scripts/medal_contention.py                      # wtcs, rolling 4y
    python scripts/medal_contention.py --scope majors
    python scripts/medal_contention.py --years 4 --country "United States"
    python scripts/medal_contention.py --since 2022-01-01
"""
import argparse
import os
import sys
from datetime import date

import pandas as pd
from sqlalchemy import text

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tri_analysis.database import get_engine  # noqa: E402

# Event-name patterns per scope. WTCS = series rounds + season Championship Finals.
# "Olympic Games" matches the Paris 2024 individual races but not qualification events.
SCOPE_REGEX = {
    "wtcs": r"World Triathlon Championship (Series|Finals)",
    "majors": r"World Triathlon Championship (Series|Finals)|Olympic Games",
}
ELITE_REGEX = r"(Elite Men|Elite Women)"  # excludes Mixed Relay team medals
OUTPUT_DIR = os.path.join(_repo_root, "outputs")


def fetch_medal_data(engine, since: str, scope: str = "wtcs") -> pd.DataFrame:
    """One row per (athlete, gender, country) with medal / top-6 / top-10 / start counts."""
    event_regex = SCOPE_REGEX[scope]
    q = text(f"""
        SELECT
            r.athlete_id,
            r.athlete_full_name,
            COALESCE(a.gender, 'unknown') AS gender,
            COALESCE(a.country, 'Unknown') AS country,
            COUNT(*) FILTER (WHERE r.finish_position BETWEEN 1 AND 3)  AS medals,
            COUNT(*) FILTER (WHERE r.finish_position = 1)              AS golds,
            COUNT(*) FILTER (WHERE r.finish_position = 2)              AS silvers,
            COUNT(*) FILTER (WHERE r.finish_position = 3)              AS bronzes,
            COUNT(*) FILTER (WHERE r.finish_position BETWEEN 1 AND 6)  AS top6,
            COUNT(*) FILTER (WHERE r.finish_position BETWEEN 1 AND 10) AS top10,
            COUNT(*) FILTER (WHERE r.finish_status = 'FINISH')         AS starts,
            MAX(e.event_date) FILTER (WHERE r.finish_position BETWEEN 1 AND 3) AS last_medal_date
        FROM race_results r
        JOIN events e
              ON e.event_id = r.event_id AND e.prog_id = r.prog_id
        LEFT JOIN athlete a
              ON a.athlete_id = r.athlete_id
        WHERE e.event_date >= :since
          AND e.event_name ~* :evt
          AND e.prog_name  ~* :elite
        GROUP BY r.athlete_id, r.athlete_full_name, a.gender, a.country
    """)
    return pd.read_sql(q, engine, params={"since": since, "evt": event_regex, "elite": ELITE_REGEX})


def country_medalists(medalists: pd.DataFrame) -> pd.DataFrame:
    """Distinct medalists per country, split by gender, sorted by total then women."""
    pivot = (medalists.groupby(["country", "gender"])["athlete_id"].nunique()
             .unstack(fill_value=0)
             .rename(columns={"male": "men", "female": "women"}))
    for c in ("men", "women"):
        if c not in pivot.columns:
            pivot[c] = 0
    pivot["total"] = pivot[["men", "women"]].sum(axis=1)
    return pivot[["men", "women", "total"]].sort_values(["total", "women"], ascending=False)


def country_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Podium vs. top-6 vs. top-10 distinct athletes per country (the pipeline behind the podium)."""
    def _n(mask):
        return df[mask].groupby("country")["athlete_id"].nunique()
    depth = pd.DataFrame({
        "medalists": _n(df["medals"] > 0),
        "top6": _n(df["top6"] > 0),
        "top10": _n(df["top10"] > 0),
    }).fillna(0).astype(int)
    return depth.sort_values(["medalists", "top10"], ascending=False)


def _hdr(txt: str) -> None:
    print("\n" + "=" * 72 + f"\n{txt}\n" + "=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=SCOPE_REGEX.keys(), default="wtcs",
                    help="wtcs (default) or majors (adds individual Olympic medals).")
    ap.add_argument("--years", type=int, default=4, help="Rolling window length (default 4).")
    ap.add_argument("--since", type=str, default=None, help="Explicit start YYYY-MM-DD (overrides --years).")
    ap.add_argument("--country", type=str, default=None, help="Print the full roster for one country.")
    ap.add_argument("--no-save", action="store_true", help="Do not write CSV outputs.")
    args = ap.parse_args()

    today = date.today()
    since = args.since or f"{today.year - args.years}-{today.month:02d}-{today.day:02d}"

    engine = get_engine()
    df = fetch_medal_data(engine, since, args.scope)
    medalists = df[df["medals"] > 0].copy()

    _hdr(f"{args.scope.upper()} MEDAL CONTENTION  |  {since} -> {today}  ({args.years}-yr window)")
    label = ("WTCS = World Triathlon Championship Series/Finals"
             if args.scope == "wtcs"
             else "MAJORS = WTCS Series/Finals + individual Olympic medals")
    print(f"{label}.  Medal = podium (1st/2nd/3rd) in an Elite race.\n")
    for name, sub in [("ALL", medalists),
                      ("MEN", medalists[medalists.gender == "male"]),
                      ("WOMEN", medalists[medalists.gender == "female"])]:
        print(f"  {name:6s} distinct medalists: {sub['athlete_id'].nunique():3d}"
              f"   (across {int(sub['medals'].sum())} podium finishes)")

    _hdr("DISTINCT MEDALISTS BY COUNTRY x GENDER")
    cm = country_medalists(medalists)
    print(cm.to_string())
    print(f"\n  Nations with >=1 medalist: {cm.shape[0]}")

    _hdr("DEPTH: podium vs. top-6 vs. top-10 distinct athletes per country")
    print("  (top-6/top-10 include the medalists -- shows the pipeline behind the podium)")
    depth = country_depth(df)
    print(depth.to_string())
    print(f"\n  GLOBAL   medalists={int((df.medals>0).sum()):3d}   "
          f"top6={int((df.top6>0).sum()):3d}   top10={int((df.top10>0).sum()):3d}")

    if args.country:
        _hdr(f"ROSTER: {args.country}")
        roster = (medalists[medalists.country.str.lower() == args.country.lower()]
                  .sort_values(["gender", "medals"], ascending=[True, False]))
        cols = ["athlete_full_name", "gender", "medals", "golds", "silvers", "bronzes",
                "top6", "top10", "starts", "last_medal_date"]
        print("  (no medalists in window)" if roster.empty else roster[cols].to_string(index=False))

    if not args.no_save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tag = f"{args.scope}_{since}_to_{today}"
        acols = ["athlete_full_name", "country", "gender", "medals", "golds", "silvers",
                 "bronzes", "top6", "top10", "starts", "last_medal_date"]
        (medalists.sort_values(["medals", "golds"], ascending=False)[acols]
         .to_csv(os.path.join(OUTPUT_DIR, f"medalists_{tag}.csv"), index=False, encoding="utf-8"))
        cm.to_csv(os.path.join(OUTPUT_DIR, f"medalists_by_country_{tag}.csv"), encoding="utf-8")
        depth.to_csv(os.path.join(OUTPUT_DIR, f"medal_depth_by_country_{tag}.csv"), encoding="utf-8")
        print(f"\nSaved 3 CSVs to outputs/ (tag: {tag})")


if __name__ == "__main__":
    main()
