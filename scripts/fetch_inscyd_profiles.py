"""
Fetch INSCYD athlete metabolic profiles (one-time data pull).

Workflow (matches the "confirm access, then fetch once and store" plan):

    # 1) Light auth/connectivity check — no bulk download:
    python scripts/fetch_inscyd_profiles.py verify --user-email you@org.com

    # 2) Once verified, pull and cache to triathlon-db/data/inscyd/:
    python scripts/fetch_inscyd_profiles.py fetch \
        --user-email you@org.com --sport-id 7 --athlete-display-id 154662600022

Credentials come from triathlon-db/.env:
    INSCYD_API_KEY=...
    INSCYD_HOST=app.inscyd.com        # host only, no scheme
    # optional: INSCYD_BIKE_SPORT_ID=...

The fetch endpoints return heavy polynomial payloads, so prefer `verify` first
and run `fetch` only once per athlete; the engine reads the cached JSON after.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tri_analysis.inscyd import InscydAPIError, InscydClient, MetabolicTest, save_raw_tests
from tri_analysis.inscyd.polynomial import METRIC_KEYS

logger = logging.getLogger("fetch_inscyd")

# Powers (W) used for a quick post-fetch sanity print of the curves.
_SANITY_POWERS = (150.0, 250.0, 350.0)


def _add_common_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--user-id", type=int, default=None)
    p.add_argument("--user-email", type=str, default=None)
    p.add_argument("--sport-id", type=int, default=None, help="INSCYD sport id (use the cycling id)")
    p.add_argument("--athlete-display-id", type=int, default=None)


def _summarize_test(test: MetabolicTest) -> None:
    print(f"  test {test.test_id} | {test.athlete_name or '?'} "
          f"| display_id={test.athlete_display_id} | sport={test.sport_id} | {test.created_at}")
    scal = test.scalars
    print(f"    vo2max={scal.get('vo2max')}  vlamax={scal.get('vlamax')}  "
          f"AT={scal.get('anaerobic_threshold_absolute')}W  "
          f"fatmax_power={scal.get('fatmax_power')}W  carb_max={scal.get('carb_max')}")
    for key in METRIC_KEYS:
        poly = test.curves.get(key)
        if poly is None:
            continue
        vals = ", ".join(f"{p:.0f}W={poly.evaluate(p):.2f}" for p in _SANITY_POWERS)
        print(f"    {key:<22} deg={poly.degree} sse={poly.sse:.4g}  [{vals}]")


def cmd_verify(args: argparse.Namespace) -> int:
    client = InscydClient()
    status = client.verify_access(
        user_id=args.user_id,
        user_email=args.user_email,
        sport_id=args.sport_id,
        athlete_display_id=args.athlete_display_id,
    )
    print("INSCYD access OK")
    for k, v in status.items():
        print(f"  {k}: {v}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    client = InscydClient()
    if args.last:
        results = client.get_last_data(
            user_id=args.user_id,
            user_email=args.user_email,
            sport_id=args.sport_id,
            athlete_display_id=args.athlete_display_id,
        )
        source = "last_data"
    else:
        results = client.get_all_data(
            user_id=args.user_id,
            user_email=args.user_email,
            sport_id=args.sport_id,
            athlete_display_id=args.athlete_display_id,
            start_date=args.start_date,
            end_date=args.end_date,
            max_pages=args.max_pages,
        )
        source = "all_data"

    if not results:
        print("No results returned for those filters. Nothing stored.")
        return 1

    print(f"Fetched {len(results)} test(s) via {source}:")
    for record in results:
        _summarize_test(MetabolicTest.from_api(record))

    if args.no_store:
        print("\n--no-store set; skipping write.")
        return 0

    path = save_raw_tests(
        results,
        athlete_display_id=args.athlete_display_id,
        sport_id=args.sport_id,
        source=source,
        filename=args.filename,
        out_dir=args.out_dir,
    )
    print(f"\nStored payload -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch INSCYD athlete metabolic profiles.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="Light auth/connectivity check (no bulk download).")
    _add_common_filters(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    p_fetch = sub.add_parser("fetch", help="Pull tests and cache JSON to disk.")
    _add_common_filters(p_fetch)
    p_fetch.add_argument("--last", action="store_true", help="Use last_data (latest test only).")
    p_fetch.add_argument("--start-date", type=str, default=None, help="ISO date, e.g. 2025-01-01T00:00:00")
    p_fetch.add_argument("--end-date", type=str, default=None)
    p_fetch.add_argument("--max-pages", type=int, default=None)
    p_fetch.add_argument("--out-dir", type=str, default=None)
    p_fetch.add_argument("--filename", type=str, default=None)
    p_fetch.add_argument("--no-store", action="store_true", help="Print only; do not write JSON.")
    p_fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args(argv)
    # out_dir default handled here so storage's DEFAULT_DIR stays the single source.
    if getattr(args, "out_dir", None) is None and args.command == "fetch":
        from tri_analysis.inscyd.storage import DEFAULT_DIR
        args.out_dir = DEFAULT_DIR

    try:
        return args.func(args)
    except InscydAPIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
