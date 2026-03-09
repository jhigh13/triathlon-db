"""
Weather data integration via Open-Meteo API.

Provides historical weather backfill and forecast fetching for triathlon events.
Uses the free Open-Meteo API (no key required for non-commercial use).

Historical: https://archive-api.open-meteo.com/v1/archive
Forecast:   https://api.open-meteo.com/v1/forecast
"""

from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Open-Meteo endpoints
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Hourly variables to request
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "wet_bulb_temperature_2m",
]

# Rate limiting
_REQUEST_DELAY_SEC = 0.5


def _estimate_globe_temperature(
    temp_c: float, solar_wm2: float, wind_ms: float
) -> float:
    """
    Estimate black globe temperature from air temp, solar radiation, and wind.

    Uses the simplified Liljegren approach:
        Tg ≈ 1.1 * Ta + 2.2 * (solar / 1000) - 0.3 * sqrt(wind) - 0.4
    where Ta is air temp (°C), solar is W/m², wind is m/s.

    This is an approximation — the full Liljegren model requires iterative
    solving, but this gives reasonable WBGT estimates for race conditions.
    """
    wind_term = math.sqrt(max(wind_ms, 0.1))
    return 1.1 * temp_c + 2.2 * (solar_wm2 / 1000.0) - 0.3 * wind_term - 0.4


def estimate_wbgt(
    temp_c: float,
    wet_bulb_c: float,
    wind_kmh: float,
    solar_wm2: float,
) -> float:
    """
    Estimate Wet Bulb Globe Temperature (WBGT) from available variables.

    WBGT = 0.7 * Twb + 0.2 * Tg + 0.1 * Tdb

    Args:
        temp_c: Dry bulb (air) temperature in °C
        wet_bulb_c: Wet bulb temperature in °C
        wind_kmh: Wind speed in km/h
        solar_wm2: Shortwave solar radiation in W/m²

    Returns:
        Estimated WBGT in °C
    """
    wind_ms = wind_kmh / 3.6
    tg = _estimate_globe_temperature(temp_c, solar_wm2, wind_ms)
    return 0.7 * wet_bulb_c + 0.2 * tg + 0.1 * temp_c


def _aggregate_race_window(
    hourly_data: dict,
    start_hour: int = 7,
    end_hour: int = 15,
) -> dict:
    """
    Aggregate hourly weather data over a typical race window.

    Args:
        hourly_data: Open-Meteo hourly response dict with 'time' and variable arrays
        start_hour: Race window start (inclusive, default 7am)
        end_hour: Race window end (exclusive, default 3pm)

    Returns:
        Dict of aggregated weather values for the race window
    """
    times = hourly_data.get("time", [])
    if not times:
        return {}

    # Find indices within race window
    indices = []
    for i, t in enumerate(times):
        # times are ISO format like "2024-01-15T08:00"
        hour = int(t.split("T")[1].split(":")[0])
        if start_hour <= hour < end_hour:
            indices.append(i)

    if not indices:
        # Fall back to all available hours
        indices = list(range(len(times)))

    def _safe_mean(key):
        vals = [hourly_data[key][i] for i in indices if hourly_data.get(key) and i < len(hourly_data[key]) and hourly_data[key][i] is not None]
        return sum(vals) / len(vals) if vals else None

    def _safe_max(key):
        vals = [hourly_data[key][i] for i in indices if hourly_data.get(key) and i < len(hourly_data[key]) and hourly_data[key][i] is not None]
        return max(vals) if vals else None

    def _safe_sum(key):
        vals = [hourly_data[key][i] for i in indices if hourly_data.get(key) and i < len(hourly_data[key]) and hourly_data[key][i] is not None]
        return sum(vals) if vals else None

    temp_c = _safe_mean("temperature_2m")
    humidity = _safe_mean("relative_humidity_2m")
    apparent_temp = _safe_mean("apparent_temperature")
    wind_speed = _safe_mean("wind_speed_10m")
    wind_gust = _safe_max("wind_gusts_10m")
    precipitation = _safe_sum("precipitation")
    cloud_cover = _safe_mean("cloud_cover")
    solar_radiation = _safe_mean("shortwave_radiation")
    wet_bulb = _safe_mean("wet_bulb_temperature_2m")

    # Estimate WBGT if we have the necessary variables
    wbgt = None
    if temp_c is not None and wet_bulb is not None and wind_speed is not None and solar_radiation is not None:
        wbgt = round(estimate_wbgt(temp_c, wet_bulb, wind_speed, solar_radiation), 1)

    result = {
        "temperature_air": round(temp_c, 1) if temp_c is not None else None,
        "humidity": round(humidity, 0) if humidity is not None else None,
        "apparent_temp": round(apparent_temp, 1) if apparent_temp is not None else None,
        "wind_speed_kmh": round(wind_speed, 1) if wind_speed is not None else None,
        "wind_gust_kmh": round(wind_gust, 1) if wind_gust is not None else None,
        "precipitation_mm": round(precipitation, 1) if precipitation is not None else None,
        "cloud_cover_pct": round(cloud_cover, 0) if cloud_cover is not None else None,
        "wet_bulb_temp": round(wet_bulb, 1) if wet_bulb is not None else None,
        "wbgt_estimate": wbgt,
    }
    return result


def fetch_historical_weather(
    lat: float,
    lon: float,
    event_date: date,
    start_hour: int = 7,
    end_hour: int = 15,
) -> Optional[dict]:
    """
    Fetch historical weather for a race day from Open-Meteo Archive API.

    IMPORTANT: Only works for past dates. Returns None for future dates.
    Use fetch_weather_smart() for automatic routing.

    Args:
        lat: Event latitude
        lon: Event longitude
        event_date: Race date (must be in the past)
        start_hour: Race window start hour (default 7am)
        end_hour: Race window end hour (default 3pm)

    Returns:
        Dict of aggregated weather values, or None on failure
    """
    # Parse event_date to a date object if it's a string
    if isinstance(event_date, str):
        event_date = date.fromisoformat(event_date)
    elif isinstance(event_date, datetime):
        event_date = event_date.date()

    # Guard: Archive API only works for past dates
    today = date.today()
    if event_date >= today:
        logger.debug(f"Skipping archive fetch for future date {event_date} (today={today})")
        return None

    date_str = event_date.isoformat()

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }

    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        if not hourly:
            logger.warning(f"No hourly data returned for {lat},{lon} on {date_str}")
            return None

        result = _aggregate_race_window(hourly, start_hour, end_hour)
        result["weather_source"] = "open_meteo"
        result["weather_fetched_at"] = datetime.now(timezone.utc).isoformat()
        return result

    except requests.RequestException as e:
        logger.error(f"Failed to fetch weather for {lat},{lon} on {date_str}: {e}")
        return None


def fetch_last_year_weather(
    lat: float,
    lon: float,
    event_date: date,
    start_hour: int = 7,
    end_hour: int = 15,
) -> Optional[dict]:
    """
    Fetch weather from the same date one year ago as a climate proxy.

    Used for future events where neither archive nor forecast data is available.
    Provides a reasonable baseline based on typical conditions for that location
    and time of year.

    Args:
        lat: Event latitude
        lon: Event longitude
        event_date: The future event date (will look up same date last year)
        start_hour: Race window start hour (default 7am)
        end_hour: Race window end hour (default 3pm)

    Returns:
        Dict of aggregated weather values with source='open_meteo_lastyear', or None
    """
    if isinstance(event_date, str):
        event_date = date.fromisoformat(event_date)
    elif isinstance(event_date, datetime):
        event_date = event_date.date()

    # Same date, one year prior
    try:
        last_year_date = event_date.replace(year=event_date.year - 1)
    except ValueError:
        # Feb 29 in a non-leap year -> use Feb 28
        last_year_date = event_date.replace(year=event_date.year - 1, day=28)

    logger.info(f"Fetching last-year weather proxy: {last_year_date} for future event {event_date}")

    result = fetch_historical_weather(lat, lon, last_year_date, start_hour, end_hour)
    if result is not None:
        result["weather_source"] = "open_meteo_lastyear"
    return result


def fetch_weather_smart(
    lat: float,
    lon: float,
    event_date: date,
    start_hour: int = 7,
    end_hour: int = 15,
) -> Optional[dict]:
    """
    Smart weather fetch that picks the best strategy based on event date.

    Routing logic:
    - Past dates: Use archive API (historical data)
    - Within 16 days: Use forecast API
    - Further future: Use same-date-last-year as climate proxy

    Args:
        lat: Event latitude
        lon: Event longitude
        event_date: Race date
        start_hour: Race window start hour (default 7am)
        end_hour: Race window end hour (default 3pm)

    Returns:
        Dict of aggregated weather values, or None on failure
    """
    if isinstance(event_date, str):
        event_date = date.fromisoformat(event_date)
    elif isinstance(event_date, datetime):
        event_date = event_date.date()

    today = date.today()
    days_ahead = (event_date - today).days

    if days_ahead < 0:
        # Past date -> archive API
        result = fetch_historical_weather(lat, lon, event_date, start_hour, end_hour)
    elif days_ahead <= 16:
        # Within forecast range -> try forecast first, fall back to last year
        result = fetch_forecast_weather(lat, lon, event_date, start_hour, end_hour)
        if result is None:
            logger.info(f"Forecast unavailable for {event_date}, falling back to last-year proxy")
            result = fetch_last_year_weather(lat, lon, event_date, start_hour, end_hour)
    else:
        # Far future -> last year's weather as proxy
        result = fetch_last_year_weather(lat, lon, event_date, start_hour, end_hour)

    # Normalize keys so callers get DB-compatible names
    # (e.g. wbgt_estimate -> wbgt for feature builder compatibility)
    if result is not None and "wbgt_estimate" in result:
        result["wbgt"] = result.pop("wbgt_estimate")

    return result


def fetch_forecast_weather(
    lat: float,
    lon: float,
    event_date: date,
    start_hour: int = 7,
    end_hour: int = 15,
) -> Optional[dict]:
    """
    Fetch forecast weather for an upcoming race from Open-Meteo Forecast API.

    Works for events up to 16 days in the future.

    Args:
        lat: Event latitude
        lon: Event longitude
        event_date: Race date (must be within 16 days of today)
        start_hour: Race window start hour (default 7am)
        end_hour: Race window end hour (default 3pm)

    Returns:
        Dict of aggregated weather values, or None on failure
    """
    date_str = event_date.isoformat() if isinstance(event_date, date) else str(event_date)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }

    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        if not hourly:
            logger.warning(f"No forecast data for {lat},{lon} on {date_str}")
            return None

        result = _aggregate_race_window(hourly, start_hour, end_hour)
        result["weather_source"] = "open_meteo_forecast"
        result["weather_fetched_at"] = datetime.now(timezone.utc).isoformat()
        return result

    except requests.RequestException as e:
        logger.error(f"Failed to fetch forecast for {lat},{lon} on {date_str}: {e}")
        return None


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode_venue(venue: str, country: Optional[str] = None) -> Optional[tuple[float, float]]:
    """
    Geocode an event venue name to (latitude, longitude) using Open-Meteo.

    Tries the venue name first, then falls back to simplified names
    (first word, last word) since venues often include park/complex names
    that confuse the geocoder.

    Args:
        venue: Venue/city name (e.g. "Hamburg", "Kelowna City Park")
        country: Optional country name for fallback attempts

    Returns:
        (lat, lon) tuple, or None if geocoding fails
    """
    # Build candidate queries: try venue as-is, then simplified forms
    candidates = [venue]
    words = venue.split()
    if len(words) > 1:
        # Try first word (often the city name)
        candidates.append(words[0])
        # Try last word
        candidates.append(words[-1])
    # Try country as fallback for very specific venue names
    if country:
        candidates.append(country)

    for query in candidates:
        params = {"name": query, "count": 3, "language": "en"}
        try:
            resp = requests.get(GEOCODING_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                continue

            # If we have a country, prefer results matching that country
            if country:
                for r in results:
                    if r.get("country", "").lower() == country.lower():
                        logger.debug(f"Geocoded '{venue}' via query '{query}' -> {r['name']}, {r['country']}")
                        return (r["latitude"], r["longitude"])

            # Otherwise take the top result
            r = results[0]
            logger.debug(f"Geocoded '{venue}' via query '{query}' -> {r['name']}, {r.get('country', '?')}")
            return (r["latitude"], r["longitude"])

        except requests.RequestException as e:
            logger.error(f"Geocoding failed for '{query}': {e}")
            continue

    logger.warning(f"No geocoding result for venue='{venue}', country='{country}'")
    return None


def backfill_missing_coordinates(engine, dry_run: bool = False) -> dict:
    """
    Geocode events that have a venue name but no coordinates.

    Returns:
        Dict with counts: {total, geocoded, failed}
    """
    from sqlalchemy import text

    query = text("""
        SELECT DISTINCT event_id, event_venue, event_country
        FROM events
        WHERE (event_latitude IS NULL OR event_longitude IS NULL)
          AND event_venue IS NOT NULL AND event_venue != ''
    """)

    with engine.connect() as conn:
        events = conn.execute(query).fetchall()

    stats = {"total": len(events), "geocoded": 0, "failed": 0}
    logger.info(f"Geocoding {len(events)} events with venue but no coordinates...")

    if dry_run:
        return stats

    for i, row in enumerate(events):
        event_id, venue, country = row[0], row[1], row[2]
        logger.info(f"[{i+1}/{len(events)}] Geocoding '{venue}' ({country})")

        coords = geocode_venue(venue, country)
        if coords is None:
            stats["failed"] += 1
            time.sleep(_REQUEST_DELAY_SEC)
            continue

        lat, lon = coords
        update_sql = text("""
            UPDATE events
            SET event_latitude = :lat, event_longitude = :lon
            WHERE event_id = :event_id
              AND (event_latitude IS NULL OR event_longitude IS NULL)
        """)
        with engine.begin() as conn:
            conn.execute(update_sql, {"lat": lat, "lon": lon, "event_id": event_id})

        stats["geocoded"] += 1
        logger.info(f"  -> {lat}, {lon}")
        time.sleep(_REQUEST_DELAY_SEC)

    logger.info(f"Geocoding complete: {stats['geocoded']} geocoded, {stats['failed']} failed")
    return stats


def backfill_event_weather(
    engine,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Backfill missing weather data for events using Open-Meteo.

    Args:
        engine: SQLAlchemy Engine
        start_date: Only process events on or after this date (YYYY-MM-DD)
        end_date: Only process events on or before this date (YYYY-MM-DD)
        overwrite: If True, re-fetch even if weather_source is already set
        dry_run: If True, just count events without fetching

    Returns:
        Dict with counts: {total, fetched, skipped, failed, no_coords}
    """
    from sqlalchemy import text

    # Build query for events needing weather
    where_parts = ["e.event_latitude IS NOT NULL", "e.event_longitude IS NOT NULL"]

    if not overwrite:
        where_parts.append("(e.weather_source IS NULL OR e.weather_source = '')")

    params = {}
    if start_date:
        where_parts.append("e.event_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_parts.append("e.event_date <= :end_date")
        params["end_date"] = end_date

    where_sql = " AND ".join(where_parts)

    # Count total events and events with coordinates
    count_query = text(f"""
        SELECT COUNT(*) FROM events e WHERE {where_sql}
    """)
    no_coords_query = text("""
        SELECT COUNT(*) FROM events
        WHERE event_latitude IS NULL OR event_longitude IS NULL
    """)

    with engine.connect() as conn:
        total = conn.execute(count_query, params).scalar()
        no_coords = conn.execute(no_coords_query).scalar()

    stats = {"total": total, "fetched": 0, "skipped": 0, "failed": 0, "no_coords": no_coords}

    if dry_run:
        logger.info(f"Dry run: {total} events to backfill, {no_coords} events missing coordinates")
        return stats

    # Fetch events to process
    fetch_query = text(f"""
        SELECT DISTINCT e.event_id, e.event_date, e.event_name,
               e.event_latitude, e.event_longitude, e.weather_source
        FROM events e
        WHERE {where_sql}
        ORDER BY e.event_date DESC
    """)

    with engine.connect() as conn:
        events = conn.execute(fetch_query, params).fetchall()

    logger.info(f"Backfilling weather for {len(events)} events...")

    for i, row in enumerate(events):
        event_id = row[0]
        event_date_val = row[1]
        event_name = row[2]
        lat = row[3]
        lon = row[4]
        existing_source = row[5]

        if existing_source and existing_source == "api" and not overwrite:
            stats["skipped"] += 1
            continue

        logger.info(f"[{i+1}/{len(events)}] Fetching weather for {event_name} ({event_date_val}) at {lat},{lon}")

        weather = fetch_weather_smart(lat, lon, event_date_val)

        if weather is None:
            stats["failed"] += 1
            continue

        # Update the events table
        update_parts = []
        update_params = {"event_id": event_id}

        # Only update fields that are non-null from API
        field_map = {
            "temperature_air": "temperature_air",
            "humidity": "humidity",
            "wbgt_estimate": "wbgt",
            "apparent_temp": "apparent_temp",
            "wind_speed_kmh": "wind_speed_kmh",
            "wind_gust_kmh": "wind_gust_kmh",
            "precipitation_mm": "precipitation_mm",
            "cloud_cover_pct": "cloud_cover_pct",
            "wet_bulb_temp": "wet_bulb_temp",
            "weather_source": "weather_source",
            "weather_fetched_at": "weather_fetched_at",
        }

        for weather_key, db_col in field_map.items():
            val = weather.get(weather_key)
            if val is not None:
                update_parts.append(f"{db_col} = :{db_col}_val")
                update_params[f"{db_col}_val"] = str(val) if db_col in ("temperature_air", "humidity", "wbgt") else val

        if update_parts:
            update_sql = text(f"""
                UPDATE events
                SET {', '.join(update_parts)}
                WHERE event_id = :event_id
            """)
            with engine.begin() as conn:
                conn.execute(update_sql, update_params)

        stats["fetched"] += 1

        # Rate limiting
        time.sleep(_REQUEST_DELAY_SEC)

        # Progress log every 50 events
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(events)} ({stats['fetched']} fetched, {stats['failed']} failed)")

    logger.info(
        f"Backfill complete: {stats['fetched']} fetched, "
        f"{stats['skipped']} skipped, {stats['failed']} failed, "
        f"{stats['no_coords']} events missing coordinates"
    )
    return stats
