from __future__ import annotations

from typing import Optional


def time_to_seconds(timestr: object) -> Optional[int]:
    """Convert time strings to seconds.

    Accepts:
    - "HH:MM:SS" or "MM:SS" or "SS"
    - int/float seconds

    Returns None for blanks/invalid values.
    """
    if timestr is None:
        return None

    if isinstance(timestr, bool):
        return None

    if isinstance(timestr, (int,)):
        return int(timestr)

    if isinstance(timestr, float):
        if timestr != timestr:  # NaN
            return None
        return int(timestr)

    s = str(timestr).strip()
    if not s:
        return None

    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = [int(float(p)) for p in parts]
            return h * 3600 + m * 60 + sec
        if len(parts) == 2:
            m, sec = [int(float(p)) for p in parts]
            return m * 60 + sec
        if len(parts) == 1:
            return int(float(parts[0]))
    except Exception:
        return None
    return None


def seconds_to_hms(seconds: object) -> Optional[str]:
    """Convert seconds to H:MM:SS (or M:SS) string.

    Returns None for invalid inputs.
    """
    if seconds is None:
        return None

    if isinstance(seconds, bool):
        return None

    if isinstance(seconds, float) and seconds != seconds:  # NaN
        return None

    try:
        sec_int = int(round(float(seconds)))
    except Exception:
        return None

    sign = "-" if sec_int < 0 else ""
    sec_int = abs(sec_int)

    h = sec_int // 3600
    m = (sec_int % 3600) // 60
    s = sec_int % 60

    if h > 0:
        return f"{sign}{h}:{m:02}:{s:02}"
    return f"{sign}{m:02}:{s:02}"


def pace_sec_per_100m(seconds: Optional[int], distance_m: Optional[float]) -> Optional[float]:
    if seconds is None or distance_m is None:
        return None
    if distance_m <= 0:
        return None
    return seconds / (distance_m / 100.0)


def pace_sec_per_km(seconds: Optional[int], distance_m: Optional[float]) -> Optional[float]:
    if seconds is None or distance_m is None:
        return None
    if distance_m <= 0:
        return None
    return seconds / (distance_m / 1000.0)
