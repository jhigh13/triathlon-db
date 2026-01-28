"""
Time parsing and formatting utilities for the prediction pipeline.

Handles time strings in mm:ss or hh:mm:ss format, converting to/from integer seconds.
Gracefully handles None, empty strings, and invalid formats (e.g., DNS, DNF).
"""

from __future__ import annotations
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def parse_time_to_seconds(s: str | None) -> int | None:
    """
    Parse a time string to integer seconds.

    Accepts:
    - "HH:MM:SS" -> hours*3600 + minutes*60 + seconds
    - "MM:SS" -> minutes*60 + seconds
    - "SS" -> seconds (numeric string)
    - int/float -> cast to int
    - None, empty, or invalid strings (e.g., "DNF", "DNS") -> None

    Examples:
        >>> parse_time_to_seconds("1:02:03")
        3723
        >>> parse_time_to_seconds("59:30")
        3570
        >>> parse_time_to_seconds("45")
        45
        >>> parse_time_to_seconds(None) is None
        True
        >>> parse_time_to_seconds("DNF") is None
        True
        >>> parse_time_to_seconds("") is None
        True
        >>> parse_time_to_seconds(3600)
        3600
        >>> parse_time_to_seconds(3600.5)
        3600
    """
    if s is None:
        return None

    # Handle bool before numeric check (bool is subclass of int in Python)
    if isinstance(s, bool):
        return None

    # Handle numeric types directly
    if isinstance(s, int):
        return s
    if isinstance(s, float):
        if s != s:  # NaN check
            return None
        return int(s)

    # String processing
    text = str(s).strip()
    if not text:
        return None

    parts = text.split(":")
    try:
        if len(parts) == 3:
            # HH:MM:SS
            h, m, sec = [int(float(p)) for p in parts]
            return h * 3600 + m * 60 + sec
        elif len(parts) == 2:
            # MM:SS
            m, sec = [int(float(p)) for p in parts]
            return m * 60 + sec
        elif len(parts) == 1:
            # SS (single number)
            return int(float(parts[0]))
    except (ValueError, TypeError):
        # Non-numeric strings like "DNF", "DNS", etc.
        return None

    return None


def seconds_to_hms(sec: int | None) -> str | None:
    """
    Convert integer seconds to a human-readable time string.

    Returns:
    - "H:MM:SS" if hours > 0
    - "MM:SS" if hours == 0
    - None if input is None or invalid

    Examples:
        >>> seconds_to_hms(3723)
        '1:02:03'
        >>> seconds_to_hms(3570)
        '59:30'
        >>> seconds_to_hms(45)
        '00:45'
        >>> seconds_to_hms(0)
        '00:00'
        >>> seconds_to_hms(None) is None
        True
        >>> seconds_to_hms(-60)
        '-01:00'
    """
    if sec is None:
        return None

    if isinstance(sec, bool):
        return None

    # Handle float (e.g., from numpy)
    if isinstance(sec, float):
        if sec != sec:  # NaN
            return None
        sec = int(round(sec))

    try:
        sec_int = int(sec)
    except (ValueError, TypeError):
        return None

    sign = "-" if sec_int < 0 else ""
    sec_int = abs(sec_int)

    hours = sec_int // 3600
    minutes = (sec_int % 3600) // 60
    secs = sec_int % 60

    if hours > 0:
        return f"{sign}{hours}:{minutes:02d}:{secs:02d}"
    return f"{sign}{minutes:02d}:{secs:02d}"


def parse_time_columns(df, columns: list[str], suffix: str = "_sec") -> None:
    """
    Parse multiple time columns in a DataFrame in-place, adding new columns with suffix.

    Args:
        df: pandas DataFrame
        columns: list of column names containing time strings
        suffix: suffix for new integer-seconds columns (default "_sec")

    Example:
        parse_time_columns(df, ["swimtime", "biketime", "runtime"])
        # Creates df["swimtime_sec"], df["biketime_sec"], df["runtime_sec"]
    """
    for col in columns:
        if col in df.columns:
            new_col = col + suffix
            df[new_col] = df[col].apply(parse_time_to_seconds)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
