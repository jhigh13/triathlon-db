"""
Unit tests for the ranking points scoring engine.
Validates formulas against known values from athlete_ranking_breakdown.
"""
import pytest
from tri_analysis.ranking_points import (
    classify_event,
    points_for_position,
    parse_time_seconds,
    is_sprint,
    country_to_continent,
    event_country_to_continent,
)


# ---------------------------------------------------------------------------
# classify_event
# ---------------------------------------------------------------------------
class TestClassifyEvent:
    def test_wtcs(self):
        result = classify_event("World Championship Series")
        assert result is not None
        base, uses_qof, etype = result
        assert base == 1000.0
        assert uses_qof is False
        assert etype == "wtcs"

    def test_champ_finals(self):
        base, uses_qof, etype = classify_event("World Championship Finals")
        assert base == 1250.0
        assert etype == "champ_finals"

    def test_world_cup(self):
        base, uses_qof, etype = classify_event("World Triathlon Cup")
        assert base == 500.0
        assert uses_qof is False

    def test_continental_cup(self):
        base, uses_qof, etype = classify_event("Continental Cup")
        assert base == 250.0
        assert uses_qof is True
        assert etype == "cont_cup"

    def test_continental_champs(self):
        base, uses_qof, etype = classify_event("Continental Championships")
        assert base == 400.0
        assert uses_qof is True
        assert etype == "cont_champs"

    def test_composite_cat_name(self):
        # "Continental Cup, Regional Championships" → Continental Cup wins (first match)
        result = classify_event("Continental Cup, Regional Championships")
        assert result is not None
        base, _, etype = result
        assert base == 250.0
        assert etype == "cont_cup"

    def test_champs_finals_composite(self):
        result = classify_event("World Championships, World Championship Finals")
        assert result is not None
        base, _, etype = result
        assert base == 1250.0
        assert etype == "champ_finals"

    def test_junior_skipped(self):
        assert classify_event("Continental Junior Cup") is None

    def test_para_skipped(self):
        assert classify_event("World Para Cup") is None

    def test_age_group_skipped(self):
        assert classify_event("Continental Championships, Age-Group Event") is None

    def test_empty(self):
        assert classify_event("") is None

    def test_t100(self):
        base, uses_qof, etype = classify_event("T100 Triathlon World Tour")
        assert base == 500.0
        assert uses_qof is False
        assert etype == "t100"


# ---------------------------------------------------------------------------
# points_for_position — validated against real athlete_ranking_breakdown data
# ---------------------------------------------------------------------------
class TestPointsForPosition:
    def test_wtcs_1st(self):
        # Real data: Hauser 2025 WTCS Yokohama 1st = 1000 pts
        pts = points_for_position(1000.0, 1)
        assert abs(pts - 1000.0) < 0.01

    def test_wtcs_2nd(self):
        # Real data: 2nd place WTCS = 925 pts
        pts = points_for_position(1000.0, 2)
        assert abs(pts - 925.0) < 0.01

    def test_wtcs_3rd(self):
        pts = points_for_position(1000.0, 3)
        assert abs(pts - 925.0 * 0.925) < 0.01  # ≈ 855.625

    def test_world_cup_1st(self):
        pts = points_for_position(500.0, 1)
        assert abs(pts - 500.0) < 0.01

    def test_world_cup_2nd(self):
        pts = points_for_position(500.0, 2)
        assert abs(pts - 462.5) < 0.01

    def test_champ_finals_1st(self):
        pts = points_for_position(1250.0, 1)
        assert abs(pts - 1250.0) < 0.01

    def test_sprint_reduction(self):
        # Sprint WTCS = 25% of 1000 = 250 (confirmed: Abu Dhabi)
        pts = points_for_position(1000.0, 1, sprint=True)
        assert abs(pts - 250.0) < 0.01

    def test_sprint_position_2(self):
        pts = points_for_position(1000.0, 2, sprint=True)
        assert abs(pts - 250.0 * 0.925) < 0.01

    def test_qof_europe_cup(self):
        # Europe Continental Cup 1st = 250 * 1.20 = 300 (confirmed from real data)
        pts = points_for_position(250.0, 1, qof_factor=1.20)
        assert abs(pts - 300.0) < 0.01

    def test_qof_europe_champs_1st_with_bonus(self):
        # Europe Continental Champs 1st:
        # base=400, top5_bonus=+25%, qof=+30%
        # 400 * 1.30 (qof) * 1.25 (bonus) = 650
        pts = points_for_position(400.0, 1, qof_factor=1.30, cont_champs_bonus=True)
        assert abs(pts - 650.0) < 0.5  # confirmed from real breakdown data

    def test_cont_champs_bonus_positions(self):
        # Only positions 1-5 get the bonus
        pts_1 = points_for_position(400.0, 1, cont_champs_bonus=True)
        pts_6 = points_for_position(400.0, 6, cont_champs_bonus=True)
        assert pts_1 > points_for_position(400.0, 1)  # bonus applied
        pts_6_no_bonus = points_for_position(400.0, 6)
        assert abs(pts_6 - pts_6_no_bonus) < 0.01  # no bonus for pos 6

    def test_decay_formula(self):
        # Verify the 0.925 decay at various positions
        for pos in range(1, 20):
            pts = points_for_position(1000.0, pos)
            expected = 1000.0 * (0.925 ** (pos - 1))
            assert abs(pts - expected) < 0.01


# ---------------------------------------------------------------------------
# parse_time_seconds
# ---------------------------------------------------------------------------
class TestParseTimeSeconds:
    def test_hms(self):
        assert parse_time_seconds("01:45:30") == 3600 + 45*60 + 30

    def test_sprint_time(self):
        assert parse_time_seconds("00:54:57") == 54*60 + 57

    def test_ms(self):
        assert parse_time_seconds("54:57") == 54*60 + 57

    def test_empty(self):
        assert parse_time_seconds("") is None

    def test_none(self):
        assert parse_time_seconds(None) is None

    def test_invalid(self):
        assert parse_time_seconds("DNF") is None

    def test_cutoff_108pct(self):
        winner = parse_time_seconds("01:45:00")
        cutoff = winner * 1.08
        # 105 min winner → cutoff ≈ 113.4 min
        assert abs(cutoff - (105 * 60 * 1.08)) < 0.01


# ---------------------------------------------------------------------------
# is_sprint
# ---------------------------------------------------------------------------
class TestIsSprint:
    def test_explicit_sprint(self):
        assert is_sprint("sprint", None) is True

    def test_explicit_standard(self):
        assert is_sprint("standard", None) is False

    def test_empty_short_swim(self):
        # No category, swim < 1000m → sprint
        assert is_sprint("", 750.0) is True

    def test_empty_long_swim(self):
        assert is_sprint("", 1500.0) is False

    def test_empty_no_swim_data(self):
        # Can't determine, default False
        assert is_sprint("", None) is False

    def test_super_sprint(self):
        # "super_sprint" contains "sprint" as substring but is a distinct format;
        # the engine only triggers on exact "sprint" category, so this returns False
        assert is_sprint("super_sprint", None) is False


# ---------------------------------------------------------------------------
# continent mapping
# ---------------------------------------------------------------------------
class TestContinentMapping:
    def test_us(self):
        assert country_to_continent("United States") == "Americas"

    def test_gb(self):
        assert country_to_continent("Great Britain") == "Europe"

    def test_australia(self):
        assert country_to_continent("Australia") == "Oceania"

    def test_south_africa(self):
        assert country_to_continent("South Africa") == "Africa"

    def test_japan(self):
        assert country_to_continent("Japan") == "Asia"

    def test_unknown(self):
        assert country_to_continent("Atlantis") == "Unknown"
