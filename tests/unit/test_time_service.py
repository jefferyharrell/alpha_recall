"""Tests for TimeService."""

import json
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from alpha_recall.services.time import TimeService, time_service


class TestTimeService:
    """Test TimeService functionality."""

    def test_utc_now(self):
        """Test utc_now returns Pendulum DateTime."""
        now = TimeService.utc_now()
        assert now.timezone.name == "UTC"
        assert hasattr(now, "isoformat")

    def test_utc_isoformat(self):
        """Test utc_isoformat returns proper ISO format."""
        iso_str = TimeService.utc_isoformat()
        assert iso_str.endswith("+00:00")
        assert "T" in iso_str

    def test_local_now(self):
        """Test local_now returns local time."""
        local = TimeService.local_now()
        assert hasattr(local, "timezone")
        assert hasattr(local, "isoformat")

    def test_now_comprehensive(self):
        """Test the comprehensive now() method."""
        now_data = TimeService.now()

        # Check required fields
        required_fields = [
            "iso_datetime",
            "utc",
            "local",
            "human_readable",
            "timezone",
            "unix_timestamp",
            "day_of_week",
        ]

        for field in required_fields:
            assert field in now_data, f"Missing field: {field}"

        # Check timezone structure
        assert "name" in now_data["timezone"]
        assert "offset" in now_data["timezone"]
        assert "display" in now_data["timezone"]

        # Check day_of_week structure
        assert "integer" in now_data["day_of_week"]
        assert "name" in now_data["day_of_week"]
        assert 1 <= now_data["day_of_week"]["integer"] <= 7

        # Check UTC and iso_datetime are the same
        assert now_data["utc"] == now_data["iso_datetime"]
        assert now_data["utc"].endswith("+00:00")

        # Check it's valid JSON
        json.dumps(now_data)  # Should not raise

    def test_parse_time_string(self):
        """Test parsing various time string formats."""
        # Test UTC string
        utc_str = "2025-07-09T15:30:00+00:00"
        parsed = TimeService.parse(utc_str)
        assert parsed.offset == 0  # UTC offset

        # Test local assumption
        naive_str = "2025-07-09T15:30:00"
        parsed_local = TimeService.parse(naive_str, assume_local=True)
        assert parsed_local.timezone is not None

        # Test UTC assumption
        parsed_utc = TimeService.parse(naive_str, assume_local=False)
        assert parsed_utc.timezone.name == "UTC"

    def test_parse_time_filter(self):
        """Test parsing time filter strings."""
        # Test hours
        cutoff = TimeService.parse_time_filter("2h")
        assert cutoff is not None
        now = TimeService.utc_now()
        assert cutoff < now

        # Test minutes
        cutoff = TimeService.parse_time_filter("30m")
        assert cutoff is not None

        # Test days
        cutoff = TimeService.parse_time_filter("1d")
        assert cutoff is not None

        # Test invalid
        cutoff = TimeService.parse_time_filter("invalid")
        assert cutoff is None

    def test_to_utc_conversion(self):
        """Test converting various inputs to UTC."""
        # Test string input
        utc_dt = TimeService.to_utc("2025-07-09T15:30:00+00:00")
        assert utc_dt.timezone.name == "UTC"

        # Test DateTime input
        local_dt = TimeService.local_now()
        utc_dt = TimeService.to_utc(local_dt)
        assert utc_dt.timezone.name == "UTC"

    def test_to_utc_isoformat(self):
        """Test converting to UTC ISO format."""
        # Test with None (current time)
        iso_str = TimeService.to_utc_isoformat()
        assert iso_str.endswith("+00:00")

        # Test with string
        iso_str = TimeService.to_utc_isoformat("2025-07-09T15:30:00+00:00")
        assert iso_str.endswith("+00:00")

    def test_format_duration(self):
        """Test duration formatting."""
        # Test seconds
        assert "seconds" in TimeService.format_duration(30)

        # Test minutes
        assert "minute" in TimeService.format_duration(90)

        # Test hours
        assert "hour" in TimeService.format_duration(3661)

    def test_age_string(self):
        """Test age string generation."""
        # Test recent time
        now = TimeService.utc_now()
        recent = now.subtract(minutes=5)
        age = TimeService.age_string(recent)
        assert "minutes ago" in age

        # Test older time
        older = now.subtract(days=2)
        age = TimeService.age_string(older)
        assert "days ago" in age

    def test_global_instance(self):
        """Test that global time_service instance works."""
        iso_str = time_service.utc_isoformat()
        assert iso_str.endswith("+00:00")

        now_data = time_service.now()
        assert "iso_datetime" in now_data
