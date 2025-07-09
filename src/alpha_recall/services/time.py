"""TimeService for Alpha-Recall - centralized time handling with opinionated defaults.

This service provides a single source of truth for all time operations in Alpha-Recall.
All internal datetime representations should be Pendulum objects, and all storage
should be in UTC with explicit timezone offsets (+00:00).

Key principles:
- All stored times are UTC with explicit offset
- Naive datetimes are assumed to be local time (not UTC)
- Comprehensive "now" method for tools that need rich time context
- Consistent parsing with clear error handling
- Timezone detection via IP geolocation for portable timezone handling
"""

from typing import Any

import pendulum
from pendulum import DateTime

from .geolocation import geolocation_service


class TimeService:
    """Centralized time service for Alpha-Recall."""

    @staticmethod
    async def now_async() -> dict[str, Any]:
        """Get comprehensive current time information (async version).

        Returns a rich dictionary with multiple time representations for use
        in tools like gentle_refresh that need comprehensive time context.

        Returns:
            Dict containing:
            - iso_datetime: UTC time in ISO format with +00:00 offset
            - utc: UTC time in ISO format (same as iso_datetime)
            - local: Local time in ISO format with local offset
            - human_readable: Human-friendly local time string
            - timezone: Dict with timezone information
            - unix_timestamp: Unix timestamp (seconds since epoch)
            - day_of_week: Dict with day of week info
        """
        # Get current UTC time (container should be in UTC)
        utc_now = pendulum.now("UTC")

        # Get local timezone from geolocation
        local_timezone_str = await geolocation_service.get_timezone(timeout=2.0)

        # Convert UTC to local timezone
        local_now = utc_now.in_timezone(local_timezone_str)

        return {
            "iso_datetime": utc_now.isoformat(),
            "utc": utc_now.isoformat(),
            "local": local_now.isoformat(),
            "human_readable": local_now.format("dddd, MMMM DD, YYYY h:mm A"),
            "timezone": {
                "name": local_now.timezone.name,
                "offset": local_now.format("ZZ"),  # e.g., "-0700"
                "display": local_now.format("zz"),  # e.g., "PDT"
            },
            "unix_timestamp": int(utc_now.timestamp()),
            "day_of_week": {
                "integer": local_now.weekday(),  # 1=Monday, 7=Sunday
                "name": local_now.format("dddd"),  # e.g., "Wednesday"
            },
        }

    @staticmethod
    def now() -> dict[str, Any]:
        """Get comprehensive current time information (sync version).

        Returns a rich dictionary with multiple time representations for use
        in tools like gentle_refresh that need comprehensive time context.

        Returns:
            Dict containing:
            - iso_datetime: UTC time in ISO format with +00:00 offset
            - utc: UTC time in ISO format (same as iso_datetime)
            - local: Local time in ISO format with local offset
            - human_readable: Human-friendly local time string
            - timezone: Dict with timezone information
            - unix_timestamp: Unix timestamp (seconds since epoch)
            - day_of_week: Dict with day of week info
        """
        # Get current UTC time (container should be in UTC)
        utc_now = pendulum.now("UTC")

        # Get local timezone from geolocation
        local_timezone_str = geolocation_service.get_timezone_sync(timeout=2.0)

        # Convert UTC to local timezone
        local_now = utc_now.in_timezone(local_timezone_str)

        return {
            "iso_datetime": utc_now.isoformat(),
            "utc": utc_now.isoformat(),
            "local": local_now.isoformat(),
            "human_readable": local_now.format("dddd, MMMM DD, YYYY h:mm A"),
            "timezone": {
                "name": local_now.timezone.name,
                "offset": local_now.format("ZZ"),  # e.g., "-0700"
                "display": local_now.format("zz"),  # e.g., "PDT"
            },
            "unix_timestamp": int(utc_now.timestamp()),
            "day_of_week": {
                "integer": local_now.weekday(),  # 1=Monday, 7=Sunday
                "name": local_now.format("dddd"),  # e.g., "Wednesday"
            },
        }

    @staticmethod
    def utc_now() -> DateTime:
        """Get current UTC time as Pendulum DateTime object.

        Returns:
            Current UTC time as Pendulum DateTime
        """
        return pendulum.now("UTC")

    @staticmethod
    def utc_isoformat() -> str:
        """Get current UTC time as ISO string with explicit +00:00 offset.

        This is the standard format for all timestamps stored in Alpha-Recall.

        Returns:
            Current UTC time as ISO string with +00:00 offset
        """
        return pendulum.now("UTC").isoformat()

    @staticmethod
    def local_now() -> DateTime:
        """Get current local time as Pendulum DateTime object.

        Returns:
            Current local time as Pendulum DateTime
        """
        return pendulum.now()

    @staticmethod
    def parse(
        time_str: str, assume_local: bool = True, strict: bool = False
    ) -> DateTime:
        """Parse time string with opinionated rules.

        Args:
            time_str: Time string to parse
            assume_local: If True, naive datetime strings are assumed to be local time.
                         If False, naive datetime strings are assumed to be UTC.
            strict: If True, raise exception on parse failure. If False, return None.

        Returns:
            Parsed time as Pendulum DateTime object

        Raises:
            ValueError: If strict=True and parsing fails

        Examples:
            >>> TimeService.parse("2025-07-09T15:30:00+00:00")
            # Returns UTC time

            >>> TimeService.parse("2025-07-09T15:30:00", assume_local=True)
            # Returns local time converted to UTC

            >>> TimeService.parse("2025-07-09 3:30 PM")
            # Returns parsed local time
        """
        try:
            # Try to parse with pendulum - it handles most formats
            parsed = pendulum.parse(time_str)

            # If the parsed time is naive (no timezone info)
            if parsed.timezone is None:
                if assume_local:
                    # Assume it's local time and convert to UTC
                    local_tz = pendulum.now().timezone
                    parsed = parsed.replace(tzinfo=local_tz)
                else:
                    # Assume it's UTC
                    parsed = parsed.replace(tzinfo=pendulum.timezone("UTC"))

            return parsed

        except Exception as e:
            if strict:
                raise ValueError(
                    f"Failed to parse time string '{time_str}': {e}"
                ) from e
            return None

    @staticmethod
    def to_utc(dt: DateTime | str) -> DateTime:
        """Convert any datetime to UTC.

        Args:
            dt: DateTime object or string to convert

        Returns:
            UTC DateTime object
        """
        if isinstance(dt, str):
            dt = TimeService.parse(dt)

        if dt is None:
            raise ValueError("Cannot convert None to UTC")

        return dt.in_timezone("UTC")

    @staticmethod
    def to_utc_isoformat(dt: DateTime | str | None = None) -> str:
        """Convert any datetime to our standard UTC ISO format.

        Args:
            dt: DateTime object, string, or None (defaults to current time)

        Returns:
            UTC ISO timestamp string with explicit +00:00 offset
        """
        if dt is None:
            return TimeService.utc_isoformat()

        if isinstance(dt, str):
            dt = TimeService.parse(dt)

        if dt is None:
            raise ValueError("Cannot convert None to UTC ISO format")

        return dt.in_timezone("UTC").isoformat()

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Human-readable duration string

        Examples:
            >>> TimeService.format_duration(3661)
            "1 hour, 1 minute, 1 second"

            >>> TimeService.format_duration(90)
            "1 minute, 30 seconds"
        """
        if seconds < 60:
            return f"{seconds:.1f} seconds"

        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)

        if minutes < 60:
            if remaining_seconds == 0:
                return f"{minutes} minute{'s' if minutes != 1 else ''}"
            return f"{minutes} minute{'s' if minutes != 1 else ''}, {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"

        hours = int(minutes // 60)
        remaining_minutes = int(minutes % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if remaining_minutes > 0:
            parts.append(
                f"{remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"
            )
        if remaining_seconds > 0:
            parts.append(
                f"{remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"
            )

        return ", ".join(parts)

    @staticmethod
    def age_string(created_at: DateTime | str) -> str:
        """Get human-readable age string from creation time.

        Args:
            created_at: Creation timestamp

        Returns:
            Human-readable age string like "2 hours ago", "3 days ago"
        """
        if isinstance(created_at, str):
            created_at = TimeService.parse(created_at)

        if created_at is None:
            return "unknown age"

        now = TimeService.utc_now()
        diff = now - created_at

        total_seconds = diff.total_seconds()

        if total_seconds < 60:
            return "just now"
        elif total_seconds < 3600:
            minutes = int(total_seconds // 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif total_seconds < 86400:
            hours = int(total_seconds // 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(total_seconds // 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"

    @staticmethod
    def parse_time_filter(time_filter: str) -> DateTime | None:
        """Parse time filter strings like '2h', '30m', '1d' into cutoff timestamps.

        Args:
            time_filter: Time filter string (e.g., '2h', '30m', '1d')

        Returns:
            Cutoff timestamp (times after this should be included)

        Examples:
            >>> TimeService.parse_time_filter('2h')
            # Returns timestamp 2 hours ago

            >>> TimeService.parse_time_filter('30m')
            # Returns timestamp 30 minutes ago
        """
        if not time_filter:
            return None

        time_filter = time_filter.strip().lower()

        # Parse number and unit
        import re

        match = re.match(r"^(\d+)([smhd])$", time_filter)
        if not match:
            return None

        amount = int(match.group(1))
        unit = match.group(2)

        now = TimeService.utc_now()

        if unit == "s":
            return now.subtract(seconds=amount)
        elif unit == "m":
            return now.subtract(minutes=amount)
        elif unit == "h":
            return now.subtract(hours=amount)
        elif unit == "d":
            return now.subtract(days=amount)
        else:
            return None


# Global instance for easy access
time_service = TimeService()
