"""Time utilities for Alpha-Recall.

All timestamps in Alpha-Recall should be stored and represented in UTC
with explicit timezone offset (+00:00) for maximum clarity.
"""

from datetime import datetime

import pendulum


def utc_timestamp(dt: datetime | str | None = None) -> str:
    """Convert any datetime to our standard UTC+offset format.

    Args:
        dt: Datetime to convert. Can be:
            - datetime object (timezone-aware or naive)
            - ISO string in any timezone
            - None (defaults to current UTC time)

    Returns:
        ISO timestamp string in UTC with explicit +00:00 offset

    Examples:
        >>> utc_timestamp()
        '2025-06-30T22:30:00.123456+00:00'

        >>> utc_timestamp(datetime(2025, 6, 30, 15, 30))  # naive = assume UTC
        '2025-06-30T15:30:00+00:00'

        >>> utc_timestamp("2025-06-30 3:30 PM PST")
        '2025-06-30T23:30:00+00:00'
    """
    if dt is None:
        # Current UTC time
        return pendulum.now("UTC").isoformat()

    if isinstance(dt, str):
        # Parse string - try pendulum first, fall back to basic formats
        try:
            parsed = pendulum.parse(dt)
            return parsed.in_timezone("UTC").isoformat()
        except Exception:
            # Fall back to basic ISO parsing
            parsed = pendulum.from_format(dt, "YYYY-MM-DDTHH:mm:ss")
            return parsed.in_timezone("UTC").isoformat()

    if isinstance(dt, datetime):
        # Convert datetime object
        if dt.tzinfo is None:
            # Naive datetime - assume UTC
            parsed = pendulum.from_timestamp(dt.timestamp(), tz="UTC")
        else:
            # Timezone-aware datetime
            parsed = pendulum.instance(dt)
        return parsed.in_timezone("UTC").isoformat()

    raise ValueError(f"Unsupported datetime type: {type(dt)}")


def current_utc() -> str:
    """Get current UTC timestamp in our standard format.

    Returns:
        Current UTC timestamp with explicit +00:00 offset
    """
    return utc_timestamp()
