"""Geolocation-based timezone detection service.

This service detects the user's timezone based on IP geolocation,
providing a clean alternative to mounting host timezone files.
"""

import asyncio

import httpx

from ..logging import get_logger

logger = get_logger("geolocation")


class GeolocationService:
    """Service for detecting timezone based on IP geolocation."""

    # Fallback timezone if all services fail
    DEFAULT_TIMEZONE = "UTC"

    # Cache timezone for session to avoid repeated API calls
    _cached_timezone: str | None = None

    @classmethod
    async def get_timezone(cls, timeout: float = 3.0) -> str:
        """Get timezone string based on IP geolocation.

        Args:
            timeout: HTTP request timeout in seconds

        Returns:
            Timezone string (e.g., 'America/Los_Angeles') or 'UTC' if detection fails
        """
        # Return cached timezone if available
        if cls._cached_timezone:
            logger.debug("Using cached timezone", timezone=cls._cached_timezone)
            return cls._cached_timezone

        # Try multiple services for reliability
        services = [
            cls._try_worldtimeapi,
            cls._try_ipinfo,
            cls._try_ipapi,
        ]

        for service_func in services:
            try:
                timezone = await service_func(timeout)
                if timezone:
                    cls._cached_timezone = timezone
                    logger.info(
                        "Timezone detected via geolocation",
                        timezone=timezone,
                        service=service_func.__name__,
                    )
                    return timezone
            except Exception as e:
                logger.debug(
                    "Geolocation service failed",
                    service=service_func.__name__,
                    error=str(e),
                )
                continue

        # All services failed - use default
        logger.warning(
            "All geolocation services failed, using default timezone",
            default_timezone=cls.DEFAULT_TIMEZONE,
        )
        cls._cached_timezone = cls.DEFAULT_TIMEZONE
        return cls.DEFAULT_TIMEZONE

    @classmethod
    async def _try_worldtimeapi(cls, timeout: float) -> str | None:
        """Try worldtimeapi.org for timezone detection."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get("https://worldtimeapi.org/api/ip")
            if response.status_code == 200:
                data = response.json()
                return data.get("timezone")
        return None

    @classmethod
    async def _try_ipinfo(cls, timeout: float) -> str | None:
        """Try ipinfo.io for timezone detection."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get("https://ipinfo.io/json")
            if response.status_code == 200:
                data = response.json()
                return data.get("timezone")
        return None

    @classmethod
    async def _try_ipapi(cls, timeout: float) -> str | None:
        """Try ipapi.co for timezone detection."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get("https://ipapi.co/json/")
            if response.status_code == 200:
                data = response.json()
                return data.get("timezone")
        return None

    @classmethod
    def clear_cache(cls):
        """Clear cached timezone (useful for testing)."""
        cls._cached_timezone = None

    @classmethod
    def get_timezone_sync(cls, timeout: float = 3.0) -> str:
        """Synchronous wrapper for get_timezone."""
        try:
            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # We're in an event loop, but we need to run sync
                # For now, just return cached timezone or default
                if cls._cached_timezone:
                    return cls._cached_timezone
                else:
                    logger.warning(
                        "Cannot run async geolocation from sync context within event loop"
                    )
                    return cls.DEFAULT_TIMEZONE
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(cls.get_timezone(timeout))
        except Exception as e:
            logger.warning("Failed to get timezone synchronously", error=str(e))
            return cls.DEFAULT_TIMEZONE


# Global instance for easy access
geolocation_service = GeolocationService()
