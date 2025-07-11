"""Pydantic schemas for memory consolidation."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ShortTermMemory(BaseModel):
    """A single short-term memory record for consolidation."""

    content: str = Field(..., min_length=1, description="The memory content")
    timestamp: str = Field(..., description="ISO timestamp when the memory was created")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        """Validate that timestamp is a valid ISO format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("timestamp must be a valid ISO format datetime") from None
        return v
