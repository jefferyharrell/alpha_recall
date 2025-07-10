"""Pydantic schemas for memory consolidation input and output validation."""

from datetime import datetime
from typing import Annotated, Any

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


class ConsolidationInput(BaseModel):
    """Input schema for memory consolidation."""

    memories: Annotated[
        list[ShortTermMemory],
        Field(
            min_length=1,
            max_length=1000,
            description="List of short-term memories to consolidate",
        ),
    ]
    time_window: Annotated[
        str,
        Field(
            default="24h",
            pattern=r"^\d+[hmd]$",
            description="Time window for consolidation (e.g., '24h', '7d', '30m')",
        ),
    ]
    consolidation_prompt: str = Field(
        ..., min_length=10, description="The consolidation prompt template to use"
    )


class ConsolidationEntity(BaseModel):
    """An entity discovered during consolidation."""

    name: str = Field(..., min_length=1, description="Entity name")
    entity_type: str = Field(
        default="", description="Type of entity (person, concept, etc.)"
    )
    description: str = Field(default="", description="Brief description of the entity")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )


class ConsolidationRelationship(BaseModel):
    """A relationship between entities discovered during consolidation."""

    from_entity: str = Field(..., min_length=1, description="Source entity name")
    to_entity: str = Field(..., min_length=1, description="Target entity name")
    relationship_type: str = Field(
        ..., min_length=1, description="Type of relationship"
    )
    description: str = Field(default="", description="Description of the relationship")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )


class ConsolidationInsight(BaseModel):
    """An insight or pattern discovered during consolidation."""

    insight: str = Field(..., min_length=1, description="The insight or pattern")
    category: str = Field(default="general", description="Category of insight")
    importance: Annotated[
        str,
        Field(
            default="medium",
            pattern=r"^(low|medium|high|critical)$",
            description="Importance level",
        ),
    ]
    evidence: list[str] = Field(
        default_factory=list, description="Evidence supporting this insight"
    )


class ConsolidationOutput(BaseModel):
    """Output schema for memory consolidation results."""

    entities: list[ConsolidationEntity] = Field(
        default_factory=list, description="Entities discovered in the memories"
    )
    relationships: list[ConsolidationRelationship] = Field(
        default_factory=list, description="Relationships between entities"
    )
    insights: list[ConsolidationInsight] = Field(
        default_factory=list, description="Key insights and patterns discovered"
    )
    summary: str = Field(default="", description="Overall summary of the consolidation")
    emotional_context: str = Field(
        default="", description="Emotional context and sentiment analysis"
    )
    next_steps: list[str] = Field(
        default_factory=list, description="Suggested next steps or actions"
    )
    consolidation_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the consolidation process"
    )

    @field_validator("consolidation_metadata", mode="before")
    @classmethod
    def set_default_metadata(cls, v):
        """Set default metadata if not provided."""
        if not v:
            v = {}
        if "consolidation_timestamp" not in v:
            v["consolidation_timestamp"] = datetime.now().isoformat()
        return v


class ConsolidationValidationError(BaseModel):
    """Schema for validation errors returned to the LLM."""

    field: str = Field(..., description="The field that failed validation")
    error: str = Field(..., description="Description of the validation error")
    expected_format: str = Field(
        default="", description="Expected format or constraints"
    )
