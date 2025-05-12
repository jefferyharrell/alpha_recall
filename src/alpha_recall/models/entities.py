"""
Data models for alpha_recall entities and observations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    Model representing an observation about an entity.
    """
    content: str = Field(..., description="The content of the observation")
    created_at: datetime = Field(default_factory=datetime.now, description="When the observation was created")


class Relationship(BaseModel):
    """
    Model representing a relationship between entities.
    """
    source: str = Field(..., description="The source entity name")
    target: str = Field(..., description="The target entity name")
    type: str = Field(..., description="The type of relationship")


class Entity(BaseModel):
    """
    Model representing an entity in the knowledge graph.
    """
    name: str = Field(..., description="The name of the entity")
    type: Optional[str] = Field(default="Entity", description="The type of entity")
    observations: List[Observation] = Field(default_factory=list, description="Observations about the entity")
    relationships: Optional[List[Relationship]] = Field(default_factory=list, description="Relationships to other entities")
    importance: Optional[Dict[str, Any]] = Field(default=None, description="Importance metrics for the entity")
