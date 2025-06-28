"""Alpha-Reminiscer: Conversational Memory Interface

A PydanticAI-based conversational interface for Alpha's memory systems.
Provides natural language querying of longterm, shortterm, and narrative memories.
"""

from .agent import ReminiscerAgent

__all__ = ["ReminiscerAgent"]