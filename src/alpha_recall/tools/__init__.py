"""Alpha-Recall MCP Tools."""

from .browse_longterm import register_browse_longterm_tools
from .browse_narrative import register_browse_narrative_tools
from .browse_shortterm import register_browse_shortterm_tool
from .gentle_refresh import register_gentle_refresh_tools
from .get_entity import register_get_entity_tools
from .get_relationships import register_get_relationships_tools
from .health import register_health_tools
from .recall_narrative import register_recall_narrative_tools
from .relate_longterm import register_relate_longterm_tools
from .remember_longterm import register_remember_longterm_tools
from .remember_narrative import register_remember_narrative_tools
from .remember_shortterm import register_remember_shortterm_tool
from .search_all_memories import register_search_all_memories_tools
from .search_longterm import register_search_longterm_tools
from .search_narratives import register_search_narratives_tools
from .search_shortterm import register_search_shortterm_tool

__all__ = [
    "register_health_tools",
    "register_gentle_refresh_tools",
    "register_remember_shortterm_tool",
    "register_browse_shortterm_tool",
    "register_search_shortterm_tool",
    "register_remember_longterm_tools",
    "register_relate_longterm_tools",
    "register_search_all_memories_tools",
    "register_search_longterm_tools",
    "register_get_entity_tools",
    "register_get_relationships_tools",
    "register_browse_longterm_tools",
    "register_remember_narrative_tools",
    "register_search_narratives_tools",
    "register_recall_narrative_tools",
    "register_browse_narrative_tools",
]
