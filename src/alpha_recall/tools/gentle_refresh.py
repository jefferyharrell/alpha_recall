"""
Gentle refresh tool that provides temporal orientation in human-readable prose format.

Provides comprehensive context for AI agents including current time, core identity,
personality traits, and recent memories in natural language prose format for
optimal tokenization efficiency.
"""

from fastmcp import FastMCP
from jinja2 import Template

from ..logging import get_logger
from ..services.geolocation import GeolocationService
from ..services.memgraph import get_memgraph_service
from ..services.redis import get_redis_service
from ..services.time import time_service
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["gentle_refresh", "register_gentle_refresh_tools"]

# Jinja2 template for prose output
PROSE_TEMPLATE = Template(
    """
Good {{ time_greeting }} and welcome to {{ location }} where it is {{ time.iso_datetime }} and the local time is {{ time.human_readable }} {{ time.timezone.display }}.

## Core Identity
{% for fact in core_identity.identity_facts %}
{{ fact.content }}.{% if not loop.last %} {% endif %}{% endfor %}

## Personality Traits
{% for trait_name, trait in personality.items() %}
**{{ trait_name|title|replace('_', ' ') }}** (weight: {{ trait.weight }}) - {{ trait.description }}
{%- for directive in trait.directives %}
- {{ directive.instruction }} (weight: {{ directive.weight }}){% endfor %}
{% endfor %}

## Recent Context

### Short-term Memories
*{{ shortterm_memories|length }} most recent memories*
{% for memory in shortterm_memories %}
{{ loop.index }}. {{ memory.content }}
   *{{ memory.created_at }}*
{% endfor %}

{% if recent_observations %}
### Recent Observations
{% for obs in recent_observations %}
- {{ obs.content }} ({{ obs.entity_name }})
{% endfor %}
{% endif %}

---
*This context summary contains {{ token_estimate }} estimated tokens*
""".strip()
)


async def gentle_refresh(tokens: int | None = None) -> str:
    """
    Gentle refresh tool for temporal orientation.

    Provides comprehensive context for AI agents including current time, core identity,
    personality traits, and recent memories in natural language prose format for
    optimal tokenization efficiency.

    Args:
        tokens: Optional token budget for output. If specified, will prioritize
               and trim content to fit within the budget.

    Returns:
        Markdown-formatted prose containing current context
    """
    logger = get_logger("tools.gentle_refresh")
    correlation_id = generate_correlation_id("gentle_refresh")
    set_correlation_id(correlation_id)

    logger.info("Gentle refresh tool called", tokens=tokens)

    try:
        # Get current time with timezone and location
        geolocation_service = GeolocationService()
        time_data = await time_service.now_async()
        location = await geolocation_service.get_location()

        # Determine time greeting based on local hour
        local_time = time_data.get("local")
        if local_time:
            from datetime import datetime

            local_dt = datetime.fromisoformat(local_time)
            hour = local_dt.hour

            if 5 <= hour < 12:
                time_greeting = "morning"
            elif 12 <= hour < 17:
                time_greeting = "afternoon"
            elif 17 <= hour < 21:
                time_greeting = "evening"
            else:
                time_greeting = "night"
        else:
            time_greeting = "day"

        # Load core identity from Redis - this is required!
        logger.info("Loading dynamic identity facts from Redis")
        redis_service = get_redis_service()
        identity_facts = redis_service.get_identity_facts()

        core_identity = {
            "name": "Alpha Core Identity",  # Static name, no need for settings
            "identity_facts": identity_facts,
        }

        if identity_facts:
            logger.info("Loaded identity facts from Redis", count=len(identity_facts))
        else:
            logger.warning("No identity facts found in Redis")

        # Load personality structure from Memgraph (same as gentle_refresh)
        try:
            memgraph_service = get_memgraph_service()
            logger.info("Loading hierarchical personality structure")

            # Graph traversal query: Agent_Personality -> Traits -> Directives
            personality_query = """
            MATCH (root:Agent_Personality)-[:HAS_TRAIT]->(trait:Personality_Trait)
            OPTIONAL MATCH (trait)-[:HAS_DIRECTIVE]->(directive:Personality_Directive)
            RETURN trait.name as trait_name,
                   trait.description as trait_description,
                   trait.weight as trait_weight,
                   directive.instruction as directive_instruction,
                   directive.weight as directive_weight
            ORDER BY trait.weight DESC, directive.weight DESC
            """

            personality_result = list(
                memgraph_service.db.execute_and_fetch(personality_query)
            )

            # Build hierarchical personality structure
            personality_traits = {}
            for row in personality_result:
                trait_name = row["trait_name"]
                trait_weight = row["trait_weight"]

                # Skip traits with weight of exactly 0.0
                if trait_weight == 0.0:
                    continue

                # Initialize trait if not seen before
                if trait_name not in personality_traits:
                    personality_traits[trait_name] = {
                        "description": row["trait_description"],
                        "weight": trait_weight,
                        "directives": [],
                    }

                # Add directive to trait (only if directive exists and weight != 0.0)
                if (
                    row["directive_instruction"] is not None
                    and row["directive_weight"] != 0.0
                ):
                    personality_traits[trait_name]["directives"].append(
                        {
                            "instruction": row["directive_instruction"],
                            "weight": row["directive_weight"],
                        }
                    )

            # Sort traits by weight for consistent ordering
            personality_data = dict(
                sorted(
                    personality_traits.items(),
                    key=lambda x: x[1]["weight"],
                    reverse=True,
                )
            )

            logger.info(
                f"Retrieved {len(personality_data)} personality traits with "
                f"{sum(len(trait['directives']) for trait in personality_data.values())} total directives"
            )

        except Exception as e:
            logger.error(f"Error loading personality directives: {e}")
            personality_data = {}

        # Check if the system appears to be uninitialized
        missing_components = []
        if not identity_facts:
            missing_components.append("identity facts (Redis)")
        if not personality_data:
            missing_components.append("personality configuration (Memgraph)")

        if missing_components:
            error_msg = (
                f"INITIALIZATION ERROR: Alpha-Recall is missing critical components: {', '.join(missing_components)}. "
                "Please initialize the system by adding identity facts (add_identity_fact tool) and/or "
                "personality traits (create_personality_trait tool) before calling gentle_refresh."
            )
            logger.warning(
                "System partially or completely uninitialized",
                missing=missing_components,
            )
            return error_msg

        # Get recent short-term memories (same logic as gentle_refresh)
        try:
            redis_service = get_redis_service()
            shortterm_limit = 10
            logger.info("Retrieving recent short-term memories", limit=shortterm_limit)

            # Get recent memory IDs from the sorted set
            memory_ids_with_scores = redis_service.client.zrevrange(
                "memory_index", 0, shortterm_limit - 1, withscores=True
            )

            shortterm_memories = []
            for memory_id_bytes, _timestamp in memory_ids_with_scores:
                memory_id = memory_id_bytes.decode("utf-8")
                memory_key = f"memory:{memory_id}"

                # Get memory data from hash
                memory_data = redis_service.client.hmget(
                    memory_key, ["content", "created_at", "client_name"]
                )

                if memory_data[0] is not None:  # Content exists
                    content = memory_data[0].decode("utf-8")
                    created_at = (
                        memory_data[1].decode("utf-8") if memory_data[1] else ""
                    )
                    client_name = (
                        memory_data[2].decode("utf-8") if memory_data[2] else "unknown"
                    )

                    shortterm_memories.append(
                        {
                            "content": content,
                            "created_at": created_at,
                            "client": {"client_name": client_name},
                        }
                    )

            logger.info("Retrieved short-term memories", count=len(shortterm_memories))

        except Exception as e:
            logger.error("Error retrieving short-term memories", error=str(e))
            shortterm_memories = []

        # Get recent observations (simplified - gentle_refresh has issues with this too)
        try:
            # For now, just use empty list since gentle_refresh also has errors here
            recent_observations = []
            logger.info(
                "Recent observations disabled for now due to compatibility issues"
            )

        except Exception as e:
            logger.error("Error retrieving recent observations", error=str(e))
            recent_observations = []

        # If token budget specified, apply prioritization and trimming
        if tokens:
            logger.info("Applying token budget", budget=tokens)
            # TODO: Implement smart trimming based on token count
            # For now, just log that we received a budget
            pass

        # Render the template
        prose_output = PROSE_TEMPLATE.render(
            time=time_data,
            time_greeting=time_greeting,
            location=location,
            core_identity=core_identity,
            personality=personality_data,
            shortterm_memories=shortterm_memories,
            recent_observations=recent_observations,
            token_estimate="[calculating...]",  # TODO: Add actual token counting
        )

        logger.info(
            "Gentle refresh completed successfully",
            core_identity_loaded=core_identity is not None,
            personality_traits_count=len(personality_data),
            shortterm_memories_count=len(shortterm_memories),
            recent_observations_count=len(recent_observations),
        )

        return prose_output

    except Exception as e:
        logger.error("Gentle refresh failed", error=str(e))
        raise


def register_gentle_refresh_tools(mcp: FastMCP) -> None:
    """Register gentle refresh tools with the MCP server."""
    logger = get_logger("tools.gentle_refresh")

    mcp.tool(gentle_refresh)

    logger.debug("Gentle refresh tools registered")
