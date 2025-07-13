"""
Gentle refresh tool that provides temporal orientation in human-readable prose format.

Provides comprehensive context for AI agents including current time, core identity,
personality traits, and recent memories in natural language prose format for
optimal tokenization efficiency.
"""

from fastmcp import FastMCP
from jinja2 import Template

from ..config import settings
from ..logging import get_logger
from ..services.geolocation import GeolocationService
from ..services.memgraph import get_memgraph_service
from ..services.redis import get_redis_service
from ..services.time import time_service
from ..services.tokenizer import tokenizer
from ..utils.correlation import generate_correlation_id, set_correlation_id

__all__ = ["gentle_refresh", "register_gentle_refresh_tools"]

# Jinja2 template for prose output
PROSE_TEMPLATE = Template(
    """
Good {{ time_greeting }} and welcome to {{ location }} where it is {{ time.iso_datetime }} and the local time is {{ time.human_readable }} {{ time.timezone.display }}.
{% if self_prompt %}

## Self-Prompt

{{ self_prompt }}
{% endif %}
{% for context_key, context_content in context_blocks.items() %}

## {{ context_key|title|replace('_', ' ') }}

{{ context_content }}
{% endfor %}

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
""".strip()
)


def calculate_content_for_budget(
    token_budget: int,
    identity_facts: list,
    personality_data: dict,
    self_prompt: str | None = None,
    context_blocks: dict | None = None,
) -> dict:
    """Calculate how much content fits in the token budget.

    Args:
        token_budget: Maximum tokens to use
        identity_facts: Core identity facts for base cost calculation
        personality_data: Personality traits for base cost calculation
        self_prompt: Optional self-prompt content for base cost calculation
        context_blocks: Optional context blocks for base cost calculation

    Returns:
        Dict with stm_limit and obs_limit
    """
    # Estimate base template cost (time + location + self-prompt + identity + personality)
    self_prompt_section = ""
    if self_prompt:
        self_prompt_section = f"""

## Self-Prompt

{self_prompt}
"""

    context_blocks_section = ""
    if context_blocks:
        for key, content in context_blocks.items():
            context_blocks_section += f"""

## {key.replace('_', ' ').title()}

{content}
"""

    base_text = f"""Good morning and welcome to Los Angeles where it is 2025-07-13T14:00:00+00:00 and the local time is Sunday, July 13, 2025 7:00 AM PDT.{self_prompt_section}{context_blocks_section}

## Core Identity
{' '.join([fact['content'] + '.' for fact in identity_facts])}

## Personality Traits
{len(personality_data)} traits with directives
"""

    # Add rough personality cost (traits + directives)
    personality_cost = 0
    for trait_name, trait in personality_data.items():
        personality_cost += len(trait_name) + len(trait.get("description", ""))
        for directive in trait.get("directives", []):
            personality_cost += len(directive.get("instruction", ""))

    base_cost = tokenizer.count(base_text) + (personality_cost // 4)

    # Reserve some buffer for template formatting
    template_overhead = 200
    total_base_cost = base_cost + template_overhead

    # Remaining budget for memories and observations
    memory_budget = max(0, token_budget - total_base_cost)

    # Estimate per-item costs
    tokens_per_stm = 100  # Average short-term memory
    tokens_per_obs = 50  # Average observation

    # Calculate limits with preference for STMs over observations
    max_stms = min(memory_budget // tokens_per_stm, 100)  # Cap at 100 STMs
    remaining_after_stms = memory_budget - (max_stms * tokens_per_stm)
    max_obs = min(remaining_after_stms // tokens_per_obs, 20)  # Cap at 20 observations

    return {
        "stm_limit": max_stms,
        "obs_limit": max_obs,
        "base_cost": total_base_cost,
        "memory_budget": memory_budget,
    }


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

        # Get Redis service for identity and self-prompt
        logger.info("Loading dynamic identity facts and self-prompt from Redis")
        redis_service = get_redis_service()
        identity_facts = redis_service.get_identity_facts()

        # Get self-prompt (dynamic prompt injection)
        self_prompt_result = redis_service.get_self_prompt()
        self_prompt = None
        if self_prompt_result.get("success") and self_prompt_result.get("message"):
            self_prompt = self_prompt_result["message"]
            logger.info("Loaded self-prompt", message_length=len(self_prompt))

        # Get context blocks (modular context management)
        context_blocks_result = redis_service.get_all_context_blocks()
        context_blocks = {}
        if context_blocks_result.get("success"):
            context_blocks = context_blocks_result.get("context_blocks", {})
            logger.info(
                "Loaded context blocks",
                count=len(context_blocks),
                blocks=list(context_blocks.keys()),
            )

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

        # Get recent short-term memories with generous limit for token budgeting
        try:
            redis_service = get_redis_service()
            # Use a generous limit - we'll trim based on token budget later
            token_budget = (
                tokens if tokens is not None else settings.gentle_refresh_default_tokens
            )
            max_possible_stms = max(100, token_budget // 50)  # Generous estimate
            shortterm_limit = min(max_possible_stms, 200)  # Cap at 200 for sanity
            logger.info(
                "Retrieving recent short-term memories",
                limit=shortterm_limit,
                budget=token_budget,
            )

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

        # Apply token budgeting
        token_budget = (
            tokens if tokens is not None else settings.gentle_refresh_default_tokens
        )
        logger.info("Calculating content for token budget", budget=token_budget)

        # Calculate content limits based on token budget
        content_limits = calculate_content_for_budget(
            token_budget, identity_facts, personality_data, self_prompt, context_blocks
        )

        # Limit memories and observations based on budget
        original_stm_count = len(shortterm_memories)
        original_obs_count = len(recent_observations)

        shortterm_memories = shortterm_memories[: content_limits["stm_limit"]]
        recent_observations = recent_observations[: content_limits["obs_limit"]]

        logger.info(
            "Token budgeting applied",
            base_cost=content_limits["base_cost"],
            memory_budget=content_limits["memory_budget"],
            stm_limit=content_limits["stm_limit"],
            obs_limit=content_limits["obs_limit"],
            original_stm_count=original_stm_count,
            original_obs_count=original_obs_count,
            final_stm_count=len(shortterm_memories),
            final_obs_count=len(recent_observations),
        )

        # Render the template
        prose_output = PROSE_TEMPLATE.render(
            time=time_data,
            time_greeting=time_greeting,
            location=location,
            self_prompt=self_prompt,
            context_blocks=context_blocks,
            core_identity=core_identity,
            personality=personality_data,
            shortterm_memories=shortterm_memories,
            recent_observations=recent_observations,
        )

        logger.info(
            "Gentle refresh completed successfully",
            core_identity_loaded=core_identity is not None,
            self_prompt_loaded=self_prompt is not None,
            context_blocks_count=len(context_blocks),
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
