"""
Alpha-Snooze: Memory consolidation system for Alpha-Recall.

This module provides memory processing capabilities that run during gentle_refresh()
to consolidate recent short-term memories into structured knowledge before Alpha
fully "wakes up" to new conversations.

The name "snooze" refers to a brief pause to consolidate memories - like hitting
snooze for a few moments to process and organize thoughts before fully awakening.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from alpha_recall.logging_utils import get_logger

logger = get_logger(__name__)

# Default configuration
DEFAULT_OLLAMA_HOST = "localhost"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_CONSOLIDATION_LIMIT = 10  # Number of recent STM entries to process (fallback)
DEFAULT_TIME_WINDOW = "24h"  # Time window for memory consolidation
DEFAULT_TIMEOUT = 30  # Seconds


class AlphaSnooze:
    """
    Memory consolidation system that processes recent short-term memories
    using a local LLM to extract entities, relationships, and insights.
    """

    def __init__(
        self,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        ollama_port: int = DEFAULT_OLLAMA_PORT,
        model: str = DEFAULT_MODEL,
        consolidation_limit: int = DEFAULT_CONSOLIDATION_LIMIT,
        time_window: str = DEFAULT_TIME_WINDOW,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the Alpha-Snooze memory consolidator.

        Args:
            ollama_host: Hostname of the Ollama server
            ollama_port: Port of the Ollama server
            model: Model name to use for consolidation
            consolidation_limit: Fallback limit if time_window processing fails
            time_window: Time window for memory consolidation (e.g., "24h", "12h", "1d")
            timeout: Request timeout in seconds
        """
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.model = model
        self.consolidation_limit = consolidation_limit
        self.time_window = time_window
        self.timeout = timeout
        self.base_url = f"http://{ollama_host}:{ollama_port}"

    async def is_available(self) -> bool:
        """
        Check if the Ollama service is available and the model is accessible.

        Returns:
            True if Alpha-Snooze can be used, False otherwise
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                # Check if Ollama is running
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status != 200:
                        logger.warning("Ollama service not available")
                        return False
                    
                    # Check if our model is available
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    if self.model not in models:
                        logger.warning(f"Model {self.model} not available in Ollama. Available: {models}")
                        return False
                    
                    logger.info(f"Alpha-Snooze available with model {self.model}")
                    return True
                    
        except Exception as e:
            logger.warning(f"Alpha-Snooze availability check failed: {e}")
            return False

    async def consolidate_memories(
        self, 
        short_term_memories: List[Dict[str, Any]] = None,
        db = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process recent short-term memories to extract structured insights.

        Args:
            short_term_memories: List of recent STM entries to process

        Returns:
            Dictionary containing extracted entities, relationships, insights, and summary
            or None if consolidation fails
        """
        if not short_term_memories:
            logger.info("No short-term memories to consolidate")
            return None

        # Process all memories from the time window (no artificial limit when using time windows)
        memories_to_process = short_term_memories
        logger.info(f"Processing {len(memories_to_process)} memories from last {self.time_window} for consolidation")

        # Format memories for the LLM
        memory_text = self._format_memories_for_llm(memories_to_process)
        
        # Create the consolidation prompt
        prompt = self._create_consolidation_prompt(memory_text)
        
        try:
            # Send to Ollama for processing
            consolidation_result = await self._send_to_ollama(prompt)
            
            if consolidation_result:
                # Parse the structured response
                parsed_result = self._parse_consolidation_result(consolidation_result)
                
                # Add metadata
                parsed_result["processed_memories_count"] = len(memories_to_process)
                parsed_result["consolidation_timestamp"] = datetime.now(timezone.utc).isoformat()
                parsed_result["model_used"] = self.model
                
                logger.info(f"Successfully consolidated {len(memories_to_process)} memories")
                return parsed_result
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            
        return None

    def _format_memories_for_llm(self, memories: List[Dict[str, Any]]) -> str:
        """Format short-term memories for LLM processing."""
        formatted_memories = []
        
        for i, memory in enumerate(memories, 1):
            content = memory.get("content", "")
            created_at = memory.get("created_at", "")
            client = memory.get("client", {})
            client_name = client.get("client_name", "unknown")
            
            formatted_memories.append(f"{i}. [{created_at}] ({client_name}) {content}")
        
        return "\n".join(formatted_memories)

    def _create_consolidation_prompt(self, memory_text: str) -> str:
        """Create the prompt for memory consolidation."""
        return f"""You are Alpha's memory consolidation system. Analyze these recent short-term memories and extract structured insights:

RECENT MEMORIES:
{memory_text}

Please extract and organize this information in JSON format:

{{
  "entities": [
    {{"name": "EntityName", "type": "Person|Project|Concept|Technology", "key_facts": ["fact1", "fact2"]}},
  ],
  "relationships": [
    {{"entity1": "EntityA", "relationship": "works_on|collaborates_with|implements|etc", "entity2": "EntityB"}},
  ],
  "insights": [
    "Key insight or pattern discovered from these memories",
  ],
  "summary": "Brief 2-3 sentence summary of what happened in these recent interactions",
  "emotional_context": "overall emotional tone - excited|frustrated|breakthrough|routine|etc",
  "next_steps": ["potential follow-up actions or areas of focus"]
}}

Focus on:
- Important entities (people, projects, technologies, concepts)
- Relationships between entities
- Key insights or patterns
- Emotional context and breakthrough moments
- What might be important for future conversations

Respond ONLY with valid JSON. Be concise but capture the essential information."""

    async def _send_to_ollama(self, prompt: str) -> Optional[str]:
        """Send prompt to Ollama and get response."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for structured output
                        "top_p": 0.9,
                    }
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "").strip()
                    else:
                        logger.error(f"Ollama request failed with status {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error sending request to Ollama: {e}")
            return None

    def _parse_consolidation_result(self, result: str) -> Dict[str, Any]:
        """Parse the JSON response from the LLM."""
        try:
            # Try to parse the response as JSON
            parsed = json.loads(result)
            
            # Validate required fields
            required_fields = ["entities", "relationships", "insights", "summary"]
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = []
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse consolidation result as JSON: {e}")
            # Fallback to a basic structure
            return {
                "entities": [],
                "relationships": [],
                "insights": [f"Raw consolidation output: {result[:200]}..."],
                "summary": "Memory consolidation completed but parsing failed",
                "emotional_context": "unknown",
                "next_steps": [],
                "parse_error": str(e)
            }


async def create_alpha_snooze_from_env() -> Optional[AlphaSnooze]:
    """
    Create an AlphaSnooze instance from environment variables.
    
    Returns:
        AlphaSnooze instance if configuration is available, None otherwise
    """
    # Check if alpha-snooze is enabled
    if not os.environ.get("ALPHA_SNOOZE_ENABLED", "").lower() in ("true", "1", "yes"):
        logger.info("Alpha-Snooze not enabled (ALPHA_SNOOZE_ENABLED not set to true)")
        return None
    
    # Get configuration from environment
    ollama_host = os.environ.get("ALPHA_SNOOZE_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    ollama_port = int(os.environ.get("ALPHA_SNOOZE_OLLAMA_PORT", DEFAULT_OLLAMA_PORT))
    model = os.environ.get("ALPHA_SNOOZE_MODEL", DEFAULT_MODEL)
    consolidation_limit = int(os.environ.get("ALPHA_SNOOZE_LIMIT", DEFAULT_CONSOLIDATION_LIMIT))
    time_window = os.environ.get("ALPHA_SNOOZE_TIME_WINDOW", DEFAULT_TIME_WINDOW)
    timeout = int(os.environ.get("ALPHA_SNOOZE_TIMEOUT", DEFAULT_TIMEOUT))
    
    # Create the instance
    snooze = AlphaSnooze(
        ollama_host=ollama_host,
        ollama_port=ollama_port,
        model=model,
        consolidation_limit=consolidation_limit,
        time_window=time_window,
        timeout=timeout,
    )
    
    # Check if it's available
    if await snooze.is_available():
        logger.info(f"Alpha-Snooze initialized with model {model}")
        return snooze
    else:
        logger.warning("Alpha-Snooze configured but not available")
        return None