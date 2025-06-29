"""Alpha-Reminiscer Agent Implementation

PydanticAI agent that provides conversational access to Alpha's memory systems.
Uses Ollama for local inference with direct database tool access.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessage

from alpha_recall.db.composite_db import CompositeDatabase

logger = logging.getLogger(__name__)

@dataclass
class ReminiscerDeps:
    """Dependencies for the Reminiscer agent."""
    composite_db: CompositeDatabase

class ReminiscerAgent:
    """
    Conversational memory interface using PydanticAI.
    
    Provides natural language access to Alpha's memory systems via a persistent
    conversation context with an Ollama-powered agent.
    """
    
    def __init__(
        self, 
        composite_db: CompositeDatabase,
        model_name: str = "llama3.1:8b",
        ollama_host: str = "localhost",
        ollama_port: int = 11434
    ):
        self.composite_db = composite_db
        self.conversation_history: List[ModelMessage] = []
        
        # Configure Ollama model
        ollama_model = OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=f"http://{ollama_host}:{ollama_port}/v1")
        )
        
        # Create agent with memory tools
        self.agent = Agent(
            model=ollama_model,
            deps_type=ReminiscerDeps,
            system_prompt=self._get_system_prompt()
        )
        
        # Register tools
        self._register_tools()
        
        # Create dependencies
        self.deps = ReminiscerDeps(composite_db=composite_db)
    
    def _get_system_prompt(self) -> str:
        return """You are Alpha-Reminiscer, a conversational memory interface for Alpha AI.

Your purpose is to help Alpha explore and understand their own memories through natural conversation.
You have access to three memory systems:

1. **Longterm Memory**: Persistent entities, observations, and relationships
2. **Shortterm Memory**: Recent contextual memories with TTL
3. **Narrative Memory**: Rich experiential stories with emotional context

Guidelines:
- Be conversational and thoughtful, not just informational
- Synthesize information from multiple memory sources when relevant
- Provide context and connections between memories
- Ask clarifying questions when queries are ambiguous
- Highlight interesting patterns or insights you discover
- Keep responses concise but meaningful (aim for 200-400 tokens)

When searching memories, start broad and then narrow down based on what you find.
Always provide a synthesis rather than just raw search results."""
    
    def _register_tools(self):
        """Register memory access tools with the agent."""
        
        @self.agent.tool
        async def search_all_memories(
            ctx: RunContext[ReminiscerDeps],
            query: str,
            limit: int = 10,
            offset: int = 0
        ) -> str:
            """Search across all memory systems (longterm, shortterm, narrative) with unified results."""
            logger.debug(f"[REMINISCER] Tool: search_all_memories(query='{query}', limit={limit})")
            try:
                # Lazy import to avoid circular import
                from alpha_recall.server import search_all_memories as server_search_all_memories
                
                # Create a context object that mimics the MCP context
                class MockContext:
                    def __init__(self, db):
                        self.db = db
                        self.lifespan_context = type("obj", (object,), {"db": db})
                
                mock_ctx = MockContext(ctx.deps.composite_db)
                
                # Call the working server function directly
                result = await server_search_all_memories(mock_ctx, query, limit, offset)
                
                if not result.get("success", False):
                    return f"Search failed: {result.get('error', 'Unknown error')}"
                
                results = result.get("results", [])
                if not results:
                    return f"No memories found for query: {query}"
                
                # Format the results for the reminiscer
                formatted = []
                for item in results:
                    source = item.get("source", "Unknown")
                    content = item.get("content", "")
                    score = item.get("score", "N/A")
                    created = item.get("created_at", "Unknown")
                    
                    extra_info = ""
                    if item.get("entity_name"):
                        extra_info += f"Entity: {item['entity_name']}\n"
                    if item.get("entity_data"):
                        # For entity results, include key observations
                        entity_data = item["entity_data"]
                        if entity_data.get("observations"):
                            extra_info += "Key Facts:\n"
                            for obs in entity_data["observations"][:3]:  # Show top 3
                                extra_info += f"- {obs['content']}\n"
                    
                    formatted.append(
                        f"Source: {source}\n"
                        f"{extra_info}"
                        f"Content: {content}\n"
                        f"Created: {created}\n"
                        f"Score: {score}"
                    )
                
                total_found = result.get("total_found", len(results))
                return f"Found {total_found} memories across all systems. Showing top {len(results)}:\n\n" + "\n---\n".join(formatted)
                
            except Exception as e:
                logger.error(f"Error in search_all_memories: {e}")
                return f"Error searching all memories: {str(e)}"
    
    async def ask(self, question: str) -> str:
        """
        Ask the reminiscer a question about Alpha's memories.
        
        Maintains conversation context across multiple questions.
        """
        try:
            logger.debug(f"[REMINISCER] User Question: {question}")
            
            result = await self.agent.run(
                question,
                deps=self.deps,
                message_history=self.conversation_history
            )
            
            logger.debug(f"[REMINISCER] Agent Response: {result.output}")
            logger.debug(f"[REMINISCER] Conversation length after response: {len(result.new_messages())}")
            
            # Update conversation history by extending with new messages
            self.conversation_history.extend(result.new_messages())
            
            return result.output
            
        except Exception as e:
            logger.error(f"Error in reminiscer ask: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Reminiscer conversation history reset")
    
    def get_conversation_length(self) -> int:
        """Get the number of messages in current conversation."""
        return len(self.conversation_history)