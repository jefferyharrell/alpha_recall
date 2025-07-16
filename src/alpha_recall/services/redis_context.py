"""Redis service for context block management."""

import json
import time
from typing import Any

import pendulum

from ..logging import get_logger
from ..utils.correlation import (
    create_child_correlation_id,
    get_correlation_id,
)
from .redis_base import RedisBaseService

logger = get_logger("services.redis_context")


class RedisContextService(RedisBaseService):
    """Service for managing modular context blocks in Redis."""

    def set_context_block(self, key: str, content: str) -> dict[str, Any]:
        """
        Set a context block for modular self-prompt management.

        Args:
            key: The context block key (e.g., 'autobiography', 'current_project')
            content: The content to store

        Returns:
            Dict with success status and operation metadata
        """
        correlation_id = get_correlation_id() or create_child_correlation_id(
            "set_context_block"
        )
        start_time = time.perf_counter()

        try:
            context_key = f"alpha:context:{key}"

            if content.strip() == "":
                # Empty content means remove the context block
                result = self.client.delete(context_key)
                operation = "removed"
                logger.info(
                    "Context block removed",
                    key=key,
                    context_key=context_key,
                    key_existed=bool(result),
                    correlation_id=correlation_id,
                )
            else:
                # Check if we're updating an existing block
                existing_data = self.client.get(context_key)
                now = pendulum.now("UTC")

                if existing_data:
                    # Try to parse as JSON to get the original created_at
                    try:
                        existing_json = json.loads(existing_data.decode("utf-8"))
                        created_at = existing_json.get("created_at", now.isoformat())
                    except (json.JSONDecodeError, AttributeError):
                        # Old format (plain string) - use now as created_at
                        created_at = now.isoformat()
                else:
                    # New context block
                    created_at = now.isoformat()

                # Store as JSON with timestamps
                context_data = {
                    "content": content,
                    "created_at": created_at,
                    "updated_at": now.isoformat(),
                }

                self.client.set(context_key, json.dumps(context_data))
                operation = "stored"
                logger.info(
                    "Context block stored",
                    key=key,
                    context_key=context_key,
                    content_length=len(content),
                    created_at=created_at,
                    updated_at=context_data["updated_at"],
                    correlation_id=correlation_id,
                )

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return {
                "success": True,
                "operation": operation,
                "key": key,
                "content_length": len(content) if operation == "stored" else 0,
                "operation_time_ms": operation_time_ms,
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Error setting context block",
                key=key,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return {
                "success": False,
                "error": f"Failed to set context block '{key}': {e}",
            }

    def get_context_block(self, key: str) -> dict[str, Any]:
        """
        Get a specific context block.

        Args:
            key: The context block key

        Returns:
            Dict with success status and content
        """
        correlation_id = get_correlation_id() or create_child_correlation_id(
            "get_context_block"
        )
        start_time = time.perf_counter()

        try:
            context_key = f"alpha:context:{key}"

            # Get the context block content
            content_bytes = self.client.get(context_key)

            if not content_bytes:
                operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return {
                    "success": True,
                    "key": key,
                    "content": None,
                    "has_content": False,
                    "operation_time_ms": operation_time_ms,
                }

            content_str = content_bytes.decode("utf-8")

            # Try to parse as JSON (new format)
            try:
                content_data = json.loads(content_str)
                content = content_data.get("content", content_str)
                created_at = content_data.get("created_at")
                updated_at = content_data.get("updated_at", created_at)
            except json.JSONDecodeError:
                # Old format (plain string) - treat as content with no timestamps
                content = content_str
                created_at = None
                updated_at = None

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.debug(
                "Context block retrieved",
                key=key,
                context_key=context_key,
                has_content=content is not None,
                content_length=len(content) if content else 0,
                has_timestamps=created_at is not None,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "success": True,
                "key": key,
                "content": content,
                "created_at": created_at,
                "updated_at": updated_at,
                "has_content": content is not None,
                "operation_time_ms": operation_time_ms,
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Error retrieving context block",
                key=key,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return {
                "success": False,
                "error": f"Failed to retrieve context block '{key}': {e}",
            }

    def list_context_blocks(self) -> dict[str, Any]:
        """
        List all available context blocks.

        Returns:
            Dict with success status and list of context blocks
        """
        correlation_id = get_correlation_id() or create_child_correlation_id(
            "list_context_blocks"
        )
        start_time = time.perf_counter()

        try:
            # Scan for all context block keys
            pattern = "alpha:context:*"
            cursor = 0
            all_keys = []

            while True:
                cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=100)
                all_keys.extend(keys)
                if cursor == 0:
                    break

            # Extract just the key names (remove prefix)
            context_blocks = []
            for key_bytes in all_keys:
                key_str = (
                    key_bytes.decode("utf-8")
                    if isinstance(key_bytes, bytes)
                    else key_bytes
                )
                if key_str.startswith("alpha:context:"):
                    block_key = key_str[14:]  # Remove "alpha:context:" prefix
                    context_blocks.append(block_key)

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.debug(
                "Context blocks listed",
                count=len(context_blocks),
                blocks=context_blocks,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "success": True,
                "context_blocks": sorted(context_blocks),
                "count": len(context_blocks),
                "operation_time_ms": operation_time_ms,
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Error listing context blocks",
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return {
                "success": False,
                "error": f"Failed to list context blocks: {e}",
            }

    def delete_context_block(self, key: str) -> dict[str, Any]:
        """
        Delete a specific context block.

        Args:
            key: The context block key to delete

        Returns:
            Dict with success status and operation metadata
        """
        correlation_id = get_correlation_id() or create_child_correlation_id(
            "delete_context_block"
        )
        start_time = time.perf_counter()

        try:
            context_key = f"alpha:context:{key}"

            # Delete the context block
            result = self.client.delete(context_key)
            key_existed = bool(result)

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "Context block deletion attempted",
                key=key,
                context_key=context_key,
                key_existed=key_existed,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "success": True,
                "key": key,
                "key_existed": key_existed,
                "operation_time_ms": operation_time_ms,
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Error deleting context block",
                key=key,
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return {
                "success": False,
                "error": f"Failed to delete context block '{key}': {e}",
            }

    def get_all_context_blocks(self) -> dict[str, Any]:
        """
        Get all context blocks with their content for template rendering.

        Returns:
            Dict with success status and all context blocks
        """
        correlation_id = get_correlation_id() or create_child_correlation_id(
            "get_all_context_blocks"
        )
        start_time = time.perf_counter()

        try:
            # First get the list of all context block keys
            list_result = self.list_context_blocks()
            if not list_result.get("success"):
                return list_result

            context_blocks = {}

            # Get content for each context block
            for key in list_result.get("context_blocks", []):
                content_result = self.get_context_block(key)
                if content_result.get("success") and content_result.get("content"):
                    context_blocks[key] = {
                        "content": content_result["content"],
                        "created_at": content_result.get("created_at"),
                        "updated_at": content_result.get("updated_at"),
                    }

            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.debug(
                "All context blocks retrieved",
                count=len(context_blocks),
                blocks=list(context_blocks.keys()),
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "success": True,
                "context_blocks": context_blocks,
                "count": len(context_blocks),
                "operation_time_ms": operation_time_ms,
            }

        except Exception as e:
            operation_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Error retrieving all context blocks",
                error=str(e),
                error_type=type(e).__name__,
                operation_time_ms=operation_time_ms,
                correlation_id=correlation_id,
            )
            return {
                "success": False,
                "error": f"Failed to retrieve all context blocks: {e}",
            }
