"""Token counting service for dynamic context optimization."""

from ..logging import get_logger

logger = get_logger(__name__)


class Tokenizer:
    """Token estimation service for optimizing model context usage.

    Provides a clean interface for token counting that can be upgraded
    from simple heuristics to actual tokenization as needed.
    """

    def count(self, text: str) -> int:
        """Estimate token count for given text.

        Current implementation uses a simple heuristic of ~4 characters per token,
        which is a reasonable approximation for most modern LLMs.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Simple heuristic: ~4 characters per token
        estimated_tokens = len(text) // 4

        # Ensure we return at least 1 token for non-empty text
        return max(1, estimated_tokens)


# Global tokenizer instance
tokenizer = Tokenizer()
