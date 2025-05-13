---
tags:
  - alpha-recall
  - ADR
---

## Status

Accepted

## Date

2025-05-12

## Context

The current alpha-recall implementation provides two separate tools for memory retrieval:

1. `recall` - Retrieves information about specific entities from the knowledge graph
2. `semantic_search` - Searches for observations semantically similar to a natural language query

While functional, this separation creates a cognitive burden when deciding which tool to use. Additionally, the system prompt needs to be updated to include documentation for the `semantic_search` tool, which adds complexity.

The multi-model architecture of Alpha requires the memory interface to be intuitive and consistent across different model sizes (from 4B parameters up to much larger models). Smaller models like Qwen3 4B can sometimes struggle with choosing between similar tools or with complex syntax.

## Decision

We will enhance and simplify the memory retrieval interface by merging the functionality of `semantic_search` into an improved `recall` tool. The enhanced `recall` will:

1. Accept a single query parameter that can be either an entity name or a semantic search query
2. Return a combined result set containing:
    - An exact entity match (if the query matches an entity name)
    - Top N semantically similar observations (regardless of whether an entity match was found)
3. Balance the number of semantic results (N) to maximize utility while minimizing token consumption

## Consequences

### Positive

- **Simplified Interface**: Reduces cognitive load by consolidating two similar functions into one
- **Natural Interaction**: Better mimics human memory by associatively retrieving both specific and related information
- **Enhanced Context**: Provides additional context even when looking up specific entities
- **Serendipitous Discovery**: Facilitates unexpected connections between concepts
- **Improved System Prompt**: Simplifies documentation and reduces token usage in the system prompt
- **Better Multi-Model Support**: Easier for smaller models to use consistently
- **Progressive Enhancement**: Maintains a simple interface while adding sophisticated functionality "under the hood"

### Negative

- **Implementation Complexity**: Slightly more complex to implement than separate tools
- **Response Size Management**: Careful tuning needed to balance the number of semantic results
- **Potential Confusion**: Users might initially be confused by receiving both types of results
- **Performance Considerations**: Each query will now potentially trigger both exact match and semantic search operations

### Neutral

- **API Signature Change**: Existing code using the `recall` function will need to be updated

## Implementation Plan

1. Modify the `recall` function to accept a query parameter instead of an entity parameter
2. Update internal logic to:
    - Check if the query matches an entity name exactly
    - If match found, retrieve entity with specified depth
    - Perform semantic search using the query
    - Combine results with entity data (if found) as the primary result
    - Include top N semantic search results as secondary information
3. Determine appropriate value for N based on testing
4. Deprecate the standalone `semantic_search` function
5. Update system prompt to document the enhanced `recall` functionality
6. Test across multiple models, particularly focusing on smaller models like Qwen3 4B

## Response Format

```json
{
  "query": "example query",
  "exact_match": {
    // Entity data if found
    "name": "Entity Name",
    "type": "Entity Type",
    "observations": [...],
    "relationships": [...]
  },
  "semantic_results": [
    // Top N semantically similar observations
    {
      "entity": "Related Entity",
      "observation": "Related observation text",
      "score": 0.87
    },
    // Additional results...
  ],
  "success": true
}
```

## Example Usage

```python
# Looking up a specific entity with related semantic information
recall(query="Sparkle", depth=1)

# Semantic search with no specific entity in mind
recall(query="cat behavior and preferences")

# Exploratory query that might match an entity
recall(query="Project Alpha")
```

## Notes

This design follows the Progressive Enhancement philosophy outlined in the alpha-recall specification. It maintains a simple interface while adding sophisticated functionality that evolves naturally as the system grows.

The combined approach is inspired by how human memory works - retrieving specific concepts while automatically activating related ideas and associations.

## Reviewers

- Jeffery Harrell
- Alpha