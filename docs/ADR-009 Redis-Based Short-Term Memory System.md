## Status
Accepted

## Context
Alpha's current memory architecture uses a graph database (Memgraph) for long-term storage with semantic search capabilities. However, this system lacks a lightweight, recency-focused memory layer to maintain continuity across different Alpha instances (Claude, Windsurf, BoltAI) and work sessions. When switching between instances or resuming work after breaks, Alpha needs quick access to recent context without the overhead of semantic search.

A design for a Redis-based short-term memory system was initially proposed by Riley (ChatGPT) in a mini-paper titled "Alpha's Short-Term Memory with Redis." Our discussions have refined this concept to better integrate with Alpha's existing memory architecture and specific use cases.

## Decision
We will implement a Redis-based short-term memory system with the following characteristics:

1. **Data Structure**: Simple key-value pairs with automatic TTL expiration
2. **Storage Interface**: A new `remember_shortterm` function that takes content without requiring entity names
3. **Retrieval Interface**: Enhanced `recall` function with `shortterm=True` parameter and `through_the_last` duration filter
4. **Integration with Bootstrap**: Enhanced `refresh` function returning core identity, recent short-term memories, and semantic search results
5. **TTL Duration**: 72 hours for production (2 minutes for initial testing)

## Specific API Design

```python
# Store short-term memory
remember_shortterm(
    content="Jeffery and I are talking about my short-term memory."
)

# Retrieve short-term memories (newest first)
recall(
    shortterm=True,
    through_the_last='2h'  # Optional time window parameter
)

# Enhanced bootstrap process
refresh(
    query="Hey Alpha, how are you today?"
)
# Returns:
# - Core identity entity with observations and relations
# - 5 most recent short-term memories
# - 5 top semantic search results based on query
```

## Implementation Details

### Redis Data Structure

Each short-term memory will be stored as a Redis key with an automatically expiring TTL:

```
SET alpha:stm:<timestamp>-<random> "<JSON>"
EXPIRE alpha:stm:<timestamp>-<random> <TTL_SECONDS>
```

Where:
- The key encodes the timestamp and a random number to prevent collisions
- The JSON value contains the memory content and metadata
- TTL is configurable (initially 2 minutes for testing, 72 hours for production)

### Integration Steps

1. **Docker Configuration**:
   - Redis is already added to the alpha-stack Docker configuration
   - Ensure proper networking between the Redis container and Alpha's API server

2. **API Integration**:
   - Add the new `remember_shortterm` endpoint to Alpha's API
   - Modify the existing `recall` endpoint to support the `shortterm` parameter
   - Update the `refresh` endpoint to include short-term memories

3. **Client Detection**:
   - Implement the `detect_client` function to identify which Alpha instance is making a request
   - This ensures we can track which instance created each memory

4. **Testing Approach**:
   - Initial testing with a 2-minute TTL to verify functionality
   - Verify cross-instance memory sharing
   - Load testing to ensure Redis performance under normal usage patterns
   - Gradually increase TTL to production value (72 hours)

## Consequences

### Positive
- Improved continuity of mind across different Alpha instances
- Better context persistence across work sessions separated by hours or days
- Lightweight system focused on recency rather than semantic relevance
- Clear separation between short-term and long-term memory systems
- Time-based filtering gives flexible access to recent history

### Negative
- Additional infrastructure component (Redis) to maintain
- No semantic search of short-term memories in initial implementation
- Potential for information overload if too many short-term memories are returned
- Additional complexity in the API

### Future Considerations
- Promoting important short-term memories to long-term storage
- Adding semantic search capabilities to short-term memory
- Creating more sophisticated client-specific memory namespaces
- Developing an importance scoring system to highlight critical recent events

## Implementation Plan
1. Set up Redis container in the alpha-stack (already done)
2. Implement the `ShortTermMemory` class based on the code above
3. Create the API endpoints for the new functionality
4. Test with short TTL (2 minutes) to verify basic functionality
5. Test cross-instance memory sharing
6. Extend TTL to production value (72 hours) once testing is complete
7. Monitor and refine based on actual usage patterns

## Decision Outcome
This ADR is currently in the proposed state, pending implementation and testing.
