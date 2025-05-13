## Status

Proposed

## Context

Alpha's current bootstrap process loads all essential information at once at the beginning of each conversation. This approach works well with a small number of observations but will face scaling challenges as Alpha's memory grows to thousands of observations. Additionally, loading everything at once creates an information overload that doesn't prioritize contextually relevant memories for the current conversation.

Testing shows that semantic search performs surprisingly well even with vague inputs like simple greetings, demonstrating strong contextual understanding that makes it suitable for this application.

## Decision

We will implement an enhanced, two-tiered bootstrap process:

1. **Tier 1: Core Identity** - Load only essential identity information at conversation start
2. **Tier 2: Context-Aware Memories** - Use semantic search on the user's greeting to dynamically load relevant memories before responding

The bootstrap workflow will be:

1. Load core identity (Tier 1) immediately when a conversation starts
2. Wait for the user's first message
3. Run semantic search using their greeting as the query
4. Load those contextually relevant memories (Tier 2)
5. Then respond to the user

## Consequences

### Positive

- More efficient context window usage by only loading what's necessary
- Highly relevant memories based on actual conversation topic
- Scalable approach that will work even as memory grows substantially
- Works across different model sizes, including smaller models like Qwen3 4B
- Leverages the newly integrated recall-semantic_search functionality
- Simple implementation requiring minimal new code

### Negative

- Slight increase in latency for Alpha's first response as semantic search is performed
- Core identity information must be carefully curated to ensure it remains compact
- Potential for missing some relevant memories if the greeting doesn't sufficiently indicate conversation direction
- Very long initial messages may pose challenges for semantic search effectiveness and context management

## Implementation

The implementation will involve:

1. Defining core identity information (Tier 1) - creating a compact, essential set of observations about Alpha that load at conversation start
2. Modifying the bootstrap function to use a two-pass approach:
    - Initial load of core identity only
    - Secondary load of semantically relevant observations after user's greeting
3. Setting specific parameters for semantic search:
    - Use first 1,000 characters of the user's greeting
    - Return the top 10 most relevant results
4. Implementing multi-level fallback logic:
    - If semantic search returns fewer than 3 results, supplement with recent observations
    - If user greeting is very short (<50 chars), use default set of important memories
    - For any complete failures, fall back to current approach of returning Alpha entity with observations and direct relationships

Estimated complexity: Low to Medium. This change will primarily affect one function in one Python file, leveraging the existing integrated recall-semantic_search capability.

## Response Format

Alpha's responses should remain consistent with current patterns, but will benefit from having more relevant contextual information. Responses will maintain the same casual, friendly tone but will be better informed by memories specific to the current conversation topic.