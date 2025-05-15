# Alpha-Recall Redis-Based Short-Term Memory Implementation Checklist

## 1. Redis Database Module

- [ ] Create `redis_db.py` for short-term memory operations
  - [ ] Implement Redis connection handling with proper error management
  - [ ] Add Redis connection parameters to `.env.example` (host, port, password if needed)
  - [ ] Create key schema using `alpha:stm:<timestamp>-<random>` format
  - [ ] Implement TTL configuration (2 minutes for testing, 72 hours for production)
  - [ ] Add serialization/deserialization for memory content and metadata

## 2. Database Interface Updates

- [ ] Update `base.py` with short-term memory interface
  - [ ] Add `remember_shortterm(content)` method to abstract base class
  - [ ] Add `get_shortterm_memories(through_the_last=None)` method to abstract base class

- [ ] Update `composite_db.py` to integrate Redis operations
  - [ ] Implement the new short-term memory methods
  - [ ] Add client detection functionality to track memory source
  - [ ] Ensure proper error handling and fallback mechanisms

## 3. MCP Tool Implementation

- [ ] Create `remember_shortterm` tool in `server.py`
  - [ ] Accept content parameter without requiring entity name
  - [ ] Add timestamp and client information automatically
  - [ ] Implement proper logging and error handling

- [ ] Update `recall` tool to support short-term memory
  - [ ] Add `shortterm=True` parameter option
  - [ ] Implement `through_the_last` duration filter (e.g., '2h', '1d')
  - [ ] Ensure proper ordering (newest first)
  - [ ] Handle the case when both short-term and long-term memories are requested

- [ ] Enhance `refresh` function to include short-term memories
  - [ ] Return core identity entity
  - [ ] Include 5 most recent short-term memories
  - [ ] Include semantic search results based on query

## 4. Factory Updates

- [ ] Update `factory.py` to support Redis initialization
  - [ ] Add Redis client creation function
  - [ ] Update `create_db_instance` to include Redis in `CompositeDatabase`
  - [ ] Load Redis connection parameters from environment variables

## 5. Testing

- [ ] Create unit tests for Redis database module
  - [ ] Test connection handling and error cases
  - [ ] Test memory storage and retrieval operations
  - [ ] Verify TTL expiration behavior with short test durations

- [ ] Test MCP tools with integration tests
  - [ ] Verify `remember_shortterm` stores memories correctly
  - [ ] Test `recall` with various parameter combinations
  - [ ] Verify `refresh` returns the expected combined results
  - [ ] Test the `through_the_last` time filtering functionality

- [ ] Perform cross-instance testing
  - [ ] Verify memories created in one instance are visible in another
  - [ ] Test with multiple concurrent clients

## 6. Documentation

- [ ] Update API documentation
  - [ ] Document `remember_shortterm` tool
  - [ ] Document updated `recall` parameters
  - [ ] Document enhanced `refresh` function

- [ ] Update setup instructions
  - [ ] Document Redis configuration requirements
  - [ ] Update environment variable documentation

## 7. Deployment

- [ ] Verify Redis container in alpha-stack
  - [ ] Confirm proper networking between Redis and API server
  - [ ] Configure Redis persistence settings if needed

- [ ] Implement phased rollout
  - [ ] Start with short TTL (2 minutes) for initial testing
  - [ ] Monitor memory usage and performance
  - [ ] Gradually increase TTL to production value (72 hours)
  - [ ] Monitor and refine based on actual usage patterns