# Alpha-Recall Redis-Based Short-Term Memory Implementation Checklist

## 1. Redis Database Module

- [x] Create `redis_db.py` for short-term memory operations
  - [x] Implement Redis connection handling with proper error management
  - [x] Add Redis connection parameters to `.env.example` (host, port, password if needed)
  - [x] Create key schema using `alpha:stm:<timestamp>-<random>` format
  - [x] Implement TTL configuration (2 minutes for testing, 72 hours for production)
  - [x] Add serialization/deserialization for memory content and metadata

## 2. Database Interface Updates

- [x] Update `base.py` with short-term memory interface
  - [x] Add `remember_shortterm(content)` method to abstract base class
  - [x] Add `get_shortterm_memories(through_the_last=None)` method to abstract base class

- [x] Update `composite_db.py` to integrate Redis operations
  - [x] Implement the new short-term memory methods
  - [x] Add client detection functionality to track memory source
  - [x] Ensure proper error handling and fallback mechanisms

## 3. MCP Tool Implementation

- [x] Create `remember_shortterm` tool in `server.py`
  - [x] Accept content parameter without requiring entity name
  - [x] Add timestamp and client information automatically
  - [x] Implement proper logging and error handling

- [x] Update `recall` tool to support short-term memory
  - [x] Add `shortterm=True` parameter option
  - [x] Implement `through_the_last` duration filter (e.g., '2h', '1d')
  - [x] Ensure proper ordering (newest first)
  - [x] Handle the case when both short-term and long-term memories are requested

- [x] Enhance `refresh` function to include short-term memories
  - [x] Return core identity entity
  - [x] Include 5 most recent short-term memories
  - [x] Include semantic search results based on query

## 4. Factory Updates

- [x] Update `factory.py` to support Redis initialization
  - [x] Add Redis client creation function
  - [x] Update `create_db_instance` to include Redis in `CompositeDatabase`
  - [x] Load Redis connection parameters from environment variables

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