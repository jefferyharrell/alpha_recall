# Alpha-Recall Short-Term Memory Implementation Checklist

## 1. Redis Database Implementation

- [ ] Create Redis database module
  - [ ] Implement `redis_db.py` with basic connection handling
  - [ ] Add Redis connection parameters to `.env.example`
  - [ ] Update environment variable loading in server startup
  - [ ] Implement TTL (Time To Live) configuration for short-term memory items

- [ ] Define short-term memory data structure
  - [ ] Design Redis key schema (e.g., `stm:{entity_name}:{timestamp}`)
  - [ ] Implement serialization/deserialization for observations
  - [ ] Define expiration policy (default TTL duration)

## 2. Database Interface Updates

- [ ] Update `base.py` with short-term memory interface
  - [ ] Add `add_shortterm_observation` method to `GraphDatabase` abstract class
  - [ ] Add `get_shortterm_observations` method to `GraphDatabase` abstract class

- [ ] Update `composite_db.py` to support short-term memory
  - [ ] Implement short-term memory methods in `CompositeDatabase` class
  - [ ] Add methods to store and retrieve short-term observations
  - [ ] Ensure proper error handling for Redis operations

## 3. MCP Tool Implementation

- [ ] Create `remember_shortterm` tool in `server.py`
  - [ ] Define parameters (entity, observation)
  - [ ] Implement validation logic
  - [ ] Add proper logging
  - [ ] Handle error cases

- [ ] Update `recall` tool to support short-term memory
  - [ ] Add optional parameter for including short-term memories
  - [ ] Implement logic to merge short-term and long-term results
  - [ ] Ensure proper ordering (newest short-term memories first)

## 4. Factory Updates

- [ ] Update `factory.py` to support Redis initialization
  - [ ] Add Redis client creation function
  - [ ] Update `create_db_instance` to include Redis in `CompositeDatabase`
  - [ ] Add Redis connection parameters from environment variables

## 5. Testing

- [ ] Create unit tests for Redis database module
  - [ ] Test connection handling
  - [ ] Test observation storage and retrieval
  - [ ] Test TTL expiration behavior

- [ ] Test `remember_shortterm` tool
  - [ ] Verify proper storage of observations
  - [ ] Verify TTL application
  - [ ] Test error handling

- [ ] Test `recall` with short-term memory integration
  - [ ] Verify proper retrieval of both memory types
  - [ ] Test ordering of results
  - [ ] Verify filtering capabilities

## 6. Documentation

- [ ] Update API documentation
  - [ ] Document `remember_shortterm` tool
  - [ ] Document updated `recall` parameters

- [ ] Update setup instructions
  - [ ] Add Redis installation/configuration steps
  - [ ] Document new environment variables

## 7. Deployment

- [ ] Update deployment configuration
  - [ ] Add Redis to deployment stack
  - [ ] Configure Redis persistence settings
  - [ ] Set up monitoring for Redis

- [ ] Test in production environment
  - [ ] Verify Redis connection in production
  - [ ] Monitor memory usage
  - [ ] Verify TTL behavior in production
