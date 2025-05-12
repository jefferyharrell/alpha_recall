---
tags:
  - alpha-recall
---


This document describes the core memory tools for Project Alpha, designed to provide a structured interface for interacting with the Neo4j knowledge graph.

## Core Tools

The memory system exposes three primary tools, each with a structured JSON interface:

1. `remember` - Creates or updates entities and observations
2. `relate` - Establishes relationships between entities
3. `recall` - Retrieves information from the knowledge graph

## Tool Specifications

### 1. `remember` Tool

**Description:**  
Creates or updates an entity in the knowledge graph with optional observations.

**Parameters:**

- `entity`: string (required) - The name of the entity to create or update
- `type`: string (optional) - The type of entity (Person, Place, Concept, etc.)
- `observation`: string (optional) - A fact or observation about the entity

**Returns:**

- Success message with entity information
- Error message if the operation fails

**Example Usage:**

```json
// Create a simple entity
{
  "entity": "Sparkplug Louise Mittenhaver",
  "type": "Person"
}

// Create an entity with an observation
{
  "entity": "Sparkle",
  "type": "Person",
  "observation": "Sparkle is an alias of Sparkplug Louise Mittenhaver"
}

// Add observation to existing entity
{
  "entity": "Sparkle",
  "observation": "Sparkle likes to eat bread"
}
```

### 2. `relate` Tool

**Description:**  
Creates a relationship between two entities in the knowledge graph.

**Parameters:**

- `entity`: string (required) - The source entity name
- `to_entity`: string (required) - The target entity name
- `as`: string (required) - The type of relationship (has, knows, located_in, etc.)

**Returns:**

- Success message with relationship information
- Error message if the operation fails

**Example Usage:**

```json
// Create a simple relationship
{
  "entity": "Sparkle",
  "to_entity": "Sparkplug Louise Mittenhaver", 
  "as": "alias"
}

// Create a different type of relationship
{
  "entity": "Sparkle",
  "to_entity": "Bread",
  "as": "likes"
}
```

### 3. `recall` Tool

**Description:**  
Retrieves information about an entity from the knowledge graph.

**Parameters:**

- `entity`: string (required) - The name of the entity to retrieve
- `depth`: number (optional, default=1) - How many relationship hops to include
    - 0: Only the entity itself
    - 1: Entity and direct relationships
    - 2+: Entity and extended network

**Returns:**

- Entity information including name, type, observations, and relationships (based on depth)
- Empty result if entity not found
- Error message if the operation fails

**Example Usage:**

```json
// Retrieve just an entity with no relationships
{
  "entity": "Sparkle",
  "depth": 0
}

// Retrieve entity with direct relationships
{
  "entity": "Sparkle"
}  // depth=1 is the default

// Retrieve entity with extended network
{
  "entity": "Sparkle",
  "depth": 2
}
```

## Implementation Notes

- These structured tools replace the previous natural language memory interface
- All parameters are passed as JSON objects
- The tools map directly to the underlying Pydantic models in the Neo4j MCP Server
- Advanced tools (create_entities, add_observations, etc.) are available in debug mode