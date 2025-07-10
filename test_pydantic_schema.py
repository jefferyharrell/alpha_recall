"""Test Pydantic JSON schema generation for consolidation."""

import json

from src.alpha_recall.schemas.consolidation import ConsolidationOutput

# Generate JSON schema from Pydantic model
schema = ConsolidationOutput.model_json_schema()

print("=== PYDANTIC GENERATED JSON SCHEMA ===")
print(json.dumps(schema, indent=2))

# Also test the model_json_schema method for human-readable format
print("\n=== FORMATTED FOR PROMPTS ===")
print(json.dumps(schema, indent=2))
