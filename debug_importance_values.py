"""Debug what importance values the LLM is actually producing."""

import json
import re

# Sample from the test output - let's see what the LLM is actually producing
sample_output = """{
  "entities": [
    {
      "name": "Alpha",
      "entity_type": "AI",
      "description": "Persistent memory consolidation AI"
    },
    {
      "name": "Rubber Baby Buggy",
      "entity_type": "placeholder",
      "description": "Template placeholder text that was discovered in the consolidation prompt"
    }
  ],
  "relationships": [
    {
      "from_entity": "Alpha",
      "to_entity": "Rubber Baby Buggy",
      "relationship_type": "discovered_during_investigation"
    }
  ],
  "insights": [
    {
      "insight": "The consolidation prompt template was just a placeholder",
      "category": "problem-solving",
      "importance": "significant"
    },
    {
      "insight": "Schema validation is working correctly to catch mismatches",
      "category": "technical",
      "importance": "important"
    }
  ],
  "summary": "Investigation revealed consolidation template issues",
  "emotional_context": "problem-solving and systematic debugging",
  "next_steps": [
    {
      "action": "Fix schema alignment",
      "description": "Update schema to match LLM natural output"
    }
  ]
}"""

try:
    data = json.loads(sample_output)
    print("🔍 Importance values the LLM is actually using:")
    for i, insight in enumerate(data.get("insights", [])):
        importance = insight.get("importance", "NOT_SET")
        print(f"  Insight {i+1}: '{importance}'")

        # Check if it matches our pattern
        pattern = r"^(low|medium|high|critical)$"
        if re.match(pattern, importance):
            print("    ✅ Matches schema pattern")
        else:
            print(f"    ❌ Does NOT match pattern '{pattern}'")

    print("\n📋 Valid values according to schema: low, medium, high, critical")
    print("🤖 LLM's natural language: significant, important, etc.")

except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
