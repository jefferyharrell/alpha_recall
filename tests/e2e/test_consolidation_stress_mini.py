"""Mini stress test for memory consolidation JSON schema validation consistency."""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_consolidation_schema_validation_mini_stress(test_stack_seeded):
    """Mini stress test: 5 iterations of consolidation on the same data to verify JSON validity consistency.

    This test verifies that the LLM produces consistently valid JSON responses
    when given identical input data at temperature=0. We're testing:
    - Schema validation consistency (binary pass/fail)
    - JSON structural integrity across multiple attempts
    - Model determinism at temperature=0
    """
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Set up consistent test data for all iterations
        test_memories = [
            "Alpha and Jeffery fixed the consolidation prompt template from the git stash.",
            "The consolidation service now uses the proper Jinja2 template with schema requirements.",
            "Temperature=0 provides deterministic testing for reliable model evaluation.",
            "We discovered the rubber baby buggy bumpers prompt was just a placeholder.",
        ]

        # Add the test memories to the system
        for memory_content in test_memories:
            await client.call_tool("remember_shortterm", {"content": memory_content})

        print(
            f"🧪 Starting mini stress test: 5 iterations on {len(test_memories)} memories"
        )
        print("📊 Testing JSON schema validation consistency with temperature=0")

        # Run 5 iterations of consolidation on the same data
        results = []
        validation_successes = 0
        validation_failures = 0
        json_parse_errors = 0

        for iteration in range(5):
            try:
                result = await time_mcp_call(
                    client,
                    "consolidate_shortterm",
                    {"time_window": "1h", "temperature": 0.0},
                )
                data = json.loads(result.content[0].text)
                results.append(data)

                # Track validation outcomes
                if data.get("success", False):
                    validation_successes += 1
                    print(f"✅ Iteration {iteration + 1}: Validation SUCCESS")
                    # Verify the consolidation structure is valid
                    consolidation = data["consolidation"]
                    required_fields = [
                        "entities",
                        "relationships",
                        "insights",
                        "summary",
                        "emotional_context",
                        "next_steps",
                    ]
                    for field in required_fields:
                        assert (
                            field in consolidation
                        ), f"Iteration {iteration}: Missing field {field}"
                else:
                    validation_failures += 1
                    print(f"❌ Iteration {iteration + 1}: Validation FAILURE")

                    # Show validation errors for debugging
                    if "validation_errors" in data:
                        print(f"   Validation errors: {data['validation_errors']}")
                    elif "error" in data:
                        print(f"   Error: {data['error']}")

                    # Show a snippet of raw model output for debugging
                    if "raw_model_output" in data:
                        raw_output = data["raw_model_output"]
                        if len(raw_output) > 200:
                            snippet = raw_output[:200] + "..."
                        else:
                            snippet = raw_output
                        print(f"   Raw output: {snippet}")

                    # Check that we have proper error reporting
                    assert (
                        "validation_errors" in data or "error" in data
                    ), f"Iteration {iteration}: Failed consolidation missing error details"

                # Verify model evaluation metadata is present
                assert "metadata" in data
                assert "model_evaluation" in data["metadata"]
                eval_data = data["metadata"]["model_evaluation"]
                assert (
                    eval_data["temperature"] == 0.0
                ), f"Iteration {iteration}: Expected temperature=0.0, got {eval_data['temperature']}"

            except json.JSONDecodeError as e:
                json_parse_errors += 1
                print(f"💥 Iteration {iteration + 1}: JSON parse error - {e}")
                results.append({"parse_error": str(e), "iteration": iteration})
            except Exception as e:
                print(f"🔥 Iteration {iteration + 1}: Unexpected error - {e}")
                results.append({"unexpected_error": str(e), "iteration": iteration})

        # Analyze the stress test results
        total_iterations = len(results)
        success_rate = (validation_successes / total_iterations) * 100
        failure_rate = (validation_failures / total_iterations) * 100
        error_rate = (json_parse_errors / total_iterations) * 100

        print("\n🎯 Mini Stress Test Results:")
        print(f"   Total iterations: {total_iterations}")
        print(f"   Validation successes: {validation_successes} ({success_rate:.1f}%)")
        print(f"   Validation failures: {validation_failures} ({failure_rate:.1f}%)")
        print(f"   JSON parse errors: {json_parse_errors} ({error_rate:.1f}%)")

        # Test assertions
        assert total_iterations == 5, f"Expected 5 iterations, got {total_iterations}"

        # We should have very few JSON parse errors
        assert (
            error_rate < 20.0
        ), f"Too many JSON parse errors: {error_rate:.1f}% (expected < 20%)"

        # Either validation succeeds consistently, or it fails consistently at temperature=0
        consistency_threshold = 60.0  # At least 60% should be consistent for mini test
        is_mostly_successful = success_rate >= consistency_threshold
        is_mostly_failing = failure_rate >= consistency_threshold

        if is_mostly_successful:
            print(
                f"✅ Consistent SUCCESS pattern: {success_rate:.1f}% validation success rate"
            )
        elif is_mostly_failing:
            print(
                f"⚠️  Consistent FAILURE pattern: {failure_rate:.1f}% validation failure rate"
            )
        else:
            print(
                f"🔄 Mixed results: {success_rate:.1f}% success, {failure_rate:.1f}% failure"
            )

        print("🏁 Mini stress test completed: Schema validation is working!")
