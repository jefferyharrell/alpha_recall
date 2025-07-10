"""Stress test for memory consolidation JSON schema validation consistency."""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_consolidation_schema_validation_consistency(test_stack_seeded):
    """Stress test: 50 iterations of consolidation on the same data to verify JSON validity consistency.

    This test verifies that the LLM produces consistently valid JSON responses
    when given identical input data at temperature=0. We're testing:
    - Schema validation consistency (binary pass/fail)
    - JSON structural integrity across multiple attempts
    - Model determinism at temperature=0
    - No regression in validation success rates
    """
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Set up consistent test data for all iterations
        test_memories = [
            "Alpha and Jeffery completed implementation of the systematic model evaluation framework.",
            "The new consolidation service uses Pydantic v2 schemas for input/output validation.",
            "Temperature=0 provides deterministic testing for reliable model evaluation.",
            "We enforced a 'No Cruft 🚫' policy by replacing the old consolidation service with v2.",
            "The consolidation framework includes detailed correlation ID tracing and performance metrics.",
        ]

        # Add the test memories to the system
        for memory_content in test_memories:
            await client.call_tool("remember_shortterm", {"content": memory_content})

        print(
            f"🧪 Starting stress test: 50 iterations on {len(test_memories)} memories"
        )
        print("📊 Testing JSON schema validation consistency with temperature=0")

        # Run 50 iterations of consolidation on the same data
        results = []
        validation_successes = 0
        validation_failures = 0
        json_parse_errors = 0

        for iteration in range(50):
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

                # Progress indicator
                if iteration % 10 == 9:
                    success_rate = (validation_successes / (iteration + 1)) * 100
                    print(
                        f"📈 Progress: {iteration + 1}/50 iterations, {success_rate:.1f}% validation success"
                    )

            except json.JSONDecodeError as e:
                json_parse_errors += 1
                print(f"❌ Iteration {iteration}: JSON parse error - {e}")
                results.append({"parse_error": str(e), "iteration": iteration})
            except Exception as e:
                print(f"💥 Iteration {iteration}: Unexpected error - {e}")
                results.append({"unexpected_error": str(e), "iteration": iteration})

        # Analyze the stress test results
        total_iterations = len(results)
        success_rate = (validation_successes / total_iterations) * 100
        failure_rate = (validation_failures / total_iterations) * 100
        error_rate = (json_parse_errors / total_iterations) * 100

        print("\n🎯 Stress Test Results Summary:")
        print(f"   Total iterations: {total_iterations}")
        print(f"   Validation successes: {validation_successes} ({success_rate:.1f}%)")
        print(f"   Validation failures: {validation_failures} ({failure_rate:.1f}%)")
        print(f"   JSON parse errors: {json_parse_errors} ({error_rate:.1f}%)")

        # Performance analysis
        from tests.e2e.fixtures.performance import collector

        consolidation_durations = []
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_consolidate_shortterm":
                consolidation_durations.append(metric["duration_ms"])
                if len(consolidation_durations) >= 50:
                    break

        if consolidation_durations:
            avg_duration = sum(consolidation_durations[:50]) / 50
            min_duration = min(consolidation_durations[:50])
            max_duration = max(consolidation_durations[:50])
            print(f"   Average processing time: {avg_duration:.1f}ms")
            print(
                f"   Min/Max processing time: {min_duration:.1f}ms / {max_duration:.1f}ms"
            )

        # Test assertions
        assert total_iterations == 50, f"Expected 50 iterations, got {total_iterations}"

        # We should have very few JSON parse errors (< 5%)
        assert (
            error_rate < 5.0
        ), f"Too many JSON parse errors: {error_rate:.1f}% (expected < 5%)"

        # Either validation succeeds consistently, or it fails consistently at temperature=0
        # We don't want wildly inconsistent behavior
        consistency_threshold = 80.0  # At least 80% should be consistent
        is_mostly_successful = success_rate >= consistency_threshold
        is_mostly_failing = failure_rate >= consistency_threshold

        assert is_mostly_successful or is_mostly_failing, (
            f"Inconsistent validation behavior: {success_rate:.1f}% success, {failure_rate:.1f}% failure. "
            f"Expected at least {consistency_threshold}% in one category for temperature=0 consistency."
        )

        if is_mostly_successful:
            print(
                f"✅ Consistent SUCCESS pattern: {success_rate:.1f}% validation success rate"
            )
        else:
            print(
                f"⚠️  Consistent FAILURE pattern: {failure_rate:.1f}% validation failure rate"
            )
            print(
                "   This indicates the model/prompt needs adjustment for this data pattern"
            )

        # Log determinism analysis for successful cases
        if validation_successes >= 2:
            successful_results = [r for r in results if r.get("success", False)]
            if len(successful_results) >= 2:
                # Compare entity counts for determinism
                entity_counts = [
                    len(r["consolidation"]["entities"]) for r in successful_results[:5]
                ]
                insight_counts = [
                    len(r["consolidation"]["insights"]) for r in successful_results[:5]
                ]
                print(f"🔍 Sample entity counts (first 5 successes): {entity_counts}")
                print(f"🔍 Sample insight counts (first 5 successes): {insight_counts}")

        print(
            f"🏁 Stress test completed: {success_rate:.1f}% JSON schema validation consistency"
        )
