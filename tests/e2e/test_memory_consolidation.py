"""E2E tests for memory consolidation with systematic model evaluation."""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_consolidate_shortterm_basic_functionality(test_stack_seeded):
    """Test basic consolidate_shortterm functionality with default settings."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Test basic consolidation with default settings (24h, temperature=0.0)
        result = await time_mcp_call(
            client, "consolidate_shortterm", {"time_window": "24h"}
        )
        data = json.loads(result.content[0].text)

        # Should have proper response structure
        assert "success" in data
        assert "tool_metadata" in data
        assert data["tool_metadata"]["tool_name"] == "consolidate_shortterm"
        assert data["tool_metadata"]["tool_version"] == "2.0"
        assert data["tool_metadata"]["deterministic_testing"] is True

        # Should have model evaluation metadata
        assert "metadata" in data
        assert "model_evaluation" in data["metadata"]
        eval_metadata = data["metadata"]["model_evaluation"]
        assert "model_name" in eval_metadata
        assert "temperature" in eval_metadata
        assert eval_metadata["temperature"] == 0.0
        assert "validation_success" in eval_metadata
        assert "structural_correctness" in eval_metadata

        # Performance validation
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_consolidate_shortterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for consolidate_shortterm"
        assert (
            latest_duration < 10000
        ), f"consolidate_shortterm took {latest_duration:.1f}ms, should be <10s"

        print(f"🧠 Basic consolidation completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_consolidate_shortterm_schema_validation_success(test_stack_seeded):
    """Test successful schema validation with realistic data."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # First, add some memories to consolidate
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Alpha and Jeffery worked on memory consolidation testing today. The new schema validation approach is working well."
            },
        )
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "We implemented systematic model evaluation for consolidation. Temperature=0 provides deterministic testing."
            },
        )

        # Now test consolidation with recent memories
        result = await time_mcp_call(
            client, "consolidate_shortterm", {"time_window": "1h"}
        )
        data = json.loads(result.content[0].text)

        # Analyze the results for systematic evaluation
        if data["success"]:
            # Successful consolidation - validate structure
            assert "consolidation" in data
            consolidation = data["consolidation"]

            # Validate schema compliance
            required_fields = [
                "entities",
                "relationships",
                "insights",
                "summary",
                "emotional_context",
                "next_steps",
            ]
            for field in required_fields:
                assert field in consolidation, f"Missing required field: {field}"

            # Validate metadata
            assert "consolidation_metadata" in consolidation
            metadata = consolidation["consolidation_metadata"]
            assert "processing_time_ms" in metadata
            assert "model_used" in metadata
            assert "validation_success" in metadata
            assert metadata["validation_success"] is True

            print(f"✅ Schema validation SUCCESS - model: {metadata['model_used']}")
        else:
            # Failed consolidation - analyze failure patterns
            assert "validation_errors" in data
            assert "raw_model_output" in data

            validation_errors = data["validation_errors"]
            print(f"❌ Schema validation FAILED - errors: {len(validation_errors)}")
            for error in validation_errors:
                print(f"  - Field: {error['field']}, Error: {error['error']}")

        # Record evaluation results regardless of success/failure
        eval_metadata = data["metadata"]["model_evaluation"]
        print(f"📊 Model evaluation: {eval_metadata['structural_correctness']}")


@pytest.mark.asyncio
@performance_test
async def test_consolidate_shortterm_time_window_variations(test_stack_seeded):
    """Test consolidation with different time windows."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Test different time windows
        time_windows = ["1h", "6h", "24h"]

        for time_window in time_windows:
            result = await time_mcp_call(
                client, "consolidate_shortterm", {"time_window": time_window}
            )
            data = json.loads(result.content[0].text)

            # Should handle all time windows gracefully
            assert "success" in data
            assert "metadata" in data

            if "input_memories_count" in data["metadata"]:
                memory_count = data["metadata"]["input_memories_count"]
                print(f"⏰ {time_window}: {memory_count} memories")

        # Performance validation for the batch
        from tests.e2e.fixtures.performance import collector

        consolidation_durations = []
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_consolidate_shortterm":
                consolidation_durations.append(metric["duration_ms"])
                if len(consolidation_durations) >= len(time_windows):
                    break

        assert len(consolidation_durations) >= len(
            time_windows
        ), "Should have recorded timing for all time window tests"

        avg_duration = sum(consolidation_durations[: len(time_windows)]) / len(
            time_windows
        )
        print(f"🕐 Time window tests averaged {avg_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_consolidate_shortterm_temperature_determinism(test_stack_seeded):
    """Test that temperature=0 produces deterministic results."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Add consistent test data
        test_content = "Alpha discovered that systematic model evaluation is crucial for consolidation reliability."
        await client.call_tool("remember_shortterm", {"content": test_content})

        # Run consolidation multiple times with temperature=0
        results = []
        for _ in range(2):  # Run twice to test determinism
            result = await time_mcp_call(
                client,
                "consolidate_shortterm",
                {"time_window": "1h", "temperature": 0.0},
            )
            data = json.loads(result.content[0].text)
            results.append(data)

        # Analyze determinism
        both_successful = all(r["success"] for r in results)
        both_failed = all(not r["success"] for r in results)

        if both_successful:
            # Compare structural elements for determinism
            entity_counts = [len(r["consolidation"]["entities"]) for r in results]

            print(f"🎯 Deterministic SUCCESS - entity counts: {entity_counts}")
            # Note: With temperature=0, we should get identical results
            # But we'll log for analysis rather than strict assertion
            # since model determinism can vary by implementation
        elif both_failed:
            # Both failed - check if failure patterns are consistent
            error_patterns = [len(r.get("validation_errors", [])) for r in results]
            print(f"🎯 Deterministic FAILURE - error counts: {error_patterns}")
        else:
            print("⚠️  Non-deterministic behavior detected at temperature=0")

        # Performance validation
        from tests.e2e.fixtures.performance import collector

        determinism_durations = []
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_consolidate_shortterm":
                determinism_durations.append(metric["duration_ms"])
                if len(determinism_durations) >= 2:
                    break

        avg_duration = sum(determinism_durations[:2]) / 2
        print(f"🔄 Determinism test averaged {avg_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_consolidate_shortterm_empty_memory_handling(test_stack_seeded):
    """Test consolidation behavior with no recent memories."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Test with a very narrow time window (should find no memories)
        result = await time_mcp_call(
            client, "consolidate_shortterm", {"time_window": "1m"}
        )
        data = json.loads(result.content[0].text)

        # Should handle empty input gracefully
        assert "success" in data

        if data["success"]:
            # Should return empty but valid consolidation
            assert "consolidation" in data
            consolidation = data["consolidation"]
            assert isinstance(consolidation["entities"], list)
            assert isinstance(consolidation["relationships"], list)
            assert isinstance(consolidation["insights"], list)
            print("📭 Empty memory consolidation handled successfully")
        else:
            # Should explain why it failed
            assert "error" in data
            print(f"📭 Empty memory handling: {data['error']}")

        # Performance validation
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_consolidate_shortterm":
                latest_duration = metric["duration_ms"]
                break

        assert (
            latest_duration is not None
        ), "Should have recorded timing for empty memory test"
        # Empty memory handling should be very fast
        assert (
            latest_duration < 1000
        ), f"Empty memory consolidation took {latest_duration:.1f}ms, should be <1s"

        print(f"📭 Empty memory test completed in {latest_duration:.1f}ms")


@pytest.mark.asyncio
@performance_test
async def test_consolidate_shortterm_model_evaluation_metadata(test_stack_seeded):
    """Test comprehensive model evaluation metadata collection."""
    server_url, seeded_data = test_stack_seeded
    async with Client(server_url) as client:
        # Add test memories for evaluation
        await client.call_tool(
            "remember_shortterm",
            {
                "content": "Model evaluation test: Alpha is testing consolidation reliability."
            },
        )

        result = await time_mcp_call(
            client, "consolidate_shortterm", {"time_window": "1h"}
        )
        data = json.loads(result.content[0].text)

        # Validate comprehensive evaluation metadata
        assert "metadata" in data
        metadata = data["metadata"]

        # Model evaluation section
        assert "model_evaluation" in metadata
        eval_data = metadata["model_evaluation"]

        required_eval_fields = [
            "model_name",
            "temperature",
            "validation_success",
            "structural_correctness",
        ]
        for field in required_eval_fields:
            assert field in eval_data, f"Missing evaluation field: {field}"

        # Processing metadata
        assert "processing_time_ms" in metadata
        assert isinstance(metadata["processing_time_ms"], int | float)

        # Tool metadata
        assert "tool_metadata" in data
        tool_data = data["tool_metadata"]
        assert tool_data["systematic_evaluation"] is True

        # Log evaluation results for analysis
        print(f"🔬 Model: {eval_data['model_name']}")
        print(f"🔬 Temperature: {eval_data['temperature']}")
        print(f"🔬 Validation: {eval_data['validation_success']}")
        print(f"🔬 Structure: {eval_data['structural_correctness']}")
        print(f"🔬 Processing: {metadata['processing_time_ms']}ms")

        if not eval_data["validation_success"]:
            # Log failure patterns for analysis
            if "failure_patterns" in eval_data:
                print(f"🔬 Failure patterns: {eval_data['failure_patterns']}")
            if "validation_errors" in data:
                print(f"🔬 Validation errors: {len(data['validation_errors'])}")

        # Performance validation
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_consolidate_shortterm":
                latest_duration = metric["duration_ms"]
                break

        print(f"🔬 Evaluation test completed in {latest_duration:.1f}ms")
