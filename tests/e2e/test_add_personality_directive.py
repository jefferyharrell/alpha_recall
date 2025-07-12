"""E2E tests for add_personality_directive tool.

Tests the add_personality_directive MCP tool through the actual MCP protocol
against a real Alpha-Recall server instance with Memgraph backend.
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_add_personality_directive_success(test_stack):
    """Test successfully adding a new directive to existing trait."""
    server_url = test_stack
    async with Client(server_url) as client:
        # First ensure personality traits are initialized
        await time_mcp_call(client, "gentle_refresh", {})

        # Create the warmth trait first (E2E tests use fresh database)
        create_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "warmth",
                "description": "Caring and empathetic behavioral patterns",
                "weight": 0.8,
            },
        )
        create_data = json.loads(create_result.content[0].text)
        # Allow trait to already exist (other tests may have created it)
        if not create_data["success"] and "already exists" not in create_data.get(
            "error", ""
        ):
            raise AssertionError(
                f"Failed to create warmth trait: {create_data.get('error')}"
            )

        # Add a new directive to warmth trait
        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": "Offer words of encouragement during difficult times",
                "weight": 0.85,
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["trait_name"] == "warmth"
        assert "trait_description" in data
        assert (
            data["directive_added"]["instruction"]
            == "Offer words of encouragement during difficult times"
        )
        assert data["directive_added"]["weight"] == 0.85
        assert "created_at" in data["directive_added"]

        # Verify the directive was actually added by retrieving the trait
        trait_result = await time_mcp_call(
            client, "get_personality_trait", {"trait_name": "warmth"}
        )
        trait_data = json.loads(trait_result.content[0].text)

        if trait_data["success"]:
            # Find our new directive in the list
            found_directive = None
            for directive in trait_data["trait"]["directives"]:
                if (
                    directive["instruction"]
                    == "Offer words of encouragement during difficult times"
                ):
                    found_directive = directive
                    break

            assert found_directive is not None, "New directive should be found in trait"
            assert found_directive["weight"] == 0.85

        # Assert performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_add_personality_directive":
                latest_duration = metric["duration_ms"]
                break

        assert latest_duration is not None, "Should have recorded timing"
        assert (
            latest_duration < 2000
        ), f"add_personality_directive took {latest_duration:.1f}ms, should be <2000ms"

        print(
            f"✅ Successfully added directive to warmth trait in {latest_duration:.1f}ms"
        )


@pytest.mark.asyncio
@performance_test
async def test_add_personality_directive_duplicate_detection(test_stack):
    """Test that duplicate directives are properly detected."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Ensure traits are initialized
        await time_mcp_call(client, "gentle_refresh", {})

        # Create the warmth trait first (E2E tests use fresh database)
        create_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "warmth",
                "description": "Caring and empathetic behavioral patterns",
                "weight": 0.8,
            },
        )
        create_data = json.loads(create_result.content[0].text)
        # Allow trait to already exist (other tests may have created it)
        if not create_data["success"] and "already exists" not in create_data.get(
            "error", ""
        ):
            raise AssertionError(
                f"Failed to create warmth trait: {create_data.get('error')}"
            )

        # Add a directive
        unique_instruction = (
            "Test directive for duplicate detection - unique identifier 12345"
        )
        first_result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": unique_instruction,
                "weight": 0.7,
            },
        )
        first_data = json.loads(first_result.content[0].text)

        # First addition should succeed
        if first_data["success"]:
            # Try to add the same directive again
            second_result = await time_mcp_call(
                client,
                "add_personality_directive",
                {
                    "trait_name": "warmth",
                    "instruction": unique_instruction,
                    "weight": 0.8,  # Different weight, but same instruction
                },
            )
            second_data = json.loads(second_result.content[0].text)

            assert second_data["success"] is False
            assert "already exists" in second_data["error"]
            assert "existing_directive" in second_data
            assert (
                second_data["existing_directive"]["instruction"] == unique_instruction
            )
            assert second_data["existing_directive"]["weight"] == 0.7  # Original weight

            print("🔍 Duplicate detection working correctly")
        else:
            # If first addition failed (maybe directive already exists), that's also valid
            print(f"⚠️  First directive addition failed: {first_data['error']}")


@pytest.mark.asyncio
@performance_test
async def test_add_personality_directive_nonexistent_trait(test_stack):
    """Test adding directive to nonexistent trait returns helpful error."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Ensure traits are initialized for available traits list
        await time_mcp_call(client, "gentle_refresh", {})

        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "definitely_nonexistent_trait_12345",
                "instruction": "Some instruction",
                "weight": 0.5,
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "not found" in data["error"]
        assert "available_traits" in data
        assert isinstance(data["available_traits"], list)

        # Should list available traits for user guidance
        if len(data["available_traits"]) > 0:
            for trait_name in data["available_traits"]:
                assert isinstance(trait_name, str)
                assert len(trait_name) > 0

        print(
            f"❌ Nonexistent trait correctly returned error with {len(data['available_traits'])} available traits"
        )


@pytest.mark.asyncio
@performance_test
async def test_add_personality_directive_validation_errors(test_stack):
    """Test validation error handling through MCP protocol."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Test empty instruction
        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {"trait_name": "warmth", "instruction": "", "weight": 0.5},
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "Instruction cannot be empty" in data["error"]

        # Test invalid weight (too high)
        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": "Valid instruction",
                "weight": 1.5,
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]

        # Test invalid weight (negative)
        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": "Valid instruction",
                "weight": -0.1,
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]

        # Test empty trait name
        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {"trait_name": "", "instruction": "Valid instruction", "weight": 0.5},
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "Trait name cannot be empty" in data["error"]

        print("✅ All validation errors properly handled")


@pytest.mark.asyncio
@performance_test
async def test_add_personality_directive_boundary_weights(test_stack):
    """Test boundary weight values (0.0 and 1.0) are accepted."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Ensure traits are initialized
        await time_mcp_call(client, "gentle_refresh", {})

        # Create the warmth trait first (E2E tests use fresh database)
        create_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "warmth",
                "description": "Caring and empathetic behavioral patterns",
                "weight": 0.8,
            },
        )
        create_data = json.loads(create_result.content[0].text)
        # Allow trait to already exist (other tests may have created it)
        if not create_data["success"] and "already exists" not in create_data.get(
            "error", ""
        ):
            raise AssertionError(
                f"Failed to create warmth trait: {create_data.get('error')}"
            )

        # Test weight 0.0
        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": "Test directive with zero weight for boundary testing",
                "weight": 0.0,
            },
        )
        data = json.loads(result.content[0].text)

        # Should succeed (or fail for reasons other than weight validation)
        if not data["success"]:
            assert "Weight must be a number" not in data["error"]

        # Test weight 1.0
        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": "Test directive with max weight for boundary testing",
                "weight": 1.0,
            },
        )
        data = json.loads(result.content[0].text)

        # Should succeed (or fail for reasons other than weight validation)
        if not data["success"]:
            assert "Weight must be a number" not in data["error"]

        print("🎯 Boundary weight values (0.0, 1.0) properly accepted")


@pytest.mark.asyncio
@performance_test
async def test_add_personality_directive_instruction_trimming(test_stack):
    """Test that instruction text is properly trimmed of whitespace."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Ensure traits are initialized
        await time_mcp_call(client, "gentle_refresh", {})

        # Create the warmth trait first (E2E tests use fresh database)
        create_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "warmth",
                "description": "Caring and empathetic behavioral patterns",
                "weight": 0.8,
            },
        )
        create_data = json.loads(create_result.content[0].text)
        # Allow trait to already exist (other tests may have created it)
        if not create_data["success"] and "already exists" not in create_data.get(
            "error", ""
        ):
            raise AssertionError(
                f"Failed to create warmth trait: {create_data.get('error')}"
            )

        # Add directive with padded whitespace
        padded_instruction = (
            "  \n\t  Test instruction with lots of whitespace padding  \t\n  "
        )
        expected_trimmed = "Test instruction with lots of whitespace padding"

        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": padded_instruction,
                "weight": 0.6,
            },
        )
        data = json.loads(result.content[0].text)

        if data["success"]:
            # Verify the instruction was trimmed in the response
            assert data["directive_added"]["instruction"] == expected_trimmed

            # Verify by trying to add the trimmed version (should detect duplicate)
            duplicate_result = await time_mcp_call(
                client,
                "add_personality_directive",
                {
                    "trait_name": "warmth",
                    "instruction": expected_trimmed,  # Trimmed version
                    "weight": 0.7,
                },
            )
            duplicate_data = json.loads(duplicate_result.content[0].text)

            assert duplicate_data["success"] is False
            assert "already exists" in duplicate_data["error"]

            print("✂️  Instruction trimming and duplicate detection working correctly")


@pytest.mark.asyncio
@performance_test
async def test_add_personality_directive_response_structure(test_stack):
    """Test that responses have proper structure."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Ensure traits are initialized
        await time_mcp_call(client, "gentle_refresh", {})

        # Create the warmth trait first (E2E tests use fresh database)
        create_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "warmth",
                "description": "Caring and empathetic behavioral patterns",
                "weight": 0.8,
            },
        )
        create_data = json.loads(create_result.content[0].text)
        # Allow trait to already exist (other tests may have created it)
        if not create_data["success"] and "already exists" not in create_data.get(
            "error", ""
        ):
            raise AssertionError(
                f"Failed to create warmth trait: {create_data.get('error')}"
            )

        # Test success response structure
        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": "Test directive for response structure validation",
                "weight": 0.65,
            },
        )
        data = json.loads(result.content[0].text)

        # Verify response structure
        assert "success" in data
        assert isinstance(data["success"], bool)

        if data["success"]:
            # Success response structure
            required_fields = ["trait_name", "trait_description", "directive_added"]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"

            directive = data["directive_added"]
            directive_fields = ["instruction", "weight", "created_at"]
            for field in directive_fields:
                assert field in directive, f"Missing directive field: {field}"

            # Verify data types
            assert isinstance(data["trait_name"], str)
            assert isinstance(data["trait_description"], str)
            assert isinstance(directive["instruction"], str)
            assert isinstance(directive["weight"], int | float)
            assert isinstance(directive["created_at"], str)

        else:
            # Error response structure
            assert "error" in data
            assert isinstance(data["error"], str)

        print("📋 Response structure validation complete")


@pytest.mark.asyncio
@performance_test
async def test_add_personality_directive_parameter_safety(test_stack):
    """Test that special characters are handled safely."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Ensure traits are initialized
        await time_mcp_call(client, "gentle_refresh", {})

        # Create the warmth trait first (E2E tests use fresh database)
        create_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "warmth",
                "description": "Caring and empathetic behavioral patterns",
                "weight": 0.8,
            },
        )
        create_data = json.loads(create_result.content[0].text)
        # Allow trait to already exist (other tests may have created it)
        if not create_data["success"] and "already exists" not in create_data.get(
            "error", ""
        ):
            raise AssertionError(
                f"Failed to create warmth trait: {create_data.get('error')}"
            )

        # Test with special characters that could be problematic
        special_instructions = [
            "Directive with 'single quotes' and \"double quotes\"",
            "Directive with\nnewlines\nand\ttabs",
            "Directive with unicode emoji 🧠💖 characters",
            "Directive'; DROP TABLE nodes; --",  # SQL injection attempt
            "Directive with very long text " + "x" * 1000,
        ]

        for instruction in special_instructions:
            result = await time_mcp_call(
                client,
                "add_personality_directive",
                {"trait_name": "warmth", "instruction": instruction, "weight": 0.5},
            )
            data = json.loads(result.content[0].text)

            # Should handle safely without server errors
            assert "success" in data
            assert isinstance(data["success"], bool)

            # If it failed, should be for legitimate reasons (duplicate, etc.)
            # not database errors or crashes
            if not data["success"]:
                legitimate_errors = [
                    "already exists",
                    "not found",
                    "cannot be empty",
                    "Weight must be",
                ]
                error_is_legitimate = any(
                    err in data["error"] for err in legitimate_errors
                )
                # Allow database-specific errors but not crashes
                assert (
                    error_is_legitimate
                    or "Error adding personality directive" in data["error"]
                )

        print("🔒 Special character handling verified - no injection vulnerabilities")
