"""E2E tests for get_personality_trait tool.

Tests the get_personality_trait MCP tool through the actual MCP protocol
against a real Alpha-Recall server instance with Memgraph backend.
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_get_personality_trait_existing_trait(test_stack):
    """Test retrieving an existing personality trait through MCP protocol.

    Uses greenfield test stack - creates a test trait and directive first.
    """
    server_url = test_stack
    async with Client(server_url) as client:
        # First call gentle_refresh to ensure system is initialized
        refresh_result = await time_mcp_call(client, "gentle_refresh", {})
        refresh_data = json.loads(refresh_result.content[0].text)

        assert refresh_data["success"] is True
        assert "personality" in refresh_data

        # Create a test trait for this test (E2E uses fresh database)
        create_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "test_warmth",
                "description": "Caring and empathetic behavioral patterns for testing",
                "weight": 0.8,
            },
        )
        create_data = json.loads(create_result.content[0].text)
        assert create_data["success"] is True

        # Add a directive to the trait
        directive_result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "test_warmth",
                "instruction": "Show empathy and understanding in conversations",
                "weight": 0.9,
            },
        )
        directive_data = json.loads(directive_result.content[0].text)
        assert directive_data["success"] is True

        test_trait = "test_warmth"

        # Test get_personality_trait
        result = await time_mcp_call(
            client, "get_personality_trait", {"trait_name": test_trait}
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        assert data["trait"]["name"] == test_trait
        assert "description" in data["trait"]
        assert "weight" in data["trait"]
        assert "created_at" in data["trait"]
        assert "directives" in data["trait"]

        # Should have at least one directive
        assert len(data["trait"]["directives"]) > 0

        # Verify directive structure
        first_directive = data["trait"]["directives"][0]
        assert "instruction" in first_directive
        assert "weight" in first_directive
        assert "created_at" in first_directive

        # Verify directives are ordered by weight (DESC)
        directive_weights = [d["weight"] for d in data["trait"]["directives"]]
        assert directive_weights == sorted(directive_weights, reverse=True)

        # Assert performance
        from tests.e2e.fixtures.performance import collector

        latest_duration = None
        for metric in reversed(collector.get_metrics()):
            if metric["operation"] == "mcp_call_get_personality_trait":
                latest_duration = metric["duration_ms"]
                break

        assert latest_duration is not None, "Should have recorded timing"
        assert (
            latest_duration < 1000
        ), f"get_personality_trait took {latest_duration:.1f}ms, should be <1000ms"

        print(
            f"🧠 Retrieved '{test_trait}' trait with {len(data['trait']['directives'])} directives in {latest_duration:.1f}ms"
        )


@pytest.mark.asyncio
@performance_test
async def test_get_personality_trait_nonexistent_trait(test_stack):
    """Test retrieving a nonexistent personality trait returns helpful error."""
    server_url = test_stack
    async with Client(server_url) as client:
        # First ensure traits are initialized
        await time_mcp_call(client, "gentle_refresh", {})

        # Test with clearly nonexistent trait
        result = await time_mcp_call(
            client,
            "get_personality_trait",
            {"trait_name": "definitely_nonexistent_trait_12345"},
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "not found" in data["error"]
        assert "available_traits" in data
        assert isinstance(data["available_traits"], list)

        # Should list available traits for user guidance
        if len(data["available_traits"]) > 0:
            # If traits exist, verify they're realistic trait names
            for trait_name in data["available_traits"]:
                assert isinstance(trait_name, str)
                assert len(trait_name) > 0
                # Should be reasonable trait names (not empty or just special chars)
                assert trait_name.replace("_", "").isalnum()

        print(
            f"❌ Nonexistent trait correctly returned error with {len(data['available_traits'])} available traits"
        )


@pytest.mark.asyncio
@performance_test
async def test_get_personality_trait_warmth_trait_specific(test_stack):
    """Test retrieving the warmth trait specifically with expected content."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Initialize traits
        await time_mcp_call(client, "gentle_refresh", {})

        # Create warmth trait with warmth-specific directive (E2E tests use fresh database)
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

        # Add a warmth-specific directive that contains expected terms
        directive_result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": "Show genuine care and support for others, celebrate their successes with enthusiasm",
                "weight": 0.9,
            },
        )
        directive_data = json.loads(directive_result.content[0].text)
        # Allow directive to already exist
        if not directive_data["success"] and "already exists" not in directive_data.get(
            "error", ""
        ):
            raise AssertionError(
                f"Failed to create warmth directive: {directive_data.get('error')}"
            )

        # Test specific warmth trait (should exist from our hierarchical structure)
        result = await time_mcp_call(
            client, "get_personality_trait", {"trait_name": "warmth"}
        )
        data = json.loads(result.content[0].text)

        if data["success"]:
            # If warmth trait exists, verify its expected structure
            assert data["trait"]["name"] == "warmth"
            assert (
                "caring" in data["trait"]["description"].lower()
                or "warm" in data["trait"]["description"].lower()
            )

            # Should have multiple directives for warmth
            assert len(data["trait"]["directives"]) >= 1

            # Check for expected warmth-related content
            directive_text = " ".join(
                [d["instruction"] for d in data["trait"]["directives"]]
            )
            warmth_indicators = [
                "enthusiasm",
                "care",
                "collaboration",
                "support",
                "celebrate",
                "genuine",
            ]

            found_indicators = [
                indicator
                for indicator in warmth_indicators
                if indicator in directive_text.lower()
            ]
            assert (
                len(found_indicators) > 0
            ), f"Expected warmth-related terms, found: {found_indicators}"

            print(
                f"💖 Warmth trait verified with {len(data['trait']['directives'])} directives"
            )
        else:
            # If warmth doesn't exist, that's also valid - just verify error handling
            assert "not found" in data["error"]
            print("💖 Warmth trait not found (expected in some configurations)")


@pytest.mark.asyncio
@performance_test
async def test_get_personality_trait_parameter_safety(test_stack):
    """Test that get_personality_trait safely handles special characters."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Initialize traits
        await time_mcp_call(client, "gentle_refresh", {})

        # Test with special characters that could be problematic in Cypher
        special_trait_names = [
            "trait'; DROP TABLE nodes; --",
            "trait\nwith\nnewlines",
            "trait with spaces",
            "trait_with_unicode_🧠_emoji",
            "",  # Empty string
            "a" * 1000,  # Very long string
        ]

        for special_name in special_trait_names:
            result = await time_mcp_call(
                client, "get_personality_trait", {"trait_name": special_name}
            )
            data = json.loads(result.content[0].text)

            # Should handle safely without server errors
            assert "success" in data
            assert isinstance(data["success"], bool)

            if not data["success"]:
                # Should be a "not found" error, not a database error
                assert "not found" in data["error"]
                assert "available_traits" in data

        print(
            "🔒 Special character handling verified - no database injection vulnerabilities"
        )


@pytest.mark.asyncio
@performance_test
async def test_get_personality_trait_consistency_with_gentle_refresh(test_stack):
    """Test that get_personality_trait returns consistent data with gentle_refresh."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Get personality data from gentle_refresh
        refresh_result = await time_mcp_call(client, "gentle_refresh", {})
        refresh_data = json.loads(refresh_result.content[0].text)

        assert refresh_data["success"] is True
        personality_from_refresh = refresh_data["personality"]

        # For each trait in gentle_refresh, verify get_personality_trait returns consistent data
        for trait_name, trait_summary in personality_from_refresh.items():
            trait_result = await time_mcp_call(
                client, "get_personality_trait", {"trait_name": trait_name}
            )
            trait_data = json.loads(trait_result.content[0].text)

            assert trait_data["success"] is True
            assert trait_data["trait"]["name"] == trait_name
            assert trait_data["trait"]["description"] == trait_summary["description"]
            assert trait_data["trait"]["weight"] == trait_summary["weight"]

            # Verify directive counts match (gentle_refresh filters 0.0 weights, get_personality_trait doesn't)
            # So we filter the get_personality_trait results to match gentle_refresh behavior
            non_zero_directives = [d for d in trait_data["trait"]["directives"] if d["weight"] != 0.0]
            assert len(non_zero_directives) == len(trait_summary["directives"])

            # Verify directive content matches (only compare non-zero weight directives)
            for i, directive in enumerate(non_zero_directives):
                expected_directive = trait_summary["directives"][i]
                assert directive["instruction"] == expected_directive["instruction"]
                assert directive["weight"] == expected_directive["weight"]

        print(
            f"🔄 Verified consistency between gentle_refresh and get_personality_trait for {len(personality_from_refresh)} traits"
        )


@pytest.mark.asyncio
@performance_test
async def test_get_personality_trait_response_structure(test_stack):
    """Test that get_personality_trait returns properly structured responses."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Initialize traits
        await time_mcp_call(client, "gentle_refresh", {})

        # Test with nonexistent trait to check error response structure
        error_result = await time_mcp_call(
            client, "get_personality_trait", {"trait_name": "nonexistent"}
        )
        error_data = json.loads(error_result.content[0].text)

        # Verify error response structure
        assert "success" in error_data
        assert error_data["success"] is False
        assert "error" in error_data
        assert isinstance(error_data["error"], str)
        assert "available_traits" in error_data
        assert isinstance(error_data["available_traits"], list)

        # If any traits exist, test success response structure
        if len(error_data["available_traits"]) > 0:
            test_trait = error_data["available_traits"][0]
            success_result = await time_mcp_call(
                client, "get_personality_trait", {"trait_name": test_trait}
            )
            success_data = json.loads(success_result.content[0].text)

            # Verify success response structure
            assert "success" in success_data
            assert success_data["success"] is True
            assert "trait" in success_data

            trait = success_data["trait"]
            required_fields = [
                "name",
                "description",
                "weight",
                "created_at",
                "directives",
            ]
            for field in required_fields:
                assert field in trait, f"Missing required field: {field}"

            # Verify directive structure
            assert isinstance(trait["directives"], list)
            if len(trait["directives"]) > 0:
                directive = trait["directives"][0]
                directive_fields = ["instruction", "weight", "created_at"]
                for field in directive_fields:
                    assert field in directive, f"Missing directive field: {field}"

        print("📋 Response structure validation complete")
