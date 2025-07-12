"""E2E tests for get_personality tool."""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_get_personality_complete_overview(test_stack):
    """Test get_personality returns complete personality structure."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Initialize personality system
        await time_mcp_call(client, "gentle_refresh", {})

        # Get complete personality overview
        result = await time_mcp_call(client, "get_personality", {})
        data = json.loads(result.content[0].text)

        if not data["success"]:
            print(f"❌ get_personality failed: {data}")

        assert data["success"] is True
        assert isinstance(data["personality"], dict)
        assert isinstance(data["trait_count"], int)
        assert isinstance(data["directive_count"], int)
        assert "retrieved_at" in data
        assert "correlation_id" in data

        # Check the personality structure
        personality = data["personality"]

        if data["trait_count"] == 0:
            # Empty database scenario - this is expected in a fresh environment
            assert personality == {}
            assert data["directive_count"] == 0
            print("✅ Empty personality database handled correctly")
        else:
            # Populated database scenario - validate structure
            for _trait_name, trait in personality.items():
                assert "description" in trait
                assert "weight" in trait
                assert "created_at" in trait
                assert "last_updated" in trait
                assert "directives" in trait
                assert isinstance(trait["directives"], list)

                # Each directive should have proper structure
                for directive in trait["directives"]:
                    assert "instruction" in directive
                    assert "weight" in directive
                    assert "created_at" in directive
                    assert isinstance(directive["weight"], int | float)

            print(
                f"✅ Populated personality retrieved with {data['trait_count']} traits and {data['directive_count']} directives"
            )


@pytest.mark.asyncio
@performance_test
async def test_get_personality_trait_discovery(test_stack):
    """Test get_personality enables trait discovery."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Initialize personality system
        await time_mcp_call(client, "gentle_refresh", {})

        # Create some test traits for discovery testing
        trait1_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "test_trait_1",
                "description": "First test trait for discovery",
                "weight": 1.0,
            },
        )
        assert json.loads(trait1_result.content[0].text)["success"] is True

        trait2_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "test_trait_2",
                "description": "Second test trait for discovery",
                "weight": 0.8,
            },
        )
        assert json.loads(trait2_result.content[0].text)["success"] is True

        # Add directives to make traits discoverable
        await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "test_trait_1",
                "instruction": "Test directive 1",
                "weight": 1.0,
            },
        )
        await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "test_trait_2",
                "instruction": "Test directive 2",
                "weight": 0.9,
            },
        )

        # Get personality overview for discovery
        result = await time_mcp_call(client, "get_personality", {})
        data = json.loads(result.content[0].text)

        assert data["success"] is True
        personality = data["personality"]

        # Verify we can discover available traits
        trait_names = list(personality.keys())
        assert len(trait_names) > 0, "Should have discoverable traits"

        # Test that the specific traits we created can be accessed via get_personality_trait
        test_trait_names = ["test_trait_1", "test_trait_2"]
        for trait_name in test_trait_names:
            if (
                trait_name in trait_names
            ):  # Only test traits we created that were discovered
                trait_result = await time_mcp_call(
                    client, "get_personality_trait", {"trait_name": trait_name}
                )
                trait_data = json.loads(trait_result.content[0].text)

                assert trait_data["success"] is True
                assert trait_data["trait"]["name"] == trait_name

                # Data should be consistent between get_personality and get_personality_trait
                overview_trait = personality[trait_name]
                specific_trait = trait_data["trait"]

                assert overview_trait["description"] == specific_trait["description"]
                assert overview_trait["weight"] == specific_trait["weight"]
                assert len(overview_trait["directives"]) == len(
                    specific_trait["directives"]
                )

        print(
            f"🔍 Trait discovery working - found {len(trait_names)} discoverable traits"
        )


@pytest.mark.asyncio
@performance_test
async def test_get_personality_empty_vs_populated(test_stack):
    """Test get_personality behavior with empty vs populated personality."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Test against populated database (seeded data should exist)
        result = await time_mcp_call(client, "get_personality", {})
        data = json.loads(result.content[0].text)

        assert data["success"] is True

        if data["trait_count"] == 0:
            # Empty database scenario
            assert data["personality"] == {}
            assert data["directive_count"] == 0
            print("📋 Empty personality database handled correctly")
        else:
            # Populated database scenario
            assert len(data["personality"]) == data["trait_count"]
            assert data["directive_count"] >= 0

            total_directives = sum(
                len(trait["directives"]) for trait in data["personality"].values()
            )
            assert total_directives == data["directive_count"]

            print(
                f"📊 Populated personality: {data['trait_count']} traits, {data['directive_count']} directives"
            )


@pytest.mark.asyncio
@performance_test
async def test_get_personality_response_structure_validation(test_stack):
    """Test get_personality returns properly structured response."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Initialize and get personality
        await time_mcp_call(client, "gentle_refresh", {})
        result = await time_mcp_call(client, "get_personality", {})
        data = json.loads(result.content[0].text)

        # Validate top-level structure
        required_fields = [
            "success",
            "personality",
            "trait_count",
            "directive_count",
            "retrieved_at",
            "correlation_id",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Validate data types
        assert isinstance(data["success"], bool)
        assert isinstance(data["personality"], dict)
        assert isinstance(data["trait_count"], int)
        assert isinstance(data["directive_count"], int)
        assert isinstance(data["retrieved_at"], str)
        assert isinstance(data["correlation_id"], str)

        # Validate personality structure
        for trait_name, trait in data["personality"].items():
            assert isinstance(trait_name, str)
            assert isinstance(trait, dict)

            trait_fields = [
                "description",
                "weight",
                "created_at",
                "last_updated",
                "directives",
            ]
            for field in trait_fields:
                assert field in trait, f"Missing trait field: {field}"

            assert isinstance(trait["description"], str)
            assert isinstance(trait["weight"], int | float)
            assert isinstance(trait["created_at"], str)
            assert isinstance(trait["last_updated"], str)
            assert isinstance(trait["directives"], list)

            # Validate directive structure
            for directive in trait["directives"]:
                directive_fields = ["instruction", "weight", "created_at"]
                for field in directive_fields:
                    assert field in directive, f"Missing directive field: {field}"

                assert isinstance(directive["instruction"], str)
                assert isinstance(directive["weight"], int | float)
                assert isinstance(directive["created_at"], str)

        print("📋 Response structure validation complete")


@pytest.mark.asyncio
@performance_test
async def test_get_personality_datetime_formatting(test_stack):
    """Test get_personality uses proper datetime formatting."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Initialize and get personality
        await time_mcp_call(client, "gentle_refresh", {})
        result = await time_mcp_call(client, "get_personality", {})
        data = json.loads(result.content[0].text)

        assert data["success"] is True

        # Check retrieved_at format
        retrieved_at = data["retrieved_at"]
        assert (
            "+00:00" in retrieved_at or "Z" in retrieved_at
        ), f"Invalid datetime format: {retrieved_at}"

        # Check trait datetime formats
        for _trait_name, trait in data["personality"].items():
            created_at = trait["created_at"]
            last_updated = trait["last_updated"]

            assert (
                "+00:00" in created_at or "Z" in created_at
            ), f"Invalid trait created_at format: {created_at}"
            assert (
                "+00:00" in last_updated or "Z" in last_updated
            ), f"Invalid trait last_updated format: {last_updated}"

            # Check directive datetime formats
            for directive in trait["directives"]:
                directive_created_at = directive["created_at"]
                assert (
                    "+00:00" in directive_created_at or "Z" in directive_created_at
                ), f"Invalid directive created_at format: {directive_created_at}"

        print("🕒 Datetime formatting validation complete")


@pytest.mark.asyncio
@performance_test
async def test_get_personality_directive_ordering(test_stack):
    """Test get_personality returns directives in proper order."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Initialize personality system
        await time_mcp_call(client, "gentle_refresh", {})

        # Get personality to check directive ordering
        result = await time_mcp_call(client, "get_personality", {})
        data = json.loads(result.content[0].text)

        assert data["success"] is True

        # Check directive ordering within each trait
        for trait_name, trait in data["personality"].items():
            directives = trait["directives"]

            if len(directives) > 1:
                # Directives should be ordered by weight DESC, then created_at ASC
                for i in range(len(directives) - 1):
                    current_weight = directives[i]["weight"]
                    next_weight = directives[i + 1]["weight"]

                    # Higher weights should come first
                    assert (
                        current_weight >= next_weight
                    ), f"Directive ordering error in {trait_name}: {current_weight} should be >= {next_weight}"

                print(
                    f"✅ {trait_name}: {len(directives)} directives properly ordered by weight"
                )

        print("📊 Directive ordering validation complete")


@pytest.mark.asyncio
@performance_test
async def test_get_personality_consistency_with_gentle_refresh(test_stack):
    """Test get_personality consistency with gentle_refresh personality data."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Get data from both tools
        refresh_result = await time_mcp_call(client, "gentle_refresh", {})
        refresh_data = json.loads(refresh_result.content[0].text)

        personality_result = await time_mcp_call(client, "get_personality", {})
        personality_data = json.loads(personality_result.content[0].text)

        assert refresh_data["success"] is True
        assert personality_data["success"] is True

        # Both should have personality data
        refresh_personality = refresh_data["personality"]
        get_personality = personality_data["personality"]

        # Should have the same traits
        refresh_traits = set(refresh_personality.keys())
        get_traits = set(get_personality.keys())
        assert refresh_traits == get_traits, "Trait sets should match between tools"

        # Check trait consistency
        for trait_name in refresh_traits:
            refresh_trait = refresh_personality[trait_name]
            get_trait = get_personality[trait_name]

            # Core trait data should match
            assert refresh_trait["description"] == get_trait["description"]
            assert refresh_trait["weight"] == get_trait["weight"]

            # Should have same number of directives (gentle_refresh filters 0.0 weights, get_personality doesn't)
            # So we filter the get_personality results to match gentle_refresh behavior
            non_zero_directives = [
                d for d in get_trait["directives"] if d["weight"] != 0.0
            ]
            assert len(refresh_trait["directives"]) == len(non_zero_directives)

        print(
            f"🔄 Consistency verified between gentle_refresh and get_personality for {len(refresh_traits)} traits"
        )


@pytest.mark.asyncio
@performance_test
async def test_get_personality_performance_reasonable(test_stack):
    """Test get_personality performs within reasonable time bounds."""
    server_url = test_stack

    async with Client(server_url) as client:
        # Initialize system
        await time_mcp_call(client, "gentle_refresh", {})

        # Test get_personality performance
        result = await time_mcp_call(client, "get_personality", {})
        data = json.loads(result.content[0].text)

        assert data["success"] is True

        # Performance validation is handled by @performance_test decorator
        # Just ensure the call completed successfully
        print(
            f"🎯 Performance test completed for {data['trait_count']} traits, {data['directive_count']} directives"
        )
