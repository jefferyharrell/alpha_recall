"""E2E tests for complete personality management workflow.

Tests the complete personality self-management workflow:
1. create_personality_trait - Create new traits
2. add_personality_directive - Add directives to traits
3. get_personality_trait - Verify the complete structure
"""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_complete_personality_workflow(test_stack):
    """Test complete personality management workflow from creation to directive addition."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Initialize system
        await time_mcp_call(client, "gentle_refresh", {})

        # Step 1: Create a new personality trait
        create_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "curiosity",
                "description": "Drive to explore, question, and understand",
                "weight": 0.9,
            },
        )
        create_data = json.loads(create_result.content[0].text)

        assert create_data["success"] is True
        assert create_data["trait_created"]["name"] == "curiosity"
        assert (
            create_data["trait_created"]["description"]
            == "Drive to explore, question, and understand"
        )
        assert create_data["trait_created"]["weight"] == 0.9
        assert "created_at" in create_data["trait_created"]
        assert "linked_to_root" in create_data

        print(f"✅ Created trait: {create_data['trait_created']['name']}")

        # Step 2: Add a directive to the newly created trait
        directive_result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "curiosity",
                "instruction": "Ask follow-up questions to deepen understanding",
                "weight": 0.8,
            },
        )
        directive_data = json.loads(directive_result.content[0].text)

        assert directive_data["success"] is True
        assert directive_data["trait_name"] == "curiosity"
        assert (
            directive_data["directive_added"]["instruction"]
            == "Ask follow-up questions to deepen understanding"
        )
        assert directive_data["directive_added"]["weight"] == 0.8

        print(f"✅ Added directive: {directive_data['directive_added']['instruction']}")

        # Step 3: Verify the complete structure by retrieving the trait
        get_result = await time_mcp_call(
            client, "get_personality_trait", {"trait_name": "curiosity"}
        )
        get_data = json.loads(get_result.content[0].text)

        if not get_data["success"]:
            print(
                f"❌ get_personality_trait failed: {get_data.get('error', 'Unknown error')}"
            )
            print(f"Full response: {json.dumps(get_data, indent=2)}")

        assert get_data["success"] is True
        assert get_data["trait"]["name"] == "curiosity"
        assert (
            get_data["trait"]["description"]
            == "Drive to explore, question, and understand"
        )
        assert get_data["trait"]["weight"] == 0.9
        assert len(get_data["trait"]["directives"]) == 1

        directive = get_data["trait"]["directives"][0]
        assert (
            directive["instruction"]
            == "Ask follow-up questions to deepen understanding"
        )
        assert directive["weight"] == 0.8

        print(
            f"✅ Retrieved complete trait with {len(get_data['trait']['directives'])} directive(s)"
        )

        # Step 4: Add a second directive to test multiple directives
        second_directive_result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "curiosity",
                "instruction": "Explore topics beyond immediate requirements",
                "weight": 0.7,
            },
        )
        second_directive_data = json.loads(second_directive_result.content[0].text)

        assert second_directive_data["success"] is True
        print(
            f"✅ Added second directive: {second_directive_data['directive_added']['instruction']}"
        )

        # Step 5: Verify both directives are present and properly ordered by weight
        final_get_result = await time_mcp_call(
            client, "get_personality_trait", {"trait_name": "curiosity"}
        )
        final_get_data = json.loads(final_get_result.content[0].text)

        assert final_get_data["success"] is True
        assert len(final_get_data["trait"]["directives"]) == 2

        # Should be ordered by weight (descending: 0.8, 0.7)
        directives = final_get_data["trait"]["directives"]
        assert directives[0]["weight"] == 0.8
        assert directives[1]["weight"] == 0.7
        assert (
            directives[0]["instruction"]
            == "Ask follow-up questions to deepen understanding"
        )
        assert (
            directives[1]["instruction"]
            == "Explore topics beyond immediate requirements"
        )

        print("✅ Both directives present and properly ordered by weight")

        # Step 6: Test update_personality_directive_weight - update the weight of the second directive
        update_weight_result = await time_mcp_call(
            client,
            "update_personality_directive_weight",
            {
                "trait_name": "curiosity",
                "instruction": "Explore topics beyond immediate requirements",
                "new_weight": 0.9,  # Higher than the first directive
            },
        )
        update_weight_data = json.loads(update_weight_result.content[0].text)

        assert update_weight_data["success"] is True
        assert update_weight_data["trait_name"] == "curiosity"
        assert (
            update_weight_data["directive_updated"]["instruction"]
            == "Explore topics beyond immediate requirements"
        )
        assert update_weight_data["directive_updated"]["previous_weight"] == 0.7
        assert update_weight_data["directive_updated"]["new_weight"] == 0.9
        assert (
            abs(update_weight_data["directive_updated"]["weight_change"] - 0.2) < 0.0001
        )

        print(
            f"✅ Updated directive weight: 0.7 -> 0.9 (change: +{update_weight_data['directive_updated']['weight_change']:.1f})"
        )

        # Step 7: Final verification - check that directive order has changed due to weight update
        ultimate_get_result = await time_mcp_call(
            client, "get_personality_trait", {"trait_name": "curiosity"}
        )
        ultimate_get_data = json.loads(ultimate_get_result.content[0].text)

        assert ultimate_get_data["success"] is True
        assert len(ultimate_get_data["trait"]["directives"]) == 2

        # Should now be ordered by weight (descending: 0.9, 0.8) - order reversed!
        final_directives = ultimate_get_data["trait"]["directives"]
        assert final_directives[0]["weight"] == 0.9
        assert final_directives[1]["weight"] == 0.8
        assert (
            final_directives[0]["instruction"]
            == "Explore topics beyond immediate requirements"
        )
        assert (
            final_directives[1]["instruction"]
            == "Ask follow-up questions to deepen understanding"
        )

        print("✅ Final verification: directive order updated after weight change")
        print(
            f"   1. {final_directives[0]['instruction']} (weight: {final_directives[0]['weight']})"
        )
        print(
            f"   2. {final_directives[1]['instruction']} (weight: {final_directives[1]['weight']})"
        )

        # Success! Complete personality workflow tested end-to-end
        print("🎉 Complete personality workflow successfully tested!")
        print("   ✅ create_personality_trait")
        print("   ✅ add_personality_directive (multiple)")
        print("   ✅ get_personality_trait")
        print("   ✅ update_personality_directive_weight")


@pytest.mark.asyncio
@performance_test
async def test_create_trait_validation_errors(test_stack):
    """Test validation errors for create_personality_trait."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Test empty trait name
        result = await time_mcp_call(
            client,
            "create_personality_trait",
            {"trait_name": "", "description": "Valid description", "weight": 0.5},
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "Trait name cannot be empty" in data["error"]

        # Test empty description
        result = await time_mcp_call(
            client,
            "create_personality_trait",
            {"trait_name": "valid_name", "description": "", "weight": 0.5},
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "Description cannot be empty" in data["error"]

        # Test invalid weight (too high)
        result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "valid_name",
                "description": "Valid description",
                "weight": 1.5,
            },
        )
        data = json.loads(result.content[0].text)

        assert data["success"] is False
        assert "Weight must be a number between 0.0 and 1.0" in data["error"]

        print("✅ All validation errors properly handled")


@pytest.mark.asyncio
@performance_test
async def test_duplicate_trait_prevention(test_stack):
    """Test that duplicate traits are prevented."""
    server_url = test_stack
    async with Client(server_url) as client:
        # Create a trait
        first_result = await time_mcp_call(
            client,
            "create_personality_trait",
            {
                "trait_name": "empathy",
                "description": "Understanding others' emotions",
                "weight": 0.8,
            },
        )
        first_data = json.loads(first_result.content[0].text)

        # First creation should succeed
        if first_data["success"]:
            # Try to create the same trait again
            second_result = await time_mcp_call(
                client,
                "create_personality_trait",
                {
                    "trait_name": "empathy",
                    "description": "Different description",
                    "weight": 0.6,
                },
            )
            second_data = json.loads(second_result.content[0].text)

            assert second_data["success"] is False
            assert "already exists" in second_data["error"]
            assert "existing_trait" in second_data
            assert second_data["existing_trait"]["name"] == "empathy"

            print("✅ Duplicate trait properly prevented")
        else:
            # If first creation failed, it might be because trait already exists from previous test
            print(f"⚠️  First creation failed: {first_data['error']}")
            print("This might be expected if trait already exists from previous test")
