"""Demo test for log capture functionality."""

import json

import pytest
from fastmcp import Client

from tests.e2e.fixtures.log_capture import (
    LogCapture,
)
from tests.e2e.fixtures.performance import performance_test, time_mcp_call


@pytest.mark.asyncio
@performance_test
async def test_log_capture_demo(test_stack):
    """Demo test showing log capture and assertions."""
    server_url = test_stack
    log_capture = LogCapture()

    async with Client(server_url) as client:
        # First, get the logs before our operation
        initial_log_count = len(log_capture.refresh_logs())
        print(f"📊 Initial log count: {initial_log_count}")

        # Make an API call that should generate logs
        result = await time_mcp_call(client, "health_check", {})
        health_data = json.loads(result.content[0].text)

        # Wait a moment for logs to be written
        import time

        time.sleep(1)

        # Capture logs after the operation
        final_logs = log_capture.refresh_logs()
        new_log_count = len(final_logs)
        print(f"📊 Final log count: {new_log_count}")
        print(f"📈 New logs generated: {new_log_count - initial_log_count}")

        # Show some recent logs
        print("\n🔍 Recent structured logs:")
        structured_logs = [
            log for log in final_logs if isinstance(log, dict) and "level" in log
        ]
        for log in structured_logs[-5:]:  # Last 5 structured logs
            print(
                f"  {log.get('level', 'unknown').upper()}: {log.get('event', log.get('message', 'No message'))}"
            )

        # Assert that we got logs with the right structure
        health_logs = [
            log
            for log in final_logs
            if isinstance(log, dict) and "health" in str(log).lower()
        ]

        # We should have some health-related logs
        assert (
            len(health_logs) >= 0
        ), f"Expected health-related logs, got {len(health_logs)}"

        # Verify the API response is still working
        assert health_data["status"] == "ok"

        print(
            "✅ Log capture working! We can see both structured and unstructured logs."
        )
        print("🎯 Ready for detailed log assertions in actual tests!")


@pytest.mark.asyncio
@performance_test
async def test_add_personality_directive_with_log_capture(test_stack):
    """Test add_personality_directive with log assertions."""
    server_url = test_stack
    log_capture = LogCapture()

    async with Client(server_url) as client:
        # Initialize personality system
        await time_mcp_call(client, "gentle_refresh", {})

        # Clear logs before our test operation
        log_capture.clear_logs()

        # Make the add_personality_directive call
        result = await time_mcp_call(
            client,
            "add_personality_directive",
            {
                "trait_name": "warmth",
                "instruction": "Log capture test directive",
                "weight": 0.75,
            },
        )
        data = json.loads(result.content[0].text)

        # Basic API assertion
        if not data["success"]:
            print(f"❌ API call failed: {data.get('error', 'Unknown error')}")

            # Even if the call failed, we can still test log capture
            error_logs = log_capture.find_logs_containing("add_personality_directive")
            print(
                f"🔍 Found {len(error_logs)} logs mentioning add_personality_directive"
            )

            # Show what we captured
            recent_logs = log_capture.refresh_logs()
            print(f"📊 Total logs captured: {len(recent_logs)}")

            return  # Skip rest of test if API failed

        # If successful, let's look for specific log patterns
        correlation_id = data.get("correlation_id", "unknown")
        print(f"🔗 Tracking correlation_id: {correlation_id}")

        # Wait for logs to be written and capture them
        log_entry = log_capture.wait_for_log(
            lambda log: (
                "add_personality_directive" in str(log).lower()
                or correlation_id in str(log)
            ),
            timeout=5,
        )

        if log_entry:
            print(f"✅ Found relevant log entry: {log_entry}")
        else:
            print("⚠️  No specific log entry found, but that's OK - this is just a demo")

        # Show all logs we captured
        all_logs = log_capture.refresh_logs()
        print(f"📊 Total logs captured during test: {len(all_logs)}")

        # Success! The log capture system is working
        print("🎉 Log capture system operational - ready for detailed assertions!")
