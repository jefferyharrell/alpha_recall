"""Performance instrumentation utilities for E2E tests.

Provides timing decorators and performance metric collection for understanding
test behavior and identifying bottlenecks in Alpha-Recall operations.
"""

import functools
import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any


class PerformanceCollector:
    """Collects performance metrics during test execution."""

    def __init__(self):
        self.metrics: list[dict[str, Any]] = []
        self.test_name: str = ""

    def set_test_name(self, name: str) -> None:
        """Set the current test name for metric attribution."""
        self.test_name = name

    def record_operation(
        self, operation: str, duration_ms: float, details: dict[str, Any] = None
    ) -> None:
        """Record a single operation's performance."""
        metric = {
            "test": self.test_name,
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
            "details": details or {},
        }
        self.metrics.append(metric)

    def get_metrics(self) -> list[dict[str, Any]]:
        """Get all collected metrics."""
        return self.metrics.copy()

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()

    def get_test_summary(self, test_name: str = None) -> dict[str, Any]:
        """Get performance summary for a specific test."""
        target_test = test_name or self.test_name
        test_metrics = [m for m in self.metrics if m["test"] == target_test]

        if not test_metrics:
            return {"test": target_test, "total_operations": 0, "total_duration_ms": 0}

        total_duration = sum(m["duration_ms"] for m in test_metrics)
        operations = {}

        for metric in test_metrics:
            op = metric["operation"]
            if op not in operations:
                operations[op] = {"count": 0, "total_ms": 0, "avg_ms": 0}

            operations[op]["count"] += 1
            operations[op]["total_ms"] += metric["duration_ms"]
            operations[op]["avg_ms"] = (
                operations[op]["total_ms"] / operations[op]["count"]
            )

        return {
            "test": target_test,
            "total_operations": len(test_metrics),
            "total_duration_ms": total_duration,
            "operations": operations,
            "slowest_operation": max(test_metrics, key=lambda x: x["duration_ms"]),
        }


# Global collector instance
collector = PerformanceCollector()


@asynccontextmanager
async def time_operation(
    operation_name: str, details: dict[str, Any] = None
) -> AsyncGenerator[None, None]:
    """Context manager to time an async operation."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        collector.record_operation(operation_name, duration_ms, details)


def performance_test(func):
    """Decorator to add performance instrumentation to test functions."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Set test name for metric attribution
        test_name = func.__name__
        collector.set_test_name(test_name)

        # Time the entire test
        start_time = time.perf_counter()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            total_duration_ms = (end_time - start_time) * 1000

            # Record the total test duration
            collector.record_operation(
                "test_total", total_duration_ms, {"test_function": test_name}
            )

            # Print performance summary for this test
            summary = collector.get_test_summary(test_name)
            print(f"\n🔍 Performance Summary for {test_name}:")
            print(f"   Total Duration: {total_duration_ms:.1f}ms")
            print(f"   Operations: {summary['total_operations']}")

            if summary["operations"]:
                print("   Operation Breakdown:")
                for op_name, op_data in summary["operations"].items():
                    if op_name != "test_total":
                        print(
                            f"     {op_name}: {op_data['avg_ms']:.1f}ms avg ({op_data['count']} calls)"
                        )

                slowest = summary["slowest_operation"]
                if slowest["operation"] != "test_total":
                    print(
                        f"   Slowest Operation: {slowest['operation']} ({slowest['duration_ms']:.1f}ms)"
                    )

    return wrapper


async def time_mcp_call(client, tool_name: str, params: dict) -> Any:
    """Time an MCP tool call and return the result."""

    async with time_operation(
        f"mcp_call_{tool_name}", {"tool": tool_name, "params": params}
    ):
        result = await client.call_tool(tool_name, params)

    # Parse JSON response to get additional timing details if available
    try:
        data = json.loads(result.content[0].text)
        if "metadata" in data and "search_time_ms" in data["metadata"]:
            # Record server-side timing
            collector.record_operation(
                f"server_{tool_name}",
                data["metadata"]["search_time_ms"],
                {"server_side": True},
            )
    except (json.JSONDecodeError, KeyError, IndexError):
        pass  # No metadata available

    return result


# Export the key utilities
__all__ = [
    "PerformanceCollector",
    "collector",
    "time_operation",
    "performance_test",
    "time_mcp_call",
]
