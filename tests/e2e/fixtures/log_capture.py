"""Log capture utilities for E2E tests."""

import json
import subprocess
import time
from collections.abc import Callable


class LogCapture:
    """Capture and parse structured logs from Docker container stdout."""

    def __init__(self, container_name: str = "alpha-recall-test-server"):
        self.container_name = container_name
        self.parsed_logs: list[dict] = []

    def refresh_logs(self) -> list[dict]:
        """Read and parse all logs from Docker container."""
        self.parsed_logs = []

        try:
            # Get logs from Docker container
            result = subprocess.run(
                ["docker", "logs", self.container_name],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse each line of stdout
            for line in result.stdout.split("\n"):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse JSON log entry
                    log_entry = json.loads(line)
                    self.parsed_logs.append(log_entry)
                except json.JSONDecodeError:
                    # Plain text log - store as raw message
                    self.parsed_logs.append(
                        {"message": line, "level": "raw", "timestamp": time.time()}
                    )

        except subprocess.CalledProcessError:
            # Container might not be running yet
            pass

        return self.parsed_logs

    def wait_for_log(
        self,
        condition: Callable[[dict], bool],
        timeout: int = 10,
        refresh_interval: float = 0.5,
    ) -> dict | None:
        """Wait for a log entry matching the condition."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            self.refresh_logs()

            for log_entry in self.parsed_logs:
                if condition(log_entry):
                    return log_entry

            time.sleep(refresh_interval)

        return None

    def find_logs_by_correlation_id(self, correlation_id: str) -> list[dict]:
        """Find all log entries with a specific correlation ID."""
        self.refresh_logs()
        return [
            log
            for log in self.parsed_logs
            if log.get("correlation_id") == correlation_id
        ]

    def find_logs_by_level(self, level: str) -> list[dict]:
        """Find all log entries at a specific level."""
        self.refresh_logs()
        return [log for log in self.parsed_logs if log.get("level") == level]

    def find_logs_containing(self, text: str) -> list[dict]:
        """Find all log entries containing specific text."""
        self.refresh_logs()
        return [log for log in self.parsed_logs if text in str(log.get("message", ""))]

    def assert_log_exists(
        self,
        condition: Callable[[dict], bool],
        timeout: int = 10,
        error_message: str = "Expected log entry not found",
    ) -> dict:
        """Assert that a log entry matching condition exists."""
        log_entry = self.wait_for_log(condition, timeout)

        if log_entry is None:
            # Include recent logs in assertion error for debugging
            recent_logs = self.parsed_logs[-10:] if self.parsed_logs else []
            raise AssertionError(
                f"{error_message}. " f"Recent logs: {json.dumps(recent_logs, indent=2)}"
            )

        return log_entry

    def clear_logs(self):
        """Clear cached logs (useful for test isolation)."""
        self.parsed_logs = []


# Helper functions for common log assertions
def correlation_id_condition(correlation_id: str) -> Callable[[dict], bool]:
    """Create condition function for matching correlation ID."""
    return lambda log: log.get("correlation_id") == correlation_id


def level_and_message_condition(
    level: str, message_substring: str
) -> Callable[[dict], bool]:
    """Create condition function for matching level and message content."""
    return lambda log: (
        log.get("level") == level and message_substring in str(log.get("message", ""))
    )


def tool_name_condition(tool_name: str) -> Callable[[dict], bool]:
    """Create condition function for matching tool name in logs."""
    return lambda log: tool_name in str(log.get("message", ""))
