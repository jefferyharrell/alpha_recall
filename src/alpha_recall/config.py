"""Configuration handling for Alpha-Recall."""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlphaRecallSettings(BaseSettings):
    """Alpha-Recall configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # MCP Server Configuration
    mcp_transport: Literal["sse", "streamable-http"] = "streamable-http"
    host: str = "localhost"
    port: int | None = None

    # Development Configuration
    alpha_recall_dev_port: int = Field(default=19005, ge=1, le=65535)

    # Logging
    log_level: str = "INFO"
    log_format: Literal["rich", "json", "rich_json"] = "rich"

    # Memgraph Database Configuration
    memgraph_uri: str = "bolt://localhost:7687"

    # Redis Configuration
    redis_uri: str = "redis://localhost:6379/0"
    redis_ttl: int = Field(default=2000000, gt=0)  # 2 megaseconds

    # Embedding Models
    semantic_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    emotional_embedding_model: str = "ng3owb/sentiment-embedding-model"
    inference_device: str | None = None

    # Core Configuration
    core_identity_node: str = "Alpha Core Identity"

    # Memory Consolidation Configuration
    memory_consolidation_enabled: bool = True
    helper_model: str = "llama3.2:1b"
    consolidation_ollama_host: str = "localhost"
    consolidation_ollama_port: int = Field(default=11434, ge=1, le=65535)
    consolidation_time_window: str = "24h"
    consolidation_timeout: int = Field(default=60, gt=0)

    # Alpha-Reminiscer Configuration
    reminiscer_enabled: bool = False
    reminiscer_ollama_host: str = "localhost"
    reminiscer_ollama_port: int = Field(default=11434, ge=1, le=65535)
    reminiscer_model: str = "qwen2.5:7b"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:  # type: ignore
        """Validate log level is a valid logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


# Global settings instance
settings = AlphaRecallSettings()
