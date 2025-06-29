# Use Python 3.11 slim image
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    cmake \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Expose the HTTP port
EXPOSE 8080

# Production stage (default)
FROM base AS production
COPY src/ ./src/
CMD ["uv", "run", "python", "-m", "alpha_recall.fastmcp_server"]

# Development stage (for bind mounting source code)
FROM base AS dev
# Don't copy source code - it will be bind mounted
# Just wait for the source to be mounted and then start
CMD ["uv", "run", "python", "-m", "alpha_recall.fastmcp_server"]