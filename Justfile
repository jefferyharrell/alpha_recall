# Alpha-Recall Development Commands

# Default recipe - show available commands
default:
    @just --list

# Start services with docker compose
up service="":
    #!/usr/bin/env sh
    if [ -z "{{service}}" ]; then
        echo "Starting all services..."
        docker compose up -d
    else
        echo "Starting {{service}}..."
        docker compose up -d {{service}}
    fi

# Stop services with docker compose
down service="":
    #!/usr/bin/env sh
    if [ -z "{{service}}" ]; then
        echo "Stopping all services..."
        docker compose down
    else
        echo "Stopping {{service}}..."
        docker compose stop {{service}}
    fi

# Hard-restart the whole stack
bounce service="":
    @just down {{service}}
    @just up {{service}}

# View logs
logs service="":
    #!/usr/bin/env sh
    if [ -z "{{service}}" ]; then
        echo "Viewing logs for all services..."
        docker compose logs
    else
        echo "Viewing logs for {{service}}..."
        docker compose logs {{service}}
    fi

# Follow logs
follow service="":
    #!/usr/bin/env sh
    if [ -z "{{service}}" ]; then
        echo "Viewing logs for all services..."
        docker compose logs -f
    else
        echo "Viewing logs for {{service}}..."
        docker compose logs -f {{service}}
    fi

# Rebuild and restart (useful for development)
restart service="":
    #!/usr/bin/env sh
    if [ -z "{{service}}" ]; then
        echo "Restarting all services..."
        docker compose restart
    else
        echo "Restarting {{service}}..."
        docker compose restart {{service}}
    fi

# Build images without starting
build service="":
    #!/usr/bin/env sh
    if [ -z "{{service}}" ]; then
        echo "Building all services..."
        docker compose build
    else
        echo "Building {{service}}..."
        docker compose build {{service}}
    fi

# Show running containers
ps:
    @echo "Container status:"
    @docker compose ps

# Clean up everything (containers, images, volumes)
clean:
    @echo "Cleaning up containers, images, and volumes..."
    docker compose down --volumes --remove-orphans
    docker compose rm -f
    docker system prune -f

# Development tools
format:
    @echo "Formatting code with isort and black..."
    uv run --group dev isort src/ tests/
    uv run --group dev black src/ tests/

check-format:
    @echo "Checking code formatting..."
    uv run --group dev isort --check-only --diff src/ tests/
    uv run --group dev black --check --diff src/ tests/

lint:
    @echo "Running Ruff linter..."
    uv run --group dev ruff check src/ tests/

lint-fix:
    @echo "Running Ruff linter with auto-fix..."
    uv run --group dev ruff check --fix src/ tests/

pre-commit:
    @echo "Running pre-commit on all files..."
    uv run --group dev pre-commit run --all-files

test:
    @echo "Running all tests (parallel unit tests → serial e2e tests)..."
    just test-unit
    just test-e2e

test-unit:
    @echo "Running unit tests in parallel..."
    uv run --group test pytest tests/unit/ -n 4 -v

test-e2e:
    @echo "Running e2e tests in proper order: warm-up → greenfield → seeded → personality..."
    uv run --group test pytest \
        tests/e2e/test_greenfield_health.py \
        tests/e2e/test_seeded_longterm_memory.py \
        tests/e2e/test_seeded_shortterm_memory.py \
        tests/e2e/test_seeded_narrative_memory.py \
        tests/e2e/test_seeded_search_all_memories.py \
        tests/e2e/test_personality_workflow.py \
        tests/e2e/test_add_personality_directive.py \
        tests/e2e/test_get_personality_trait.py \
        tests/e2e/test_get_personality.py \
        -v -s --maxfail=1

# Export dependencies to requirements.txt
export-requirements:
    @echo "Exporting dependencies to requirements.txt..."
    uv export --format requirements-txt > requirements.txt
