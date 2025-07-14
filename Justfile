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
    @echo "Formatting code with ruff and black..."
    uv run --group dev ruff check --fix src/ tests/
    uv run --group dev black src/ tests/

check-format:
    @echo "Checking code formatting..."
    uv run --group dev ruff check src/ tests/
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

# Create hot backups of Alpha's databases
dump:
    #!/usr/bin/env sh
    set -e

    # Create backup directory with timestamp
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    echo "📸 Creating hot backups in $BACKUP_DIR..."

    # Redis backup
    echo "  → Backing up Redis..."
    docker exec redis redis-cli BGSAVE > /dev/null

    # Wait for Redis save to complete (with timeout)
    sleep 1
    TIMEOUT=30
    ELAPSED=0
    while true; do
        if docker exec redis redis-cli --raw INFO persistence | grep -q "rdb_bgsave_in_progress:0"; then
            break
        fi
        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo "  ⚠️  Redis backup timed out after ${TIMEOUT}s"
            exit 1
        fi
        sleep 0.5
        ELAPSED=$((ELAPSED + 1))
    done

    # Copy Redis dump
    docker cp redis:/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb"
    echo "  ✓ Redis backup complete"

    # Memgraph backup
    echo "  → Backing up Memgraph..."
    docker exec memgraph mgconsole "CREATE SNAPSHOT;" > /dev/null

    # Wait a moment for snapshot creation
    sleep 2

    # Get latest snapshot filename
    LATEST_SNAPSHOT=$(docker exec memgraph ls -t /var/lib/memgraph/snapshots/ | head -1)

    # Copy Memgraph snapshot
    docker cp "memgraph:/var/lib/memgraph/snapshots/$LATEST_SNAPSHOT" "$BACKUP_DIR/memgraph_snapshot"
    echo "  ✓ Memgraph backup complete"

    # Create metadata file
    echo "{" > "$BACKUP_DIR/metadata.json"
    echo "  \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$BACKUP_DIR/metadata.json"
    echo "  \"redis_size\": \"$(ls -lh $BACKUP_DIR/redis_dump.rdb | awk '{print $5}')\"," >> "$BACKUP_DIR/metadata.json"
    echo "  \"memgraph_size\": \"$(ls -lh $BACKUP_DIR/memgraph_snapshot | awk '{print $5}')\"," >> "$BACKUP_DIR/metadata.json"
    echo "  \"memgraph_snapshot\": \"$LATEST_SNAPSHOT\"" >> "$BACKUP_DIR/metadata.json"
    echo "}" >> "$BACKUP_DIR/metadata.json"

    echo ""
    echo "✅ Backup complete: $BACKUP_DIR"
    echo "   - Redis: $(ls -lh $BACKUP_DIR/redis_dump.rdb | awk '{print $5}')"
    echo "   - Memgraph: $(ls -lh $BACKUP_DIR/memgraph_snapshot | awk '{print $5}')"

# Restore databases from a backup
restore backup_dir:
    #!/usr/bin/env sh
    set -e

    if [ ! -d "{{backup_dir}}" ]; then
        echo "❌ Backup directory not found: {{backup_dir}}"
        exit 1
    fi

    echo "⚠️  WARNING: This will replace all current data!"
    echo "Restoring from: {{backup_dir}}"
    echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    sleep 5

    # Stop services
    echo "Stopping services..."
    docker compose stop redis memgraph

    # Restore Redis
    echo "Restoring Redis..."
    docker cp "{{backup_dir}}/redis_dump.rdb" redis:/data/dump.rdb
    docker exec redis chown redis:redis /data/dump.rdb

    # Restore Memgraph
    echo "Restoring Memgraph..."
    # Clear existing snapshots
    docker exec memgraph rm -rf /var/lib/memgraph/snapshots/*
    # Copy new snapshot
    docker cp "{{backup_dir}}/memgraph_snapshot" memgraph:/var/lib/memgraph/snapshots/
    docker exec memgraph chown -R memgraph:memgraph /var/lib/memgraph/snapshots/

    # Restart services
    echo "Restarting services..."
    docker compose start redis memgraph

    # Wait for services
    sleep 3

    echo "✅ Restore complete!"
    echo "   Note: You may need to restart alpha-recall for changes to take effect"
