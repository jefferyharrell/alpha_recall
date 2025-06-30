# Alpha-Recall v1.0.0 Development Commands

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

# View logs (add -f to follow)
logs service="" *flags="":
    #!/usr/bin/env sh
    if [ -z "{{service}}" ]; then
        echo "Viewing logs for all services..."
        docker compose logs {{flags}}
    else
        echo "Viewing logs for {{service}}..."
        docker compose logs {{flags}} {{service}}
    fi

# Rebuild and restart (useful for development)
restart service="":
    @just down {{service}}
    @just up {{service}}

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

# Follow logs for alpha-recall-1 (most common use case)
follow:
    @just logs alpha-recall-1 -f

# Quick health check
health:
    @echo "Alpha-Recall v1.0.0 Health Check:"
    @curl -s http://localhost:19005/health || echo "Service not responding"