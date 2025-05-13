#!/bin/bash

# Neo4j Backup Script for Alpha Recall Project

# Exit on any error
set -e

# Configuration
NEO4J_HOME="${NEO4J_HOME:-/opt/homebrew}"  # Default Neo4j installation path, adjust if different
BACKUP_DIR="${BACKUP_DIR:-/Users/jefferyharrell/Projects/Alpha/BACKUPS/neo4j}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="neo4j_backup_${TIMESTAMP}"

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Function to print error and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Check if Neo4j is installed
[ -d "${NEO4J_HOME}" ] || error_exit "Neo4j home directory not found. Please set NEO4J_HOME correctly."

# Stop Neo4j
echo "Stopping Neo4j database..."
"${NEO4J_HOME}/bin/neo4j" stop || error_exit "Failed to stop Neo4j"

# Perform backup
echo "Creating backup of Neo4j database..."
"${NEO4J_HOME}/bin/neo4j-admin" database dump \
    --to-path="${BACKUP_DIR}" \
    neo4j \
    || error_exit "Backup failed"

# Backup successful message
echo "Backup completed successfully: ${BACKUP_DIR}/${BACKUP_NAME}"

# Restore instructions
cat << EOF

Restore Instructions:
1. Stop Neo4j: ${NEO4J_HOME}/bin/neo4j stop
2. Restore command: 
   ${NEO4J_HOME}/bin/neo4j-admin database restore \\
   --from=${BACKUP_DIR}/${BACKUP_NAME} \\
   --to=neo4j

Note: Ensure you're using the same Neo4j version for backup and restore.
EOF
