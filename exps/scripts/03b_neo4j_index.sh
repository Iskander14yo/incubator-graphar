#!/usr/bin/env bash
set -euo pipefail

DB_NAME="${2:-neo4j}"

echo "Waiting for Neo4j to be ready..."
until neo4j status | grep -q "Neo4j is running"; do sleep 1; done

echo "Creating node_id index on database '$DB_NAME'..."
cypher-shell -d "$DB_NAME" \
    "CREATE INDEX node_id IF NOT EXISTS FOR (n:node) ON (n.id)"

echo "Waiting for index to come online (up to 5 minutes)..."
cypher-shell -d "$DB_NAME" "CALL db.awaitIndexes(300)"

echo "Index ready on database '$DB_NAME'."
