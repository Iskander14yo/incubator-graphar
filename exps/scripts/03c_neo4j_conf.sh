#!/usr/bin/env bash
# Apply Neo4j config required for reproducible benchmark runs.
# Safe to re-run: uses sed for keys that already exist, appends new ones.
set -euo pipefail

CONF=/etc/neo4j/neo4j.conf

set_key() {
    local key="$1" value="$2"
    if sudo grep -qE "^#?${key}=" "$CONF"; then
        sudo sed -i "s|^#\?${key}=.*|${key}=${value}|" "$CONF"
    else
        echo "${key}=${value}" | sudo tee -a "$CONF" >/dev/null
    fi
}

echo "Applying Neo4j config to $CONF..."

# Memory — kept small so page-cache pressure is measurable in cold runs
set_key "server.memory.pagecache.size"    "4g"
set_key "server.memory.heap.initial_size" "2g"
set_key "server.memory.heap.max_size"     "4g"

# Auth — disabled for scripting convenience on the benchmark machine
set_key "dbms.security.auth_enabled" "false"

# Query logging — passive logging of planning/cpu/wait times per query.
# Keys renamed in Neo4j 2026.x: db.logs.query.* (old dbms.logs.query.* deprecated).
# time_logging_enabled and page_logging_enabled removed (always on when logging is enabled).
set_key "db.logs.query.enabled"   "INFO"
set_key "db.logs.query.threshold" "0ms"

# Fix ownership — neo4j-admin import (run as root/sudo) leaves database files
# owned by root; the neo4j service user cannot open them → DatabaseUnavailable.
echo "Fixing database file ownership..."
sudo chown -R neo4j:neo4j /var/lib/neo4j/data/databases/

echo "Restarting Neo4j..."
sudo neo4j restart

echo "Waiting for Neo4j to be ready..."
until neo4j status | grep -q "Neo4j is running"; do sleep 1; done

echo "Neo4j config applied and service is running."
