#!/usr/bin/env bash
set -euo pipefail

echo "Dropping OS page cache..."
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo "Restarting Neo4j to clear page cache..."
sudo neo4j stop
sudo neo4j start

echo "Waiting for Neo4j to be ready..."
until neo4j status | grep -q "Neo4j is running"; do sleep 1; done

echo "Caches cleared, Neo4j is running."
