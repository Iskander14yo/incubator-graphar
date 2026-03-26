#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-ogbn-products}"
SPARK_JAR="${2:-${GRAPHAR_SPARK_JAR:-}}"

echo "Environment setup"
bash exps/scripts/00_setup.sh

if [[ -z "${SPARK_JAR}" ]]; then
  SPARK_JAR="$(bash exps/scripts/resolve_spark_jar.sh)"
fi

if [[ ! -f "${SPARK_JAR}" ]]; then
  echo "Failed to detect GraphAr Spark JAR. Run exps/scripts/00_setup.sh first."
  exit 1
fi

echo "Download dataset: ${DATASET}"
.venv/bin/python exps/scripts/01_download.py --dataset "${DATASET}"

echo "Convert dataset to GAR: ${DATASET}"
.venv/bin/python exps/scripts/02_load_gar.py --dataset "${DATASET}"

echo "Load dataset into Neo4j: ${DATASET}"
.venv/bin/python exps/scripts/03_load_neo4j.py --dataset "${DATASET}"

echo "Create Neo4j index: ${DATASET}"
bash exps/scripts/03b_neo4j_index.sh "${DATASET}"

echo "Configure Neo4j"
bash exps/scripts/03c_neo4j_conf.sh

echo "Verify GAR and Neo4j: ${DATASET}"
.venv/bin/python exps/scripts/04_verify_formats.py --dataset "${DATASET}"

echo "Smoke test: GAR training loop"
.venv/bin/python exps/scripts/05_smoke_train_gar.py --dataset "${DATASET}"

echo "Smoke test: Neo4j loaders"
.venv/bin/python exps/scripts/05_smoke_train_neo4j.py --dataset "${DATASET}"

echo "Run benchmark"
.venv/bin/python exps/scripts/04_run_benchmark.py --config exps/config/benchmark.yaml
