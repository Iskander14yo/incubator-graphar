#!/usr/bin/env bash
# Full pipeline for benchmark prep + smoke tests + benchmark run.
# Ground truth: exps/config/benchmark.yaml (override path with first argument).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_CONFIG="${1:-exps/config/benchmark.yaml}"
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

echo "Download dataset (benchmark config: ${BENCHMARK_CONFIG})"
.venv/bin/python exps/scripts/01_download.py --config "${BENCHMARK_CONFIG}"

echo "Convert dataset to GAR (benchmark config: ${BENCHMARK_CONFIG})"
.venv/bin/python exps/scripts/02_load_gar.py --config "${BENCHMARK_CONFIG}"

echo "Load dataset into Neo4j (benchmark config: ${BENCHMARK_CONFIG})"
.venv/bin/python exps/scripts/03_load_neo4j.py --config "${BENCHMARK_CONFIG}"

echo "Create Neo4j index"
bash exps/scripts/03b_neo4j_index.sh

echo "Configure Neo4j"
bash exps/scripts/03c_neo4j_conf.sh

echo "Verify GAR and Neo4j (benchmark config: ${BENCHMARK_CONFIG})"
.venv/bin/python exps/scripts/04_verify_formats.py --config "${BENCHMARK_CONFIG}"

echo "Smoke test: GAR training loop"
.venv/bin/python exps/scripts/05_smoke_train_gar.py --config "${BENCHMARK_CONFIG}"

echo "Smoke test: Neo4j loaders"
.venv/bin/python exps/scripts/05_smoke_train_neo4j.py --config "${BENCHMARK_CONFIG}"

echo "Run benchmark"
sudo perf record -g -F 99 --call-graph dwarf .venv/bin/python exps/scripts/04_run_benchmark.py --config "${BENCHMARK_CONFIG}"
sudo perf report --stdio > exps/results/profiles/gar_perf.txt
sudo perf script | ./FlameGraph/stackcollapse-perf.pl > out.folded
