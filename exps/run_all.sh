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
.venv/bin/python exps/scripts/02_convert_gar.py --dataset "${DATASET}" --spark-jar "${SPARK_JAR}"
