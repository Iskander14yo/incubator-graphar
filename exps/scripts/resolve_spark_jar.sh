#!/usr/bin/env bash
set -euo pipefail

VERSION="$(
  cd maven-projects/spark &&
  mvn help:evaluate -Dexpression=project.version -q -DforceStdout 2>/dev/null |
    awk 'NF { print; exit }'
)"

if [[ -z "${VERSION}" ]]; then
  echo "Failed to resolve Maven project version."
  exit 1
fi

echo "maven-projects/spark/graphar/target/graphar-commons-${VERSION}-shaded.jar"
