#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
VENV_PIP="${ROOT_DIR}/.venv/bin/pip"

install_neo4j_if_missing() {
  if command -v neo4j >/dev/null 2>&1 && command -v neo4j-admin >/dev/null 2>&1; then
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install Neo4j."
    exit 1
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Automatic Neo4j install currently supports Debian/Ubuntu (apt-get)."
    exit 1
  fi

  echo "Installing Neo4j via apt..."
  sudo apt-get update
  sudo apt-get install -y wget gnupg ca-certificates
  sudo install -d -m 0755 /etc/apt/keyrings
  wget -qO - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor --batch --yes -o /etc/apt/keyrings/neotechnology.gpg
  echo "deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest" | sudo tee /etc/apt/sources.list.d/neo4j.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y neo4j
}

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Creating virtualenv at .venv"
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -e "./python[ml-benchmark]"

install_neo4j_if_missing

if ! command -v neo4j >/dev/null 2>&1 || ! command -v neo4j-admin >/dev/null 2>&1; then
  echo "Neo4j installation failed or binaries are not in PATH."
  exit 1
fi

echo "Setup OK: Python deps installed, Neo4j installed."
