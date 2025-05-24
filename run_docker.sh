#!/usr/bin/env bash
set -euo pipefail

# Log file
LOGFILE="$(dirname "$0")/run_docker.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== $(date '+%Y-%m-%d %H:%M:%S') - Starting Docker orchestrator ==="

# Pull latest images
echo "Pulling latest images..."
docker-compose pull || {
  echo "[ERROR] Failed to pull images"
  exit 1
}

echo "Starting services..."
docker-compose up -d --build --remove-orphans || {
  echo "[ERROR] Failed to start services"
  exit 1
}

echo "Current status (docker-compose ps):"
docker-compose ps || {
  echo "[ERROR] Failed to get status"
  exit 1
}

echo "=== $(date '+%Y-%m-%d %H:%M:%S') - Docker orchestrator completed successfully ==="
