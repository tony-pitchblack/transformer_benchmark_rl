#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing .env: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

PORT_CLEAN="$(printf '%s' "${MLFLOW_PORT:-5000}" | tr -d "\"'")"
HOST_CLEAN="$(printf '%s' "${MLFLOW_HOST:-localhost}" | tr -d "\"'")"
DISPLAY_HOST="$HOST_CLEAN"
if [ -z "$DISPLAY_HOST" ] || [ "$DISPLAY_HOST" = "0.0.0.0" ] || [ "$DISPLAY_HOST" = "::" ]; then
  DISPLAY_HOST="localhost"
fi

if [ -x "$PROJECT_ROOT/.venv/bin/mlflow" ]; then
  MLFLOW_BIN="$PROJECT_ROOT/.venv/bin/mlflow"
elif command -v mlflow >/dev/null 2>&1; then
  MLFLOW_BIN="$(command -v mlflow)"
else
  echo "mlflow not found in .venv or PATH" >&2
  exit 1
fi

SESSION="mlflow-${PORT_CLEAN}"
ROOT="$PROJECT_ROOT/mlflow"
mkdir -p "$ROOT/artifacts"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "http://${DISPLAY_HOST}:${PORT_CLEAN}"
  echo "tmux attach -t ${SESSION}"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd \"$PROJECT_ROOT\" && set -a && source \"$ENV_FILE\" && set +a && exec \"$MLFLOW_BIN\" server --host \"$HOST_CLEAN\" --port \"$PORT_CLEAN\" --backend-store-uri \"sqlite:///$ROOT/mlflow.db\" --default-artifact-root \"file://$ROOT/artifacts\" --serve-artifacts --workers 1"

for _ in $(seq 1 10); do
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "http://${DISPLAY_HOST}:${PORT_CLEAN}"
    echo "tmux attach -t ${SESSION}"
    exit 0
  fi
  sleep 0.2
done

echo "Failed to start tmux session: ${SESSION}" >&2
exit 1
