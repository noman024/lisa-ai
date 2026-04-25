#!/usr/bin/env bash
# ============================================================
# Start the vLLM-Metal server (Apple Silicon, MLX backend).
# Exposes an OpenAI-compatible API on :8001.
#
# Prerequisites (one-time):
#   curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
#
# Usage (from repo root):
#   bash scripts/start_vllm.sh
# ============================================================
set -euo pipefail

VENV="${HOME}/.venv-vllm-metal"
MODEL="${VLLM_MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8001}"
API_KEY="${VLLM_API_KEY:-not-needed}"

if [[ ! -d "$VENV" ]]; then
  echo "ERROR: vllm-metal venv not found at $VENV" >&2
  echo "Install it with:" >&2
  echo "  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash" >&2
  exit 1
fi

echo "==> Activating vllm-metal venv: $VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "==> Starting vLLM-Metal server"
echo "    Model : $MODEL"
echo "    Listen: $HOST:$PORT"
echo "    API   : http://$HOST:$PORT/v1"
    echo "    (first run will download the model — Qwen2.5-3B-4bit ≈ 2 GB)"
echo ""

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --max-model-len 4096
