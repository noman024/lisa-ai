#!/usr/bin/env bash
# ============================================================
# LISA Full Stack Launcher
# Starts vLLM-Metal server AND the LISA FastAPI backend
# together in background processes (foreground monitor).
#
# Usage (from repo root):
#   bash scripts/start_all.sh
#
# Stop with Ctrl-C — both services are gracefully killed.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

VENV="${HOME}/.venv-vllm-metal"
PY="${LISA_PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${VLLM_MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
VLLM_PORT="${VLLM_PORT:-8001}"
LISA_PORT="${LISA_PORT:-8000}"
# Must match vllm serve --api-key (OpenAI client sends Authorization: Bearer …)
VLLM_API_KEY="${VLLM_API_KEY:-not-needed}"
VLLM_LOG="$ROOT/logs/vllm.log"
LISA_LOG="$ROOT/logs/lisa.log"

mkdir -p "$ROOT/logs"

cleanup() {
  echo ""
  echo "==> Shutting down services..."
  [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
  [[ -n "${LISA_PID:-}" ]]  && kill "$LISA_PID"  2>/dev/null || true
  wait 2>/dev/null || true
  echo "==> Done."
}
trap cleanup EXIT INT TERM

# --- Validate prerequisites ---
if [[ ! -d "$VENV" ]]; then
  echo "ERROR: vllm-metal venv not found at $VENV" >&2
  echo "Install:  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "ERROR: LISA Python not found at $PY" >&2
  echo "Create venv:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# --- 1. Start vLLM-Metal ---
echo "==> [1/3] Starting vLLM-Metal server on :$VLLM_PORT (model: $MODEL)"
echo "    Logs: $VLLM_LOG"
(
  source "$VENV/bin/activate"
  exec vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --api-key "$VLLM_API_KEY" \
    --max-model-len 4096
) >> "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# --- 2. Wait for vLLM to be ready (up to 3 minutes for first-run model download) ---
echo "==> [2/3] Waiting for vLLM server to become ready..."
MAX_WAIT=180
WAITED=0
until curl -sf --max-time 3 \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM process died. Check logs: $VLLM_LOG" >&2
    exit 1
  fi
  if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo "ERROR: vLLM did not become ready within ${MAX_WAIT}s. Check $VLLM_LOG" >&2
    exit 1
  fi
  echo -n "."
  sleep 5
  WAITED=$((WAITED + 5))
done
echo ""
echo "    vLLM ready at http://127.0.0.1:${VLLM_PORT}/v1"

# --- 3. Ingest KB and start LISA ---
echo "==> [3/3] Ingesting KB and starting LISA API on :$LISA_PORT"
echo "    Logs: $LISA_LOG"
"$PY" scripts/ingest_kb.py >> "$LISA_LOG" 2>&1
"$PY" -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$LISA_PORT" \
  --log-level info >> "$LISA_LOG" 2>&1 &
LISA_PID=$!

# Wait for LISA to be ready
WAITED=0
until curl -sf --max-time 3 "http://127.0.0.1:${LISA_PORT}/health" > /dev/null 2>&1; do
  if ! kill -0 "$LISA_PID" 2>/dev/null; then
    echo "ERROR: LISA process died. Check logs: $LISA_LOG" >&2
    exit 1
  fi
  if [[ $WAITED -ge 60 ]]; then
    echo "ERROR: LISA API did not become ready within 60s. Check $LISA_LOG" >&2
    exit 1
  fi
  echo -n "."
  sleep 2
  WAITED=$((WAITED + 2))
done
echo ""
echo ""
echo "============================================================"
echo "  LISA is fully operational!"
echo "  vLLM-Metal : http://127.0.0.1:${VLLM_PORT}/v1"
echo "  LISA API   : http://127.0.0.1:${LISA_PORT}"
echo "  Swagger UI : http://127.0.0.1:${LISA_PORT}/docs"
echo ""
echo "  Verify:  python scripts/verify_e2e.py --skip-llm-ping"
echo "  Chat:    python scripts/chat_cli.py"
echo "============================================================"
echo ""
echo "Tailing logs (Ctrl-C to stop all services):"
tail -f "$VLLM_LOG" "$LISA_LOG"
