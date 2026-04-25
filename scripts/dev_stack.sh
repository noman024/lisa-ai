#!/usr/bin/env bash
# ============================================================
# Start the LISA API stack (FastAPI + LangGraph + FAISS).
# Run vLLM-Metal FIRST in another terminal:
#   bash scripts/start_vllm.sh
#
# Usage (from repo root):
#   bash scripts/dev_stack.sh
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${LISA_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python not found at $PY" >&2
  echo "Create venv:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# Confirm vLLM-Metal server is up (same Bearer as vllm's --api-key)
VLLM_API_KEY="${VLLM_API_KEY:-not-needed}"
VLLM_URL="${VLLM_BASE_URL:-http://127.0.0.1:8001}/v1/models"
echo "==> Checking vLLM-Metal server at ${VLLM_URL} ..."
if ! curl -sf --max-time 5 \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  "$VLLM_URL" > /dev/null 2>&1; then
  echo "WARNING: vLLM server not reachable at ${VLLM_URL}." >&2
  echo "  Start it first:  bash scripts/start_vllm.sh" >&2
  echo "  Continuing — LISA will start but LLM calls will fail until vLLM is up." >&2
fi

echo "==> Ingesting KB into data/faiss.index (idempotent)"
"$PY" scripts/ingest_kb.py

echo "==> Starting LISA on http://0.0.0.0:8000"
echo "    Once ready, verify with:  python scripts/verify_e2e.py"
exec "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
