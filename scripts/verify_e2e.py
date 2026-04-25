#!/usr/bin/env python3
"""
End-to-end verification for LISA (submission / ops checklist).

What it checks
  1) Optional: vLLM (or any OpenAI-compatible) HTTP server — GET {base}/models
  2) LISA API GET /health — status, index_ready, llm_model, etc.
  3) LISA API POST /chat — JSON shape and HTTP 200

Run in order (three terminals is typical for real hardware):
  Terminal A: vLLM on :8001 (see README)
  Terminal B: export PYTHONPATH=. && python scripts/ingest_kb.py && uvicorn app.main:app --port 8000
  Terminal C:  export PYTHONPATH=. && python scripts/verify_e2e.py

  export PYTHONPATH=.
  python scripts/verify_e2e.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

import httpx


def _models_url(llm_base: str) -> str:
    b = (llm_base or "").rstrip("/")
    if b.endswith("/v1"):
        return f"{b}/models"
    return f"{b}/v1/models"


def check_llm_server(client: httpx.Client, llm_base: str, api_key: str = "") -> bool:
    """Return True if the OpenAI-compatible server responds to GET /v1/models."""
    url = _models_url(llm_base)
    headers = {}
    if api_key and api_key not in ("", "none"):
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = client.get(url, timeout=5.0, headers=headers)
    except httpx.RequestError as e:
        print(f"FAIL: LLM server not reachable at {url!r}: {e}", flush=True)
        return False
    if r.status_code == 401 and not api_key:
        # Server requires auth — try without insisting; consider it reachable
        print(f"OK:  LLM server {url!r} is up (401 — requires API key, pass --llm-api-key)", flush=True)
        return True
    if r.status_code != 200:
        print(
            f"FAIL: LLM server {url!r} returned {r.status_code}: {r.text[:200]}",
            flush=True,
        )
        return False
    try:
        data = r.json()
    except json.JSONDecodeError:
        print("FAIL: LLM /models is not valid JSON", flush=True)
        return False
    if "data" in data and isinstance(data["data"], list):
        n = len(data["data"])
        print(f"OK:  LLM server {url!r} — {n} model(s) listed")
    else:
        print(f"OK:  LLM server {url!r} responded 200 (payload keys: {list(data)[:5]!r})")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="LISA API base URL (no /chat).",
    )
    p.add_argument(
        "--skip-llm-ping",
        action="store_true",
        help="Do not GET the vLLM/OpenAI-compatible /models endpoint.",
    )
    p.add_argument(
        "--llm-api-key",
        default="",
        help="API key (Bearer token) for the LLM /v1/models health check. "
             "Reads VLLM_API_KEY / LLM_API_KEY env var if not set.",
    )
    p.add_argument(
        "--ingest",
        action="store_true",
        help="If set, run `python scripts/ingest_kb.py` before API checks (same Python env, cwd).",
    )
    args = p.parse_args()
    lisa = args.base_url.rstrip("/")

    if args.ingest:
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        r = subprocess.run(
            [sys.executable, str(root / "scripts" / "ingest_kb.py")],
            cwd=root,
            check=False,
            env={**__import__("os").environ, "PYTHONPATH": str(root)},
        )
        if r.returncode != 0:
            print("FAIL: ingest_kb.py exited with non-zero status", file=sys.stderr)
            return 1
        print("OK:  ingest_kb.py completed")

    with httpx.Client(timeout=240.0) as client:
        # --- LISA health ---
        try:
            h = client.get(f"{lisa}/health")
            h.raise_for_status()
        except httpx.RequestError as e:
            print(
                f"FAIL: LISA API not reachable at {lisa!r} — start: uvicorn app.main:app --port 8000. ({e})",
                file=sys.stderr,
            )
            return 1
        except httpx.HTTPStatusError as e:
            print(f"FAIL: GET /health: {e}", file=sys.stderr)
            return 1

        health = h.json()
        print("Health:", json.dumps(health, indent=2), flush=True)
        if health.get("status") != "ok":
            print("FAIL: health.status is not 'ok'", file=sys.stderr)
            return 1
        if not health.get("index_ready"):
            err = health.get("index_error")
            hint = f" ({err})" if err else ""
            print(
                f"FAIL: index_ready is false{hint} — run: export PYTHONPATH=. && python scripts/ingest_kb.py",
                file=sys.stderr,
            )
            return 1
        if not health.get("llm_model"):
            print("FAIL: missing llm_model in /health", file=sys.stderr)
            return 1
        print("OK:  /health: index ready, model routing visible", flush=True)

        llm_base = health.get("llm_base_url", "")

        if not args.skip_llm_ping and llm_base:
            print("Step: LLM server (OpenAI-compatible GET /v1/models) ...", flush=True)
            import os
            llm_key = args.llm_api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("LLM_API_KEY") or ""
            if not check_llm_server(client, str(llm_base), api_key=llm_key):
                return 1

        # --- One chat turn ---
        sid = f"e2e-{uuid.uuid4().hex[:10]}"
        q = "What is term life insurance in one short sentence?"
        r = client.post(
            f"{lisa}/chat",
            json={"session_id": sid, "message": q},
        )
        if r.status_code == 503:
            print("FAIL: POST /chat returned 503 (pipeline error) — see API logs", file=sys.stderr)
            print(r.text, file=sys.stderr)
            return 1
        r.raise_for_status()
        data = r.json()
        for key in ("response", "sources", "query_type", "low_confidence"):
            if key not in data:
                print(f"FAIL: missing key in /chat response: {key!r}", file=sys.stderr)
                return 1
        if not isinstance(data.get("response"), str):
            print("FAIL: response is not a string", file=sys.stderr)
            return 1
        print("Q:", q)
        print("A:", (data.get("response") or "")[:500])
        print("low_confidence:", data.get("low_confidence"), "query_type:", data.get("query_type"))
        print("OK:  POST /chat completed (200)")

    print("---\nE2E checks passed. For multi-turn, use: python scripts/chat_cli.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
