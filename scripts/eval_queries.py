#!/usr/bin/env python3
"""
Smoke test: POST a fixed set of questions to a running local API (default: http://127.0.0.1:8000).

    export PYTHONPATH=.
    # Start API: uvicorn app.main:app --port 8000
    python scripts/eval_queries.py
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

import httpx


def run_queries(
    base_url: str, queries: list[str]
) -> None:
    sid = f"eval-{uuid.uuid4().hex[:8]}"
    with httpx.Client(base_url=base_url, timeout=300.0) as client:
        for q in queries:
            payload = {"session_id": sid, "message": q}
            r = client.post("/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            print("Q:", q)
            print("A:", data.get("response", ""))
            print("Sources:", data.get("sources", []))
            print("---")
            time.sleep(0.2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API root (no trailing /chat).",
    )
    args = p.parse_args()
    # Full URL to server root for httpx - httpx needs base with scheme+host+port, path in request
    base = args.base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=30.0) as c:
        h = c.get("/health")
        h.raise_for_status()
        print("Health:", json.dumps(h.json(), indent=2))
    questions: list[str] = [
        "What is term life insurance?",
        "Who is eligible for life insurance in general?",
        "How do I file a life insurance claim?",
        "Compare term life versus whole life in simple terms.",
    ]
    run_queries(base, questions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
