#!/usr/bin/env python3
"""
Interactive text chat with the LISA API (POST /chat), same code path and session
memory as production.

Requires the API to be running, e.g.:
  export PYTHONPATH=.
  uvicorn app.main:app --host 0.0.0.0 --port 8000

Then:
  export PYTHONPATH=.
  python scripts/chat_cli.py
  # or: python scripts/chat_cli.py --base-url http://127.0.0.1:8000

Commands:  quit, exit, q  — end session.  clear  — new session_id (new context).
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

import httpx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url",
        default=os.environ.get("LISA_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        help="API root (no trailing /chat). Default: LISA_API_URL or http://127.0.0.1:8000",
    )
    p.add_argument(
        "--session-id",
        default="",
        help="Fixed session id (default: auto-generated for this run).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds (default: 300).",
    )
    args = p.parse_args()

    session_id = (args.session_id or "").strip() or f"cli-{uuid.uuid4().hex[:12]}"
    base = args.base_url.rstrip("/")

    print("LISA — Life Insurance Support Assistant (CLI)")
    print(f"API: {base}  |  session: {session_id}")
    print("Type a message, or: quit / exit / q  |  clear = new session")
    print("---")

    with httpx.Client(base_url=base, timeout=args.timeout) as client:
        try:
            h = client.get("/health")
            h.raise_for_status()
        except httpx.RequestError as e:
            print(f"Cannot reach API: {e}", file=sys.stderr)
            print("Start the server: uvicorn app.main:app --port 8000", file=sys.stderr)
            return 1

        data = h.json()
        if not data.get("index_ready", False):
            print(
                "Warning: FAISS index not ready — run: python scripts/ingest_kb.py",
                file=sys.stderr,
            )

        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            low = line.lower()
            if low in ("quit", "exit", "q"):
                return 0
            if low == "clear":
                session_id = f"cli-{uuid.uuid4().hex[:12]}"
                print(f"(new session) {session_id}\n---")
                continue

            try:
                r = client.post(
                    "/chat",
                    json={"session_id": session_id, "message": line},
                )
                r.raise_for_status()
            except httpx.HTTPError as e:
                print(f"Error: {e}", file=sys.stderr)
                if hasattr(e, "response") and e.response is not None:
                    try:
                        print(e.response.text, file=sys.stderr)
                    except Exception:
                        pass
                continue

            out = r.json()
            text = out.get("response", "")
            sources = out.get("sources", [])
            qtype = out.get("query_type", "")
            lowc = out.get("low_confidence", False)
            print("Assistant:", text)
            if qtype or lowc or sources:
                extra = f"type={qtype}" if qtype else ""
                if lowc:
                    extra = f"{extra}, low_confidence" if extra else "low_confidence"
                if extra:
                    print(f"  ({extra})")
                if sources:
                    print("  sources:", ", ".join(sources))
            print("---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
