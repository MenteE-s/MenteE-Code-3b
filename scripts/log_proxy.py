"""Logging proxy for OpenAI-compatible inference endpoints.

Captures every user↔model conversation to JSONL for later conversion to
training data. Sits in front of any OpenAI-compatible backend (Ollama,
vLLM, Modal, etc.).

Usage:
    python scripts/log_proxy.py --backend http://localhost:11434 --port 8080

Then point OpenCode at http://localhost:8080/v1 instead of the backend.
Conversations are saved to data/collected/conversations.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://localhost:11434", help="OpenAI-compatible backend URL")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--output", default="data/collected/conversations.jsonl")
    ap.add_argument("--push-hf", default=None, help="HuggingFace dataset repo to push to (optional)")
    args = ap.parse_args()

    try:
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse
        import uvicorn
    except ImportError:
        print("  ! install: pip install fastapi uvicorn httpx")
        raise SystemExit(1)

    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="MenteE Log Proxy")

    @app.post("/v1/chat/completions")
    async def proxy_chat(request: Request):
        body = await request.json()
        session_id = request.headers.get("X-Session-Id", str(uuid.uuid4()))
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Forward to backend
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{args.backend.rstrip('/')}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            )

        response_data = resp.json()

        # Log the conversation
        record = {
            "session_id": session_id,
            "started_at": started,
            "model": body.get("model", "unknown"),
            "messages": body.get("messages", []),
            "response": response_data,
            "usage": response_data.get("usage", {}),
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return response_data

    @app.get("/v1/models")
    async def proxy_models():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{args.backend.rstrip('/')}/v1/models")
            return resp.json()

    @app.get("/health")
    async def health():
        return {"status": "ok", "logged_to": str(out_path)}

    print(f"  proxy listening on http://0.0.0.0:{args.port}")
    print(f"  backend: {args.backend}")
    print(f"  logging to: {out_path}")
    if args.push_hf:
        print(f"  will push to HF: {args.push_hf}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())