"""A logging, OpenAI-compatible passthrough proxy.

Why it exists
-------------
The canonical schema stores ``arguments_raw``: the *exact bytes* the model emitted
for a tool call's arguments. A harness that reports arguments as a parsed object
cannot give you that — and re-serialising a parsed dict can reorder keys, which
shifts the token sequence. The model would then be trained on a distribution it
never sees at inference.

Sitting a proxy between the agent and the inference server fixes this, and gives
three more things for free:

* the **exact rendered messages** the model saw (so training re-renders faithfully
  instead of guessing how the harness formats its prompts);
* the **exact tool schema array** that was offered (so training uses the same
  schemas as inference);
* per-request token accounting, for the cost model.

The proxy is a passthrough. It changes nothing about the traffic.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

__all__ = ["ProxyServer", "ProxyConfig"]


class ProxyConfig:
    def __init__(
        self,
        upstream: str,
        log_dir: str | Path,
        listen_host: str = "127.0.0.1",
        listen_port: int = 8010,
        log_bodies: bool = True,
    ):
        self.upstream = upstream.rstrip("/")
        self.log_dir = Path(log_dir)
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.log_bodies = log_bodies
        self._counter = 0
        self._lock = threading.Lock()


class ProxyServer:
    """Threaded logging proxy. Usage::

        proxy = ProxyServer(ProxyConfig(upstream="http://127.0.0.1:8000/v1",
                                        log_dir="data/raw/proxy_logs"))
        proxy.start()
        ...
        proxy.stop()
    """

    def __init__(self, config: ProxyConfig):
        self.config = config
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        config.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ control
    def start(self) -> "ProxyServer":
        cfg = self.config

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:  # silence stderr spam
                return

            def _proxy(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""

                with cfg._lock:
                    cfg._counter += 1
                    seq = cfg._counter

                request_path = self.path
                url = cfg.upstream + ("" if request_path.startswith("/v1") else request_path)
                if not url.endswith("/v1") and request_path.startswith("/v1"):
                    url = cfg.upstream + request_path

                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in ("host", "content-length", "connection")
                }
                headers["Content-Type"] = self.headers.get(
                    "Content-Type", "application/json"
                )
                if "authorization" in {k.lower() for k in headers}:
                    pass  # forwarded as-is
                headers.setdefault("Authorization", "Bearer EMPTY")

                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                started = time.time()
                try:
                    with urllib.request.urlopen(req, timeout=600) as resp:
                        self._relay(resp, seq, body, started)
                except urllib.error.HTTPError as exc:
                    payload = exc.read()
                    self._write_logs(seq, body, payload, started, error=f"http {exc.code}")
                    self.send_response(exc.code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as exc:  # noqa: BLE001
                    self._write_logs(seq, body, b"", started, error=repr(exc))
                    payload = json.dumps({"error": {"message": repr(exc)}}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def _relay(self, resp: Any, seq: int, request_body: bytes, started: float) -> None:
                chunks: list[bytes] = []
                self.send_response(resp.status)
                ctype = resp.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", ctype)
                for key in ("cache-control", "x-opencode-session"):
                    if resp.headers.get(key):
                        self.send_header(key, resp.headers[key])
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                finally:
                    self._write_logs(seq, request_body, b"".join(chunks), started)

            def _write_logs(
                self,
                seq: int,
                request_body: bytes,
                response_body: bytes,
                started: float,
                error: str | None = None,
            ) -> None:
                if not cfg.log_bodies:
                    return
                base = cfg.log_dir / f"{seq:06d}"
                try:
                    base.with_suffix(".request.json").write_bytes(request_body)
                    base.with_suffix(".response.txt").write_bytes(response_body)
                    meta = {
                        "seq": seq,
                        "duration_s": round(time.time() - started, 3),
                        "request_bytes": len(request_body),
                        "response_bytes": len(response_body),
                        "error": error,
                    }
                    base.with_suffix(".meta.json").write_text(
                        json.dumps(meta, indent=2), encoding="utf-8"
                    )
                except OSError:
                    pass

            def do_POST(self) -> None:  # noqa: N802
                self._proxy()

            def do_GET(self) -> None:  # noqa: N802
                url = self.config.upstream + self.path
                try:
                    with urllib.request.urlopen(url, timeout=60) as resp:
                        payload = resp.read()
                        self.send_response(resp.status)
                        self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                except Exception:  # noqa: BLE001
                    self.send_response(502)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        self._httpd = ThreadingHTTPServer((cfg.listen_host, cfg.listen_port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "ProxyServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


# --------------------------------------------------------------------------- reader
def read_proxy_log(log_dir: str | Path) -> list[dict[str, Any]]:
    """Read a proxy log directory back into structured entries."""
    from .normalizer import find_tool_call_blocks

    d = Path(log_dir)
    entries: list[dict[str, Any]] = []
    if not d.exists():
        return entries
    for meta_path in sorted(d.glob("*.meta.json")):
        seq = int(meta_path.stem.split(".")[0])
        req_path = d / f"{seq:06d}.request.json"
        resp_path = d / f"{seq:06d}.response.txt"
        entry: dict[str, Any] = {"seq": seq}
        if req_path.exists():
            try:
                entry["request"] = json.loads(req_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                entry["request"] = None
        if resp_path.exists():
            text = resp_path.read_text(encoding="utf-8", errors="replace")
            entry["tool_calls"] = find_tool_call_blocks(text)
            entry["raw"] = text
        entries.append(entry)
    return entries
