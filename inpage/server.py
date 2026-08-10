"""A small local web UI for in-page QA search.

Deliberately stdlib-only: the point of the repo is the reranker, not a web
framework. One HTML file, one JSON endpoint, one model held in memory across
requests so you pay the load cost once.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from .reader import fetch_page
from .search import InPageSearcher

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 1 << 20


class _State:
    """Holds the model and caches the last page fetched.

    Re-asking a question about the same URL should not re-fetch it, which is
    what makes changing granularity or top-n feel instant in the UI.
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self._searcher: Optional[InPageSearcher] = None
        self._lock = threading.Lock()
        self._cache_key: Optional[str] = None
        self._cache_html: Optional[str] = None

    @property
    def searcher(self) -> InPageSearcher:
        # Loading ~1.1 GB of weights on first use keeps startup instant.
        if self._searcher is None:
            print("Loading jina-reranker-v3.5-mlx ...", flush=True)
            self._searcher = InPageSearcher(self.model_dir)
            print("Model ready.", flush=True)
        return self._searcher

    def html_for(self, url: str) -> tuple[str, str]:
        """Return (html, note). Cached, so re-ranking never re-fetches."""
        if self._cache_key == url and self._cache_html:
            return self._cache_html, "cached"
        page = fetch_page(url)
        note = page.note or ("via Reader" if page.source == "reader" else "fetched directly")
        self._cache_key, self._cache_html = url, page.html
        return page.html, note

    @property
    def model_loaded(self) -> bool:
        return self._searcher is not None


class _Handler(BaseHTTPRequestHandler):
    server_version = "inpage/0.1"

    def __init__(self, *args, state: _State, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    # -- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(
                200,
                (STATIC_DIR / "index.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/search":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(400, {"error": "bad request body"})
            return

        payload = json.loads(self.rfile.read(length))

        # Server-sent events, so the UI can show real stages rather than a
        # spinner that means nothing. Fetching a large page and loading 1.1 GB
        # of weights both take long enough to be worth reporting.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: str, **data) -> None:
            body = json.dumps(data)
            self.wfile.write(f"event: {event}\ndata: {body}\n\n".encode())
            self.wfile.flush()

        try:
            url = (payload.get("url") or "").strip()
            question = (payload.get("question") or "").strip()
            if not url or not question:
                raise ValueError("url and question are both required")

            emit("stage", step="fetch", message="fetching page")
            html, note = self.state.html_for(url)
            emit("stage", step="fetched", message=note, bytes=len(html))

            # Notes only carry what the step label does not already say.
            if not self.state.model_loaded:
                emit("stage", step="model", message="first run, ~1.1 GB")
            searcher = self.state.searcher

            emit("stage", step="rank")
            with self.state._lock:  # one forward pass at a time
                result = searcher.search(
                    html,
                    question,
                    granularity=int(payload.get("granularity", 2)),
                    top_n=int(payload.get("top_n", 1)),
                    base_url=url,
                )

            emit(
                "result",
                html=result.html,
                sentences=result.sentence_count,
                chunks=result.chunk_count,
                elapsed_ms=result.elapsed_ms,
                source=note,
                hits=[
                    {"rank": h.rank, "score": round(h.score, 4), "text": h.text}
                    for h in result.hits
                ],
            )
        except Exception as exc:  # surfaced in the UI
            try:
                emit("failed", error=str(exc))
            except Exception:
                pass

    def log_message(self, fmt, *args):  # quieter console
        return


def serve(model_dir: str, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    state = _State(model_dir)
    handler = partial(_Handler, state=state)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"In-page QA search UI on {url}  (ctrl-c to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .cli import DEFAULT_MODEL_DIR

    parser = argparse.ArgumentParser(
        prog="inpage-serve", description="Local web UI for in-page QA search."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)

    serve(args.model_dir, args.host, args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
