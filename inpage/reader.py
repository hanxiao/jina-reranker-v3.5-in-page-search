"""Fetch a page as HTML through the Jina Reader API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

READER_ENDPOINT = "https://r.jina.ai/"
__version__ = "0.1.0"


def fetch_html(url: str, api_key: str | None = None, timeout: int = 300) -> str:
    """Return the rendered HTML for ``url``.

    A key is optional for light use but avoids the anonymous rate limit; set
    JINA_API_KEY or pass ``api_key``. Get one at https://jina.ai/api-dashboard/.
    """
    api_key = api_key or os.environ.get("JINA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Return-Format": "html",
        # urllib's default Python-urllib/3.x agent is refused by the edge with
        # a 403 before the request ever reaches Reader, key or no key.
        "User-Agent": f"inpage-search/{__version__} (+https://github.com/hanxiao/jina-reranker-v3.5-in-page-search)",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        READER_ENDPOINT,
        data=json.dumps({"url": url}).encode(),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        # Anonymous Reader calls are rate limited and can be blocked outright
        # depending on the network you are on. A free key clears both.
        if exc.code in (401, 403) and not api_key:
            raise RuntimeError(
                f"Reader refused an anonymous request ({exc.code}). Set a free key:\n"
                "  export JINA_API_KEY=jina_...   # https://jina.ai/api-dashboard/"
            ) from exc
        raise RuntimeError(f"Reader error {exc.code} for {url}: {exc.reason}") from exc

    html = payload.get("data", {}).get("html")
    if not html:
        raise RuntimeError(f"Reader returned no HTML for {url}: {payload}")
    return html
