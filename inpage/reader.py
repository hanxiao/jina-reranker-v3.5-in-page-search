"""Fetch a page as HTML through the Jina Reader API."""

from __future__ import annotations

import json
import os
import urllib.request

READER_ENDPOINT = "https://r.jina.ai/"


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
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        READER_ENDPOINT,
        data=json.dumps({"url": url}).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    html = payload.get("data", {}).get("html")
    if not html:
        raise RuntimeError(f"Reader returned no HTML for {url}: {payload}")
    return html
