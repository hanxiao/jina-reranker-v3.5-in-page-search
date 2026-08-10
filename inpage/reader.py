"""Fetch a page as HTML.

Two sources, tried in order. The Jina Reader API renders client-side pages and
returns clean markup, but anonymous calls are rate limited and some networks are
refused outright. A plain HTTP GET needs no key and no account, and for ordinary
server-rendered pages it is perfectly good input. So Reader is preferred when it
works and a direct fetch is the fallback, which means the tool runs with zero
configuration.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass

READER_ENDPOINT = "https://r.jina.ai/"
__version__ = "0.1.0"

# urllib's default Python-urllib/3.x agent is refused by edges and CDNs, which
# surfaces as a 403 long before the request reaches anything that would answer.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
)


@dataclass
class Page:
    html: str
    #: "reader" or "direct", so callers can report which path was taken.
    source: str
    note: str = ""


def fetch_page(url: str, api_key: str | None = None, timeout: int = 300) -> Page:
    """Return the HTML for ``url``, preferring Reader and falling back to GET."""
    api_key = api_key if api_key is not None else os.environ.get("JINA_API_KEY")

    try:
        return Page(_via_reader(url, api_key, timeout), "reader")
    except _ReaderUnavailable as exc:
        # Only fall back for auth/rate problems. A genuine 404 upstream should
        # not be retried as though it were a different kind of failure.
        html = _via_direct(url, timeout)
        return Page(html, "direct", note=str(exc))


def fetch_html(url: str, api_key: str | None = None, timeout: int = 300) -> str:
    """Backwards-compatible wrapper returning just the HTML."""
    return fetch_page(url, api_key, timeout).html


class _ReaderUnavailable(RuntimeError):
    """Reader could not serve this request, but a direct fetch might."""


def _via_reader(url: str, api_key: str | None, timeout: int) -> str:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Return-Format": "html",
        "User-Agent": f"inpage-search/{__version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        READER_ENDPOINT, data=json.dumps({"url": url}).encode(), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 402, 403, 429):
            raise _ReaderUnavailable(
                f"Reader unavailable ({exc.code}), fetched the page directly instead"
            ) from exc
        raise RuntimeError(f"Reader error {exc.code} for {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise _ReaderUnavailable(f"Reader unreachable ({exc.reason}), fetched directly") from exc

    html = (payload.get("data") or {}).get("html")
    if not html:
        raise _ReaderUnavailable("Reader returned no HTML, fetched directly")
    return html


def _via_direct(url: str, timeout: int) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            encoding = (response.headers.get("Content-Encoding") or "").lower()
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Could not fetch {url}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc.reason}") from exc

    if encoding == "gzip":
        raw = gzip.decompress(raw)
    elif encoding == "deflate":
        raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw.decode(charset, errors="replace")
