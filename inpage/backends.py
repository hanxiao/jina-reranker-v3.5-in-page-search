"""Two ways to run the same reranker: locally through MLX, or the hosted API.

Both expose ``rerank(query, documents, top_n) -> [{index, relevance_score}]``,
which is all the search code needs, so switching between them changes nothing
downstream.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Protocol

API_ENDPOINT = "https://api.jina.ai/v1/rerank"
API_MODEL = "jina-reranker-v3.5"
DEFAULT_MODEL_REPO = "jinaai/jina-reranker-v3.5-mlx"

# The hosted endpoint caps how much it will take in one call.
API_MAX_DOCUMENTS = 2048


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[dict]:
        ...


class LocalReranker:
    """The MLX checkpoint, running on this machine."""

    name = "local (mlx)"

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir).expanduser().resolve()
        module_path = model_dir / "rerank.py"
        if not module_path.exists():
            raise FileNotFoundError(
                f"{module_path} not found. Download the checkpoint first:\n"
                f"  hf download {DEFAULT_MODEL_REPO} --local-dir {model_dir}"
            )
        # The checkpoint ships its own modelling and scoring code next to the
        # weights, and rerank.py imports its sibling modeling.py by name.
        sys.path.insert(0, str(model_dir))
        spec = importlib.util.spec_from_file_location("jina_rerank_v35", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._impl = module.MLXReranker(model_path=str(model_dir))

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[dict]:
        return self._impl.rerank(query, documents, top_n=top_n)


class ApiReranker:
    """The hosted reranker at api.jina.ai. Needs a key."""

    name = "api.jina.ai"

    def __init__(self, api_key: str | None = None, timeout: int = 300):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "the hosted reranker needs a key: export JINA_API_KEY=***\n"
                "get one at https://jina.ai/api-dashboard/"
            )
        self.timeout = timeout

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[dict]:
        if len(documents) > API_MAX_DOCUMENTS:
            raise RuntimeError(
                f"{len(documents)} chunks exceeds the hosted limit of "
                f"{API_MAX_DOCUMENTS}; raise granularity or run locally"
            )
        request = urllib.request.Request(
            API_ENDPOINT,
            data=json.dumps(
                {
                    "model": API_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                }
            ).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Cloudflare answers urllib's default agent with a 1010
                # "access denied", which looks like an auth failure and is not.
                "User-Agent": "inpage-search/0.1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"rerank API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach {API_ENDPOINT}: {exc.reason}") from exc

        # The API nests the document as {"text": ...} where MLX returns a bare
        # string, but only index and score are read downstream.
        return payload.get("results", [])
