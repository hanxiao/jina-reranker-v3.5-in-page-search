"""In-page QA search with a single listwise jina-reranker-v3.5 call."""

from .backends import ApiReranker, LocalReranker
from .highlight import Span, apply_highlights, build_chunks, merge_spans
from .reader import fetch_html
from .search import DEFAULT_MODEL_REPO, Hit, InPageSearcher, SearchResult, strip_api_flag
from .segment import Sentence, segment_html
from .server import serve

__version__ = "0.1.0"

__all__ = [
    "ApiReranker",
    "DEFAULT_MODEL_REPO",
    "Hit",
    "LocalReranker",
    "strip_api_flag",
    "InPageSearcher",
    "SearchResult",
    "Sentence",
    "Span",
    "apply_highlights",
    "build_chunks",
    "fetch_html",
    "merge_spans",
    "segment_html",
    "serve",
]
