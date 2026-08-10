"""In-page QA search with a single listwise jina-reranker-v3.5 call."""

from .highlight import Span, apply_highlights, build_chunks, merge_spans
from .reader import fetch_html
from .search import DEFAULT_MODEL_REPO, Hit, InPageSearcher, SearchResult
from .segment import Sentence, segment_html

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODEL_REPO",
    "Hit",
    "InPageSearcher",
    "SearchResult",
    "Sentence",
    "Span",
    "apply_highlights",
    "build_chunks",
    "fetch_html",
    "merge_spans",
    "segment_html",
]
