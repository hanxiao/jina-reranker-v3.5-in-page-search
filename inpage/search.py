"""In-page QA search: one listwise rerank call over a whole document."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .backends import DEFAULT_MODEL_REPO, ApiReranker, LocalReranker, Reranker
from .highlight import Chunk, Span, apply_highlights, build_chunks, inject_assets, merge_spans
from .segment import Sentence, segment_html

__all__ = [
    "DEFAULT_MODEL_REPO",
    "Hit",
    "InPageSearcher",
    "SearchResult",
    "strip_api_flag",
]

#: Appending this to a url switches that search to the hosted reranker.
API_FLAG = "api=true"


@dataclass
class Hit:
    """One ranked chunk, with the sentences and source offsets behind it."""

    rank: int
    score: float
    text: str
    start: int
    end: int


@dataclass
class SearchResult:
    html: str
    hits: List[Hit] = field(default_factory=list)
    sentence_count: int = 0
    chunk_count: int = 0
    elapsed_ms: int = 0
    backend: str = ""


def strip_api_flag(url: str) -> tuple[str, bool]:
    """Split ``api=true`` off a url, returning the clean url and the flag.

    The flag is a property of the request, not of the page, so it has to come
    off before fetching or the query string would change what is fetched.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    use_api = any(k == "api" and v.lower() == "true" for k, v in pairs)
    if not use_api:
        return url, False
    kept = [(k, v) for k, v in pairs if k != "api"]
    return urlunsplit(parts._replace(query=urlencode(kept))), True


class InPageSearcher:
    """Rank every passage of one page against a question, in a single call."""

    def __init__(self, model_dir: str | Path | None = None, reranker: Reranker | None = None):
        if reranker is None:
            if model_dir is None:
                raise ValueError("pass either model_dir or reranker")
            reranker = LocalReranker(model_dir)
        self.reranker = reranker

    @classmethod
    def hosted(cls, api_key: str | None = None) -> "InPageSearcher":
        """A searcher backed by api.jina.ai instead of the local weights."""
        return cls(reranker=ApiReranker(api_key))

    def search(
        self,
        html: str,
        question: str,
        granularity: int = 2,
        top_n: int = 1,
        base_url: Optional[str] = None,
    ) -> SearchResult:
        """Highlight the answer to ``question`` inside ``html``.

        Args:
            html: The source page, as returned by the Reader API or a file.
            question: A natural-language question; keywords are not required.
            granularity: Sentences per chunk sent to the reranker.
            top_n: How many ranked chunks to highlight.
            base_url: Page URL, so relative assets still resolve offline.
        """
        _, sentences = segment_html(html)
        if not sentences:
            raise ValueError("no searchable text found in this document")

        chunks = build_chunks(sentences, granularity)
        started = time.perf_counter()
        # The whole document goes in as one candidate list: this is the single
        # listwise call the demo is about.
        ranked = self.reranker.rerank(question, [c.text for c in chunks], top_n)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        spans, hits = self._to_spans(ranked, chunks, sentences)
        merged = merge_spans(spans, html)
        highlighted = inject_assets(apply_highlights(html, merged), base_url)

        return SearchResult(
            html=highlighted,
            hits=hits,
            sentence_count=len(sentences),
            chunk_count=len(chunks),
            elapsed_ms=elapsed_ms,
            backend=getattr(self.reranker, "name", ""),
        )

    @staticmethod
    def _to_spans(
        ranked: List[dict],
        chunks: List[Chunk],
        sentences: List[Sentence],
    ) -> tuple[List[Span], List[Hit]]:
        spans: List[Span] = []
        hits: List[Hit] = []
        for position, item in enumerate(ranked, start=1):
            chunk = chunks[item["index"]]
            # One span per sentence, not first.start..last.end: a chunk's
            # sentences can straddle a block boundary, and spanning that gap
            # would highlight whatever sits between them.
            for member in chunk.members:
                sentence = sentences[member]
                spans.append(Span(sentence.start, sentence.end, position))
            hits.append(
                Hit(
                    rank=position,
                    score=float(item["relevance_score"]),
                    text=chunk.text,
                    start=sentences[chunk.members[0]].start,
                    end=sentences[chunk.members[-1]].end,
                )
            )
        return spans, hits
