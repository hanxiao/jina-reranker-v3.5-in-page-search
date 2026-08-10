"""In-page QA search: one listwise rerank call over a whole document."""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .highlight import Chunk, Span, apply_highlights, build_chunks, inject_assets, merge_spans
from .segment import Sentence, segment_html

DEFAULT_MODEL_REPO = "jinaai/jina-reranker-v3.5-mlx"


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


def _load_reranker_module(model_dir: Path):
    """Import the checkpoint's own rerank.py.

    The MLX checkpoint ships its modelling and scoring code alongside the
    weights, so we load that rather than reimplementing the LBNL readout.
    """
    module_path = model_dir / "rerank.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"{module_path} not found. Download the checkpoint first:\n"
            f"  hf download {DEFAULT_MODEL_REPO} --local-dir {model_dir}"
        )
    # The checkpoint's rerank.py imports its sibling modeling.py by name.
    sys.path.insert(0, str(model_dir))
    spec = importlib.util.spec_from_file_location("jina_rerank_v35", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InPageSearcher:
    """Rank every passage of one page against a question, in a single call."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir).expanduser().resolve()
        module = _load_reranker_module(model_dir)
        self.reranker = module.MLXReranker(model_path=str(model_dir))

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
        ranked = self.reranker.rerank(question, [c.text for c in chunks], top_n=top_n)
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
