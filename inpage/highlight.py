"""Turn ranked chunks into <mark> spans in the original HTML."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from .segment import Sentence


@dataclass
class Span:
    """A highlight range in source-HTML coordinates, with its rank."""

    start: int
    end: int
    rank: int


@dataclass(frozen=True)
class Chunk:
    """A unit sent to the reranker: consecutive sentences joined together."""

    text: str
    members: tuple[int, ...]


def build_chunks(sentences: List[Sentence], granularity: int) -> List[Chunk]:
    """Group sentences into consecutive chunks of ``granularity`` sentences."""
    if granularity < 1:
        raise ValueError("granularity must be >= 1")
    chunks: List[Chunk] = []
    for i in range(0, len(sentences), granularity):
        members = tuple(range(i, min(i + granularity, len(sentences))))
        if not members:
            break
        chunks.append(Chunk(" ".join(sentences[j].text for j in members), members))
    return chunks


# A run of whitespace, optionally wrapped in inline tags. Anything else between
# two spans (a heading, a table cell, a figure) must break the merge.
_JOINABLE_GAP = re.compile(
    r"^(?:\s|</?(?:b|i|em|strong|span|code|a|sup|sub|u|small|mark|wbr)\b[^>]*>)*$",
    re.I,
)


def merge_spans(spans: Iterable[Span], html: str) -> List[Span]:
    """Collapse overlapping or touching spans into continuous blocks.

    Two spans join only when the source HTML between them is blank or purely
    inline markup. That keeps a highlight from swallowing a heading that sits
    between two matching sentences, while still letting a multi-sentence answer
    render as one unbroken block. A merged span keeps the best (lowest) rank,
    which is what decides its colour.
    """
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    if len(ordered) < 2:
        return [Span(s.start, s.end, s.rank) for s in ordered]

    merged: List[Span] = [Span(ordered[0].start, ordered[0].end, ordered[0].rank)]
    for span in ordered[1:]:
        prev = merged[-1]
        if span.start <= prev.end:
            prev.end = max(prev.end, span.end)
            prev.rank = min(prev.rank, span.rank)
            continue
        gap = html[prev.end : span.start]
        if _JOINABLE_GAP.match(gap):
            prev.end = max(prev.end, span.end)
            prev.rank = min(prev.rank, span.rank)
        else:
            merged.append(Span(span.start, span.end, span.rank))
    return merged


def apply_highlights(html: str, spans: List[Span]) -> str:
    """Splice <mark> tags into the HTML at the given spans.

    Applied back to front so each insertion leaves earlier offsets valid.
    """
    out = html
    for i, span in enumerate(sorted(spans, key=lambda s: s.start, reverse=True)):
        anchor = len(spans) - 1 - i
        open_tag = f'<mark class="jina-hit jina-rank-{span.rank}" id="jina-hit-{anchor}">'
        out = out[: span.start] + open_tag + out[span.start : span.end] + "</mark>" + out[span.end :]
    return out


HIGHLIGHT_CSS = """
<style>
mark.jina-hit { padding: 1px 2px; border-radius: 2px; color: #1d1d1d; }
mark.jina-rank-1 { background: #ffd54f; }
mark.jina-rank-2 { background: #ffe082; }
mark.jina-rank-3 { background: #fff0b3; }
mark.jina-rank-4, mark.jina-rank-5 { background: #fff8e1; }
</style>
"""


def inject_assets(html: str, base_url: str | None = None) -> str:
    """Add the highlight stylesheet and a <base> so relative assets resolve."""
    head = HIGHLIGHT_CSS
    if base_url:
        head = f'<base href="{base_url}">' + head
    match = re.search(r"<head[^>]*>", html, re.I)
    if match:
        return html[: match.end()] + head + html[match.end() :]
    return head + html
