"""Split an HTML document into sentences that remember where they came from.

The point of this module is the offset bookkeeping. Everything downstream --
chunking, reranking, highlighting -- works on a flat plain-text string, but the
final output has to be the *original* HTML with <mark> tags inserted. So every
sentence carries the byte range in the source HTML that produced it, and those
ranges are what the highlighter splices into.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List

# Containers whose text is chrome, not prose.
SKIP_TAGS = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "head",
        "nav", "header", "footer", "aside", "form", "button", "select",
        "sup", "figcaption",
        # Code listings are not prose. Left in, a snippet gets paired with the
        # sentence next to it and drags the chunk's score around.
        "pre", "code",
    }
)

# Tags that end a sentence even without punctuation, so a heading never runs
# into the paragraph beneath it.
BLOCK_TAGS = frozenset(
    {
        "p", "div", "li", "td", "th", "section", "article", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6", "br", "ul", "ol", "dl", "dd", "dt",
        "table", "tr", "figure", "main", "body", "hr",
    }
)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？;；])\s+|\n{2,}")

# Reference cruft that carries no meaning for ranking.
NOISE = re.compile(
    r"cite_ref|cite_note|\bdoi:|\bISBN\b|\bPMID\b|\bS2CID\b|Retrieved \d{4}|Archived from",
    re.I,
)

MIN_SENTENCE_CHARS = 25


@dataclass(frozen=True)
class Sentence:
    """A sentence plus the span of source HTML it was extracted from."""

    text: str
    #: Character offsets into the original HTML string.
    start: int
    end: int


class _TextSpanExtractor(HTMLParser):
    """Collect visible text and map every character back to the source HTML.

    ``convert_charrefs`` stays off so entity references arrive as their own
    events; that keeps ``getpos`` aligned with the raw markup and lets us
    record exact source offsets rather than offsets into a decoded copy.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: List[str] = []
        #: (text_offset, html_start, html_end) for each emitted run of text.
        self.spans: List[tuple[int, int, int]] = []
        self._skip_depth = 0
        self._length = 0
        self._line_starts: List[int] = []

    def feed_html(self, html: str) -> None:
        # getpos() reports (line, column), so precompute line offsets to turn
        # those back into absolute indices.
        offset = 0
        self._line_starts = [0]
        for line in html.splitlines(keepends=True):
            offset += len(line)
            self._line_starts.append(offset)
        self._html = html
        self.feed(html)
        self.close()

    def _abs_pos(self) -> int:
        line, col = self.getpos()
        return self._line_starts[line - 1] + col

    def _append(self, text: str, html_start: int, html_end: int) -> None:
        if not text:
            return
        self.parts.append(text)
        self.spans.append((self._length, html_start, html_end))
        self._length += len(text)

    def _break(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n\n"):
            self.parts.append("\n\n")
            self._length += 2

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self._break()

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in BLOCK_TAGS:
            self._break()

    def handle_data(self, data):
        if self._skip_depth:
            return
        start = self._abs_pos()
        self._append(data, start, start + len(data))

    def handle_entityref(self, name):
        if self._skip_depth:
            return
        start = self._abs_pos()
        raw = f"&{name};"
        from html import unescape

        self._append(unescape(raw), start, start + len(raw))

    def handle_charref(self, name):
        if self._skip_depth:
            return
        start = self._abs_pos()
        raw = f"&#{name};"
        from html import unescape

        self._append(unescape(raw), start, start + len(raw))

    def text_to_html_offset(self, index: int) -> int:
        """Map an offset in the flattened text back into the source HTML."""
        best = 0
        for text_start, html_start, html_end in self.spans:
            if text_start > index:
                break
            delta = index - text_start
            best = min(html_start + delta, html_end)
        return best


def segment_html(html: str) -> tuple[str, List[Sentence]]:
    """Return the flattened text and its sentences, with source HTML offsets."""
    parser = _TextSpanExtractor()
    parser.feed_html(html)
    text = "".join(parser.parts)

    sentences: List[Sentence] = []
    cursor = 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        _push(sentences, parser, text, cursor, match.start())
        cursor = match.end()
    _push(sentences, parser, text, cursor, len(text))
    return text, sentences


def _push(
    out: List[Sentence],
    parser: _TextSpanExtractor,
    text: str,
    start: int,
    end: int,
) -> None:
    raw = text[start:end]
    stripped = raw.strip()
    if len(stripped) < MIN_SENTENCE_CHARS or NOISE.search(stripped):
        return
    lead = len(raw) - len(raw.lstrip())
    text_start = start + lead
    text_end = text_start + len(stripped)
    out.append(
        Sentence(
            text=re.sub(r"\s+", " ", stripped),
            start=parser.text_to_html_offset(text_start),
            end=parser.text_to_html_offset(text_end),
        )
    )
