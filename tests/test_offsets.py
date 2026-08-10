"""The offset contract: a sentence's span must reproduce it in the source HTML."""

from __future__ import annotations

import re

from inpage.highlight import Span, apply_highlights, build_chunks, merge_spans
from inpage.segment import segment_html

SAMPLE = """<!DOCTYPE html>
<html><head><title>t</title><style>p{color:red}</style></head>
<body>
<nav>Home About Contact Us Today</nav>
<h1>Speculative decoding explained</h1>
<p>The model runs one token at a time, which is slow for long outputs.
A smaller model can guess several tokens ahead of the large one.</p>
<h2>Verification</h2>
<p>The larger model then checks all guesses in parallel and keeps the valid prefix.</p>
<script>var x = "This should never be indexed at all.";</script>
</body></html>
"""


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def test_spans_round_trip_to_source_html():
    _, sentences = segment_html(SAMPLE)
    assert sentences
    for sentence in sentences:
        assert _normalise(SAMPLE[sentence.start : sentence.end]) == sentence.text


def test_script_style_and_nav_are_skipped():
    text, _ = segment_html(SAMPLE)
    assert "never be indexed" not in text
    assert "color:red" not in text
    assert "About Contact" not in text


def test_heading_is_not_swallowed_between_two_hits():
    _, sentences = segment_html(SAMPLE)
    before = next(s for s in sentences if "guess several tokens" in s.text)
    after = next(s for s in sentences if "checks all guesses" in s.text)
    merged = merge_spans(
        [Span(before.start, before.end, 1), Span(after.start, after.end, 1)], SAMPLE
    )
    assert len(merged) == 2, "an <h2> between two hits must break the merge"

    highlighted = apply_highlights(SAMPLE, merged)
    assert "<h2>Verification</h2>" in highlighted


def test_adjacent_sentences_merge_into_one_block():
    _, sentences = segment_html(SAMPLE)
    first = next(s for s in sentences if "one token at a time" in s.text)
    second = next(s for s in sentences if "guess several tokens" in s.text)
    merged = merge_spans(
        [Span(first.start, first.end, 2), Span(second.start, second.end, 1)], SAMPLE
    )
    assert len(merged) == 1, "sentences in one paragraph should form a single block"
    assert merged[0].rank == 1, "a merged span keeps the best rank"


def test_highlight_preserves_document_text():
    _, sentences = segment_html(SAMPLE)
    target = sentences[0]
    out = apply_highlights(SAMPLE, [Span(target.start, target.end, 1)])
    assert out.count("<mark") == 1
    assert out.count("</mark>") == 1
    assert _normalise(re.sub(r"</?mark[^>]*>", "", out)) == _normalise(SAMPLE)


def test_chunking_covers_every_sentence_once():
    _, sentences = segment_html(SAMPLE)
    for granularity in (1, 2, 3):
        chunks = build_chunks(sentences, granularity)
        seen = [m for c in chunks for m in c.members]
        assert seen == list(range(len(sentences)))
