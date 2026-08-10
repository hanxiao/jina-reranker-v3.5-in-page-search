"""Command line entry point: ask a page a question, get highlighted HTML."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .reader import fetch_html
from .search import DEFAULT_MODEL_REPO, InPageSearcher

DEFAULT_MODEL_DIR = "~/models/jina-reranker-v3.5-mlx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inpage",
        description="Highlight the answer to a question inside a web page, "
        "using one listwise jina-reranker-v3.5 call.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="page to fetch through the Jina Reader API")
    source.add_argument("--file", help="local HTML file to read instead")
    parser.add_argument("-q", "--question", required=True, help="question in plain language")
    parser.add_argument(
        "-g", "--granularity", type=int, default=2,
        help="sentences per chunk sent to the reranker (default: 2)",
    )
    parser.add_argument(
        "-n", "--top-n", type=int, default=1,
        help="how many ranked chunks to highlight (default: 1)",
    )
    parser.add_argument(
        "-o", "--output", default="highlighted.html", help="where to write the result",
    )
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help=f"local copy of {DEFAULT_MODEL_REPO} (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument("--open", action="store_true", help="open the result in a browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.url:
        print(f"Reading {args.url}", file=sys.stderr)
        html = fetch_html(args.url)
    else:
        html = Path(args.file).expanduser().read_text(encoding="utf-8")

    searcher = InPageSearcher(args.model_dir)
    result = searcher.search(
        html,
        args.question,
        granularity=args.granularity,
        top_n=args.top_n,
        base_url=args.url,
    )

    print(
        f"{result.sentence_count} sentences -> {result.chunk_count} chunks, "
        f"{result.elapsed_ms} ms in a single request",
        file=sys.stderr,
    )
    for hit in result.hits:
        preview = hit.text if len(hit.text) <= 160 else hit.text[:157] + "..."
        print(f"  [{hit.rank}] {hit.score:.4f}  {preview}", file=sys.stderr)

    output = Path(args.output).expanduser()
    output.write_text(result.html, encoding="utf-8")
    print(f"Wrote {output}", file=sys.stderr)

    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
