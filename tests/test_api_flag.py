"""?api=true selects the hosted reranker and must not reach the fetch."""

from __future__ import annotations

import pytest

from inpage.search import strip_api_flag


def test_absent_flag_leaves_url_untouched():
    url = "https://example.com/docs/page"
    assert strip_api_flag(url) == (url, False)


def test_flag_is_detected_and_removed():
    assert strip_api_flag("https://example.com/x?api=true") == ("https://example.com/x", True)


def test_other_query_params_survive():
    url, use_api = strip_api_flag("https://example.com/x?v=2&api=true&lang=en")
    assert use_api
    assert url == "https://example.com/x?v=2&lang=en"


def test_flag_is_case_insensitive_on_the_value():
    assert strip_api_flag("https://example.com/x?api=TRUE")[1] is True


def test_other_values_do_not_enable_it():
    url, use_api = strip_api_flag("https://example.com/x?api=false")
    assert use_api is False
    # A value we did not ask for is left alone rather than silently dropped.
    assert url.endswith("api=false")


def test_fragment_and_path_are_preserved():
    url, use_api = strip_api_flag("https://example.com/a/b?api=true#frag")
    assert use_api
    assert url == "https://example.com/a/b#frag"


@pytest.mark.parametrize(
    "url",
    [
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)?api=true",
        "https://example.com/search?q=a+b&api=true",
    ],
)
def test_awkward_urls_round_trip(url):
    cleaned, use_api = strip_api_flag(url)
    assert use_api
    assert "api=true" not in cleaned
