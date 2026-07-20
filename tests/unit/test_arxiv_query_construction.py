import pytest

import tools


@pytest.fixture
def captured_query(monkeypatch):
    """Patch arxiv.Client/arxiv.Search at the tools module level and capture the
    query string passed to arxiv.Search, without ever hitting the network."""
    captured = {}

    class FakeClient:
        def results(self, search):
            return []

    class FakeSearch:
        def __init__(self, query, max_results, sort_by):
            captured["query"] = query
            captured["max_results"] = max_results

    monkeypatch.setattr(tools.arxiv, "Client", FakeClient)
    monkeypatch.setattr(tools.arxiv, "Search", FakeSearch)
    return captured


def test_scoped_query_with_multiple_categories(captured_query):
    tools.arxiv_search("test query", categories=["cs.AI", "cs.CL"])
    assert captured_query["query"] == "(cat:cs.AI OR cat:cs.CL) AND all:test query"


def test_scoped_query_with_single_category(captured_query):
    tools.arxiv_search("test query", categories=["cs.LG"])
    assert captured_query["query"] == "(cat:cs.LG) AND all:test query"


def test_empty_categories_list_is_unscoped(captured_query):
    tools.arxiv_search("test query", categories=[])
    assert captured_query["query"] == "test query"


def test_none_categories_is_unscoped(captured_query):
    tools.arxiv_search("test query", categories=None)
    assert captured_query["query"] == "test query"


def test_default_categories_param_is_unscoped(captured_query):
    tools.arxiv_search("test query")
    assert captured_query["query"] == "test query"
