import nodes
import tools


class FakeTavilySearchResults:
    def __init__(self, max_results=5):
        self.max_results = max_results

    def invoke(self, payload):
        return [
            {"title": "Result A", "url": "https://a.com", "content": "snippet a"},
            "not a dict, should be skipped",
            {"title": "Result B", "url": "https://b.com", "content": "snippet b"},
        ]


class FakeTavilySearchResultsRaises:
    def __init__(self, max_results=5):
        raise RuntimeError("tavily down")


class TestTavilySearch:
    def test_success_maps_and_skips_non_dict_items(self, monkeypatch):
        monkeypatch.setattr(
            "langchain_community.tools.tavily_search.TavilySearchResults",
            FakeTavilySearchResults,
        )
        results = tools.tavily_search("some query")
        assert results == [
            {"title": "Result A", "url": "https://a.com", "snippet": "snippet a"},
            {"title": "Result B", "url": "https://b.com", "snippet": "snippet b"},
        ]

    def test_exception_returns_error_sentinel(self, monkeypatch):
        monkeypatch.setattr(
            "langchain_community.tools.tavily_search.TavilySearchResults",
            FakeTavilySearchResultsRaises,
        )
        results = tools.tavily_search("some query")
        assert results == [{"error": "tavily_search failed: tavily down"}]


class FakeResponse:
    def __init__(self, json_data, raises=None):
        self._json = json_data
        self._raises = raises

    def raise_for_status(self):
        if self._raises:
            raise self._raises

    def json(self):
        return self._json


class TestSemanticScholarLookup:
    def test_found_peer_reviewed_venue(self, monkeypatch):
        monkeypatch.setattr(
            tools.httpx, "get",
            lambda *a, **k: FakeResponse({"data": [{"venue": "NeurIPS", "year": 2023}]}),
        )
        result = tools.semantic_scholar_lookup("Some Title", "Some Author")
        assert result == {"exists": True, "peer_reviewed": True, "venue": "NeurIPS", "year": 2023}

    def test_arxiv_venue_is_not_peer_reviewed(self, monkeypatch):
        monkeypatch.setattr(
            tools.httpx, "get",
            lambda *a, **k: FakeResponse({"data": [{"venue": "arXiv", "year": 2023}]}),
        )
        result = tools.semantic_scholar_lookup("Some Title", "Some Author")
        assert result["exists"] is True
        assert result["peer_reviewed"] is False

    def test_no_results_returns_default(self, monkeypatch):
        monkeypatch.setattr(tools.httpx, "get", lambda *a, **k: FakeResponse({"data": []}))
        result = tools.semantic_scholar_lookup("Some Title", "Some Author")
        assert result == {"exists": False, "peer_reviewed": False, "venue": "", "year": None}

    def test_exception_returns_default_plus_error_key(self, monkeypatch):
        def raise_get(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(tools.httpx, "get", raise_get)
        result = tools.semantic_scholar_lookup("Some Title", "Some Author")
        assert result["exists"] is False
        assert result["error"] == "semantic_scholar_lookup failed: network down"


class TestPaperFromAcademicSemanticScholarErrorPropagation:
    def test_error_from_semantic_scholar_lookup_is_logged(self, monkeypatch):
        # Override the autouse no_network_semantic_scholar stub for this one test to
        # exercise _paper_from_academic's own error-append-to-error_log behavior.
        monkeypatch.setattr(
            nodes,
            "semantic_scholar_lookup",
            lambda title, authors: {
                "exists": False,
                "peer_reviewed": False,
                "venue": "",
                "year": None,
                "error": "semantic_scholar_lookup failed: boom",
            },
        )
        error_log = []
        paper = nodes._paper_from_academic({"title": "T", "authors": "A", "url": "u"}, "Layer", error_log)

        assert paper["verified"] is False
        assert paper["venue"] == "arXiv"
        assert any("semantic_scholar_lookup failed: boom" in e for e in error_log)
