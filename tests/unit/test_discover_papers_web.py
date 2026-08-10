import nodes


class TestPaperFromWeb:
    def test_maps_fields_with_web_defaults(self):
        item = {"title": "A Blog Post", "url": "https://example.com/post"}
        paper = nodes._paper_from_web(item, "My Layer")

        assert paper["title"] == "A Blog Post"
        assert paper["url"] == "https://example.com/post"
        assert paper["authors"] == ""
        assert paper["venue"] == "Web"
        assert paper["verified"] is True
        assert paper["peer_reviewed"] is False
        assert "My Layer" in paper["relevance"]

    def test_missing_title_defaults_to_untitled(self):
        paper = nodes._paper_from_web({}, "My Layer")
        assert paper["title"] == "Untitled"
        assert paper["url"] == ""


class TestDiscoverPapersWebBranch:
    def test_web_search_type_uses_tavily_only(self, monkeypatch):
        monkeypatch.setattr(
            nodes,
            "tavily_search",
            lambda query, max_results=5: [{"title": "Web Result", "url": "https://example.com"}],
        )

        def fail_if_called(*args, **kwargs):
            raise AssertionError("arxiv_search should not be called for search_type='web'")

        monkeypatch.setattr(nodes, "arxiv_search", fail_if_called)

        state = {
            "goal": "test goal",
            "plan": [{"layer_name": "L1", "rationale": "r", "search_type": "web", "arxiv_categories": []}],
            "papers": {},
            "error_log": [],
        }
        result = nodes.discover_papers_node(state)

        titles = [p["title"] for p in result["papers"]["L1"]]
        assert titles == ["Web Result"]
        assert result["papers"]["L1"][0]["venue"] == "Web"

    def test_web_result_with_error_is_logged_and_skipped(self, monkeypatch):
        monkeypatch.setattr(
            nodes,
            "tavily_search",
            lambda query, max_results=5: [
                {"error": "tavily_search failed: boom"},
                {"title": "Good Result", "url": "https://example.com"},
            ],
        )

        state = {
            "goal": "test goal",
            "plan": [{"layer_name": "L1", "rationale": "r", "search_type": "web", "arxiv_categories": []}],
            "papers": {},
            "error_log": [],
        }
        result = nodes.discover_papers_node(state)

        titles = [p["title"] for p in result["papers"]["L1"]]
        assert titles == ["Good Result"]
        assert any("discover_papers_node: tavily_search failed: boom" in e for e in result["error_log"])

    def test_both_search_type_dispatches_web_and_academic(self, monkeypatch):
        monkeypatch.setattr(
            nodes,
            "tavily_search",
            lambda query, max_results=5: [{"title": "Web Result", "url": "https://example.com"}],
        )
        monkeypatch.setattr(
            nodes,
            "arxiv_search",
            lambda query, categories=None, max_results=5: [
                {"title": "Academic Result", "url": "https://arxiv.org/abs/1", "authors": "A", "abstract": "x"}
            ],
        )
        monkeypatch.setattr(nodes, "_is_relevant_paper", lambda *a, **k: True)

        state = {
            "goal": "test goal",
            "plan": [{"layer_name": "L1", "rationale": "r", "search_type": "both", "arxiv_categories": []}],
            "papers": {},
            "error_log": [],
        }
        result = nodes.discover_papers_node(state)

        titles = sorted(p["title"] for p in result["papers"]["L1"])
        assert titles == ["Academic Result", "Web Result"]
