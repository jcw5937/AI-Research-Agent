import nodes

FAKE_PAPER = {
    "title": "Fake Paper Title",
    "url": "https://arxiv.org/abs/9999.99999",
    "authors": "A. Author",
    "abstract": "A fake abstract for testing.",
    "published": "2024-01-01T00:00:00",
}


def _base_state():
    return {
        "goal": "test goal",
        "plan": [
            {
                "layer_name": "Graph-Based Workflow and State Machine Theory",
                "rationale": "Foundational theory for state machines in workflow graphs.",
                "search_type": "academic",
                "arxiv_categories": ["cs.LG"],
            }
        ],
        "papers": {},
        "error_log": [],
    }


def _raise_llm_down(*args, **kwargs):
    raise RuntimeError("llm down")


def test_relevance_check_exception_keeps_paper_and_logs(monkeypatch):
    monkeypatch.setattr(
        nodes, "arxiv_search", lambda query, categories=None, max_results=5: [FAKE_PAPER]
    )
    monkeypatch.setattr(nodes, "_is_relevant_paper", _raise_llm_down)

    state = _base_state()
    layer_name = state["plan"][0]["layer_name"]

    result = nodes.discover_papers_node(state)

    titles = [p["title"] for p in result["papers"][layer_name]]
    assert FAKE_PAPER["title"] in titles

    expected = (
        f"discover_papers_node: relevance check failed for '{FAKE_PAPER['title']}' "
        "(keeping paper): llm down"
    )
    assert expected in result["error_log"]


def test_relevance_check_false_drops_paper_and_logs(monkeypatch):
    monkeypatch.setattr(
        nodes, "arxiv_search", lambda query, categories=None, max_results=5: [FAKE_PAPER]
    )
    monkeypatch.setattr(nodes, "_is_relevant_paper", lambda *a, **k: False)

    state = _base_state()
    layer_name = state["plan"][0]["layer_name"]

    result = nodes.discover_papers_node(state)

    titles = [p["title"] for p in result["papers"][layer_name]]
    assert FAKE_PAPER["title"] not in titles

    expected = (
        "discover_papers_node: filtered out as not relevant to "
        f"layer '{layer_name}': '{FAKE_PAPER['title']}'"
    )
    assert expected in result["error_log"]


def test_relevance_check_true_keeps_paper_no_filter_log(monkeypatch):
    monkeypatch.setattr(
        nodes, "arxiv_search", lambda query, categories=None, max_results=5: [FAKE_PAPER]
    )
    monkeypatch.setattr(nodes, "_is_relevant_paper", lambda *a, **k: True)

    state = _base_state()
    layer_name = state["plan"][0]["layer_name"]

    result = nodes.discover_papers_node(state)

    titles = [p["title"] for p in result["papers"][layer_name]]
    assert FAKE_PAPER["title"] in titles
    assert not any("filtered out as not relevant" in e for e in result["error_log"])
    assert not any("relevance check failed" in e for e in result["error_log"])
