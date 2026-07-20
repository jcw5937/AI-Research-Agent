"""Live test: adversarial repro of the vocabulary-overlap noise bug.

"Graph-Based Workflow and State Machine Theory" (cs.LG) previously pulled in
off-topic cs.LG papers (federated learning, biology ML validation) that share
ML vocabulary but aren't about state machine / workflow theory. This test
runs real arxiv_search + real _is_relevant_paper against the live model and
prints what got kept vs. filtered for manual review.

Run with: pytest tests/live/test_live_relevance_adversarial.py -v -s
Requires ANTHROPIC_API_KEY. Costs real API calls and hits real arXiv. Not run by default.
"""

import os

import pytest

import nodes

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="requires a real ANTHROPIC_API_KEY to call the live model",
    ),
]


def test_relevance_filter_against_known_noisy_layer():
    layer_name = "Graph-Based Workflow and State Machine Theory"
    state = {
        "goal": "learn more about langgraph",
        "plan": [
            {
                "layer_name": layer_name,
                "rationale": (
                    "LangGraph models agent control flow as a state machine over a "
                    "graph; understanding state machine theory clarifies how nodes, "
                    "edges, and conditional transitions compose into a workflow."
                ),
                "search_type": "academic",
                "arxiv_categories": ["cs.LG"],
            }
        ],
        "papers": {},
        "error_log": [],
    }

    result = nodes.discover_papers_node(state)

    assert isinstance(result["error_log"], list)

    kept = result["papers"].get(layer_name, [])
    print(f"\n--- kept papers for layer {layer_name!r} ---")
    for paper in kept:
        print(f"  KEPT: {paper['title']}")

    filtered_entries = [e for e in result["error_log"] if "filtered out as not relevant" in e]
    print(f"\n--- filtered-out entries ({len(filtered_entries)}) ---")
    for entry in filtered_entries:
        print(f"  {entry}")

    other_errors = [e for e in result["error_log"] if "filtered out as not relevant" not in e]
    if other_errors:
        print(f"\n--- other error_log entries ({len(other_errors)}) ---")
        for entry in other_errors:
            print(f"  {entry}")
