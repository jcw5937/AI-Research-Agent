"""Live test: calls the real model to check arXiv category inference across domains.

Run with: pytest tests/live/test_live_category_inference.py -v -s
Requires ANTHROPIC_API_KEY. Costs real API calls. Not run by default.
"""

import os
import re

import pytest

import nodes

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="requires a real ANTHROPIC_API_KEY to call the live model",
    ),
]

# Loosely matches arXiv category codes: "cs.AI", "math.OC", "physics.optics",
# "q-bio.NC", or older archive-only names like "hep-th".
ARXIV_CATEGORY_RE = re.compile(r"^[a-zA-Z-]+(\.[a-zA-Z]+)?$")

GOALS = [
    "recent techniques for tool-use and function-calling in large language model agents",
    "gut microbiome diversity and its relationship to metabolic disease",
    "monetary policy transmission mechanisms and interest rate pass-through",
]


@pytest.mark.parametrize("goal", GOALS)
def test_category_inference_prints_layers(goal):
    state = {"goal": goal, "import_mode": False}
    result = nodes.generate_plan_node(state)
    plan = result["plan"]

    assert plan, "expected at least one layer in the plan"

    print(f"\n--- goal: {goal!r} ---")
    for layer in plan:
        categories = layer.get("arxiv_categories", [])
        print(f"  layer: {layer['layer_name']!r} -> arxiv_categories: {categories}")
        for cat in categories:
            assert ARXIV_CATEGORY_RE.match(cat), f"suspicious category code: {cat!r}"

    non_empty = sum(1 for layer in plan if layer.get("arxiv_categories"))
    assert non_empty >= max(1, len(plan) // 2), (
        f"expected most layers to get non-empty arxiv_categories, got {non_empty}/{len(plan)}"
    )
