import nodes


def _patch_call_llm(monkeypatch, response=None, raises=None):
    def fake(system_prompt, user_prompt):
        if raises is not None:
            raise raises
        return response

    monkeypatch.setattr(nodes, "_call_llm", fake)


class TestGeneratePlanNode:
    def test_import_mode_skips_entirely(self):
        result = nodes.generate_plan_node({"import_mode": True})
        assert result == {}

    def test_success_normalizes_layers(self, monkeypatch):
        raw = (
            '[{"layer_name": "Foundations", "rationale": "why", "search_type": "academic", '
            '"arxiv_categories": ["cs.LG", "  cs.AI "]}]'
        )
        _patch_call_llm(monkeypatch, response=raw)

        result = nodes.generate_plan_node({"goal": "test goal", "error_log": []})

        assert result["current_stage"] == "planning"
        assert result["error_log"] == []
        assert result["plan"] == [{
            "layer_name": "Foundations",
            "rationale": "why",
            "search_type": "academic",
            "arxiv_categories": ["cs.LG", "cs.AI"],
        }]

    def test_malformed_json_falls_back_and_logs(self, monkeypatch):
        _patch_call_llm(monkeypatch, response="not json at all")

        result = nodes.generate_plan_node({"goal": "test goal", "error_log": ["prior error"]})

        assert result["error_log"][0] == "prior error"
        assert any("generate_plan_node: failed to parse plan JSON" in e for e in result["error_log"])

        assert len(result["plan"]) == 1
        fallback = result["plan"][0]
        assert fallback["layer_name"] == "General Background"
        assert fallback["search_type"] == "both"
        # Regression check: fallback layer must go through _normalize_layer, so it
        # has the same keys as every other layer (previously missing arxiv_categories).
        assert fallback["arxiv_categories"] == []
        assert set(fallback.keys()) == {"layer_name", "rationale", "search_type", "arxiv_categories"}

    def test_call_llm_exception_falls_back_and_logs(self, monkeypatch):
        _patch_call_llm(monkeypatch, raises=RuntimeError("api down"))

        result = nodes.generate_plan_node({"goal": "test goal", "error_log": []})

        assert any("api down" in e for e in result["error_log"])
        assert result["plan"][0]["arxiv_categories"] == []


class TestRevisePlanNode:
    def test_success_updates_plan_and_change_log(self, monkeypatch):
        raw = (
            '{"plan": [{"layer_name": "Updated Layer", "rationale": "r", "search_type": "web", '
            '"arxiv_categories": []}], "change_log": ["renamed layer per feedback"]}'
        )
        _patch_call_llm(monkeypatch, response=raw)

        state = {
            "plan": [{"layer_name": "Old Layer", "rationale": "r", "search_type": "web", "arxiv_categories": []}],
            "feedback": "please rename it",
            "plan_change_log": ["earlier change"],
            "error_log": [],
        }
        result = nodes.revise_plan_node(state)

        assert result["plan"] == [{
            "layer_name": "Updated Layer",
            "rationale": "r",
            "search_type": "web",
            "arxiv_categories": [],
        }]
        assert result["plan_change_log"] == ["earlier change", "renamed layer per feedback"]
        assert result["error_log"] == []
        assert result["current_stage"] == "planning"

    def test_malformed_json_leaves_plan_unchanged_and_logs(self, monkeypatch):
        _patch_call_llm(monkeypatch, response="not json")

        original_plan = [{"layer_name": "Keep Me", "rationale": "r", "search_type": "both", "arxiv_categories": []}]
        state = {
            "plan": original_plan,
            "feedback": "feedback text",
            "plan_change_log": [],
            "error_log": [],
        }
        result = nodes.revise_plan_node(state)

        assert result["plan"] == original_plan
        assert any("Revision failed to parse; plan left unchanged" in c for c in result["plan_change_log"])
        assert any("revise_plan_node: failed to parse revision JSON" in e for e in result["error_log"])
