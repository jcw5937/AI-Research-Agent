import pytest

import nodes


def _patch_call_llm(monkeypatch, response=None, raises=None):
    def fake(system_prompt, user_prompt):
        if raises is not None:
            raise raises
        return response

    monkeypatch.setattr(nodes, "_call_llm", fake)


class TestIsRelevantPaperJSON:
    def test_json_true(self, monkeypatch):
        _patch_call_llm(monkeypatch, response='{"relevant": true}')
        assert nodes._is_relevant_paper("T", "A", "layer", "rationale") is True

    def test_json_false(self, monkeypatch):
        _patch_call_llm(monkeypatch, response='{"relevant": false}')
        assert nodes._is_relevant_paper("T", "A", "layer", "rationale") is False

    def test_code_fenced_json_true(self, monkeypatch):
        _patch_call_llm(monkeypatch, response='```json\n{"relevant": true}\n```')
        assert nodes._is_relevant_paper("T", "A", "layer", "rationale") is True

    def test_code_fenced_json_false(self, monkeypatch):
        _patch_call_llm(monkeypatch, response='```json\n{"relevant": false}\n```')
        assert nodes._is_relevant_paper("T", "A", "layer", "rationale") is False


class TestIsRelevantPaperLooseFallback:
    @pytest.mark.parametrize(
        "response", ["true", "True", "yes", "Yes, this is relevant."]
    )
    def test_unparseable_but_starts_true_or_yes_returns_true(self, monkeypatch, response):
        _patch_call_llm(monkeypatch, response=response)
        assert nodes._is_relevant_paper("T", "A", "layer", "rationale") is True

    @pytest.mark.parametrize(
        "response", ["false", "no", "No, not relevant.", "unrelated free text"]
    )
    def test_non_matching_text_returns_false(self, monkeypatch, response):
        _patch_call_llm(monkeypatch, response=response)
        assert nodes._is_relevant_paper("T", "A", "layer", "rationale") is False


class TestIsRelevantPaperErrorPropagation:
    def test_call_llm_exception_propagates(self, monkeypatch):
        # _is_relevant_paper does not catch exceptions from _call_llm itself -- it's
        # discover_papers_node's job to catch and fail open, per its docstring.
        _patch_call_llm(monkeypatch, raises=RuntimeError("api down"))
        with pytest.raises(RuntimeError, match="api down"):
            nodes._is_relevant_paper("T", "A", "layer", "rationale")
