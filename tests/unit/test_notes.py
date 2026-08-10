import pytest

import nodes


class TestParseNotesSections:
    def test_missing_keys_default_to_empty_string(self):
        raw = '{"Key Claims": "claim text"}'
        result = nodes._parse_notes_sections(raw, ["Key Claims", "Tools", "Ideas"])
        assert result == {"Key Claims": "claim text", "Tools": "", "Ideas": ""}

    def test_non_json_raises(self):
        with pytest.raises(Exception):
            nodes._parse_notes_sections("not json", ["Key Claims"])

    def test_extra_keys_in_response_are_ignored(self):
        raw = '{"Key Claims": "x", "Unrequested Section": "should be dropped"}'
        result = nodes._parse_notes_sections(raw, ["Key Claims"])
        assert result == {"Key Claims": "x"}


def _paper(**overrides):
    paper = {
        "title": "Test Paper",
        "url": "https://arxiv.org/abs/1234.5678",
        "authors": "Jane Doe",
        "venue": "NeurIPS",
        "verified": True,
        "peer_reviewed": True,
    }
    paper.update(overrides)
    return paper


class TestGenerateNotesNode:
    def test_success_preserves_metadata_and_uses_custom_sections(self, monkeypatch):
        monkeypatch.setattr(nodes, "fetch_paper_text", lambda url: "some fetched paper text")
        monkeypatch.setattr(
            nodes, "_call_llm", lambda s, u: '{"Claims": "x", "Notes": "y"}'
        )

        state = {
            "goal": "test goal",
            "papers": {"L1": [_paper()]},
            "notes": {},
            "error_log": [],
            "note_sections": ["Claims", "Notes"],
        }
        result = nodes.generate_notes_node(state)

        layer_notes = result["notes"]["L1"]
        assert len(layer_notes) == 1
        entry = layer_notes[0]
        assert entry["title"] == "Test Paper"
        assert entry["url"] == "https://arxiv.org/abs/1234.5678"
        assert entry["authors"] == "Jane Doe"
        assert entry["venue"] == "NeurIPS"
        assert entry["verified"] is True
        assert entry["peer_reviewed"] is True
        assert entry["sections"] == {"Claims": "x", "Notes": "y"}
        assert result["error_log"] == []

    def test_missing_note_sections_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(nodes, "fetch_paper_text", lambda url: "text")
        captured_prompts = []

        def fake_call_llm(system_prompt, user_prompt):
            captured_prompts.append(user_prompt)
            return "{}"

        monkeypatch.setattr(nodes, "_call_llm", fake_call_llm)

        state = {
            "goal": "test goal",
            "papers": {"L1": [_paper()]},
            "notes": {},
            "error_log": [],
        }
        result = nodes.generate_notes_node(state)

        sections = result["notes"]["L1"][0]["sections"]
        assert set(sections.keys()) == set(nodes.DEFAULT_NOTE_SECTIONS)
        assert "Key Claims" in captured_prompts[0]

    def test_fetch_paper_text_error_skips_paper_and_logs(self, monkeypatch):
        monkeypatch.setattr(
            nodes, "fetch_paper_text", lambda url: f"ERROR: fetch_paper_text failed for {url}: boom"
        )
        monkeypatch.setattr(nodes, "_call_llm", lambda s, u: pytest.fail("should not be called"))

        state = {
            "goal": "test goal",
            "papers": {"L1": [_paper()]},
            "notes": {},
            "error_log": [],
        }
        result = nodes.generate_notes_node(state)

        assert result["notes"]["L1"] == []
        assert any("generate_notes_node: ERROR: fetch_paper_text failed" in e for e in result["error_log"])

    def test_llm_parse_failure_skips_paper_and_logs(self, monkeypatch):
        monkeypatch.setattr(nodes, "fetch_paper_text", lambda url: "some text")
        monkeypatch.setattr(nodes, "_call_llm", lambda s, u: "not valid json")

        state = {
            "goal": "test goal",
            "papers": {"L1": [_paper(title="Broken Paper")]},
            "notes": {},
            "error_log": [],
        }
        result = nodes.generate_notes_node(state)

        assert result["notes"]["L1"] == []
        assert any(
            "generate_notes_node: failed to parse notes for 'Broken Paper'" in e for e in result["error_log"]
        )
