import pytest

import nodes
import ui


@pytest.fixture(autouse=True)
def silence_ui(monkeypatch):
    """Unit tests exercise node functions directly; ui.* only prints to stdout."""
    monkeypatch.setattr(ui, "show_discovery_progress", lambda *a, **k: None)
    monkeypatch.setattr(ui, "show_papers", lambda *a, **k: None)
    monkeypatch.setattr(ui, "show_plan", lambda *a, **k: None)
    monkeypatch.setattr(ui, "show_plan_changes", lambda *a, **k: None)
    monkeypatch.setattr(ui, "show_notes_progress", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def no_network_semantic_scholar(monkeypatch):
    """Guarantee unit tests never hit the real Semantic Scholar API by default."""
    monkeypatch.setattr(
        nodes,
        "semantic_scholar_lookup",
        lambda title, authors: {
            "exists": True,
            "peer_reviewed": False,
            "venue": "arXiv",
            "year": None,
        },
    )
