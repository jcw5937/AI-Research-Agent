"""Shared state schema for the LangGraph research agent."""

from typing import TypedDict


class AgentState(TypedDict):
    goal: str
    import_mode: bool
    plan: list[dict]
    plan_change_log: list[str]
    papers: dict[str, list[dict]]
    paper_change_log: list[str]
    notes: dict[str, list[dict]]
    current_stage: str
    feedback: str
    error_log: list[str]
    # Active project's config, set once at startup by the pre-graph project
    # picker (see project.py) and read by generate_notes_node/write_documents_node.
    note_sections: list[str]
    output_dir: str
