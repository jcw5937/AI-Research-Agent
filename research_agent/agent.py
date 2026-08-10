"""Main LangGraph graph definition and entry point for the research agent."""

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

import project
import ui
from nodes import (
    discover_papers_node,
    generate_notes_node,
    generate_plan_node,
    input_node,
    paper_review_node,
    plan_review_node,
    revise_papers_node,
    revise_plan_node,
    write_documents_node,
)
from state import AgentState


def route_after_input(state: AgentState) -> str:
    return "plan_review" if state.get("import_mode") else "generate_plan"


def route_after_plan_review(state: AgentState) -> str:
    stage = state.get("current_stage")
    if stage == "plan_feedback":
        return "revise_plan"
    if stage == "paper_discovery":
        return "discover_papers"
    if stage == "researching":
        return "generate_notes"
    return "revise_plan"


def route_after_paper_review(state: AgentState) -> str:
    stage = state.get("current_stage")
    if stage == "paper_feedback":
        return "revise_papers"
    if stage == "researching":
        return "generate_notes"
    return "revise_papers"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("input_node", input_node)
    graph.add_node("generate_plan", generate_plan_node)
    graph.add_node("plan_review", plan_review_node)
    graph.add_node("revise_plan", revise_plan_node)
    graph.add_node("discover_papers", discover_papers_node)
    graph.add_node("paper_review", paper_review_node)
    graph.add_node("revise_papers", revise_papers_node)
    graph.add_node("generate_notes", generate_notes_node)
    graph.add_node("write_documents", write_documents_node)

    graph.add_edge(START, "input_node")
    graph.add_conditional_edges(
        "input_node",
        route_after_input,
        {"generate_plan": "generate_plan", "plan_review": "plan_review"},
    )
    graph.add_edge("generate_plan", "plan_review")
    graph.add_conditional_edges(
        "plan_review",
        route_after_plan_review,
        {
            "revise_plan": "revise_plan",
            "discover_papers": "discover_papers",
            "generate_notes": "generate_notes",
        },
    )
    graph.add_edge("revise_plan", "plan_review")
    graph.add_edge("discover_papers", "paper_review")
    graph.add_conditional_edges(
        "paper_review",
        route_after_paper_review,
        {"revise_papers": "revise_papers", "generate_notes": "generate_notes"},
    )
    graph.add_edge("revise_papers", "paper_review")
    graph.add_edge("generate_notes", "write_documents")
    graph.add_edge("write_documents", END)

    return graph


# --- pre-graph project picker ----------------------------------------------------
# Project selection must happen before the graph is compiled, since it determines
# which checkpoints.db file the checkpointer opens and what thread_id to use --
# it can't be a graph node.

def _create_project_interactive(user_slug: str) -> dict:
    display_name = input("Project name: ").strip() or "Untitled Project"

    existing_slugs = {p["project_slug"] for p in project.list_projects(user_slug)}
    base_slug = project.slugify(display_name)
    project_slug = base_slug
    suffix = 2
    while project_slug in existing_slugs:
        project_slug = f"{base_slug}-{suffix}"
        suffix += 1

    sections_choice = input(
        "Note sections: [d]efault (7 standard sections) or [c]ustom? [d/c]: "
    ).strip().lower()
    if sections_choice.startswith("c"):
        print("Enter custom note section names, one per line. Press Enter on a blank line when done.")
        custom_sections = []
        while True:
            line = input("  section name: ").strip()
            if not line:
                break
            custom_sections.append(line)
        note_sections = custom_sections or list(project.DEFAULT_NOTE_SECTIONS)
    else:
        note_sections = list(project.DEFAULT_NOTE_SECTIONS)

    entry = project.register_project(
        project_slug=project_slug,
        display_name=display_name,
        goal="",
        note_sections=note_sections,
        user_slug=user_slug,
    )
    ui.show_project_created(entry)
    return entry


def pick_or_create_project(user_slug: str = project.DEFAULT_USER_SLUG) -> dict:
    projects = project.list_projects(user_slug)
    while True:
        ui.show_project_picker(projects)
        choice = input("Choice: ").strip().lower()

        if choice == "n":
            return _create_project_interactive(user_slug)
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(projects):
                return projects[idx]

        print("Invalid choice. Please enter a project number or 'n'.")


def run():
    user_slug = project.DEFAULT_USER_SLUG
    chosen = pick_or_create_project(user_slug)
    project.ensure_project_layout(chosen["project_slug"], user_slug)

    db_path = project.checkpoints_db_path(chosen["project_slug"], user_slug)
    output_dir = project.output_dir_path(chosen["project_slug"], user_slug)
    thread_id = project.thread_id_for(chosen["project_slug"], user_slug)

    graph = build_graph()
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}

        snapshot = app.get_state(config)
        has_prior_state = bool(snapshot.values)
        is_resumable = has_prior_state and bool(snapshot.next)

        try:
            if is_resumable:
                ui.show_project_resuming(chosen["display_name"])
                app.invoke(None, config=config)
            else:
                if has_prior_state:
                    ui.show_project_new_pass(chosen["display_name"])
                app.invoke(
                    {"note_sections": chosen["note_sections"], "output_dir": output_dir},
                    config=config,
                )
        finally:
            # Best-effort even on crash/Ctrl+C: persist whatever goal is now known
            # so the project picker reflects it on the next run.
            final_values = app.get_state(config).values or {}
            project.register_project(
                project_slug=chosen["project_slug"],
                display_name=chosen["display_name"],
                goal=final_values.get("goal") or chosen.get("goal", ""),
                note_sections=chosen["note_sections"],
                user_slug=user_slug,
                created_at=chosen.get("created_at"),
            )


if __name__ == "__main__":
    run()
