"""Main LangGraph graph definition and entry point for the research agent."""

import os

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

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

THREAD_ID = "research-agent-session-1"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


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


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    graph = build_graph()
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 200}
        app.invoke({}, config=config)


if __name__ == "__main__":
    run()
