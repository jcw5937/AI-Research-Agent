"""Agent node functions for the LangGraph research agent."""

import json
import re

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI  # noqa: F401 (kept available for the swap below)
from langchain_core.messages import HumanMessage, SystemMessage

import ui
from project import DEFAULT_NOTE_SECTIONS
from tools import arxiv_search, fetch_paper_text, semantic_scholar_lookup, tavily_search, write_docx

# --- single model switch point -------------------------------------------------
model = ChatAnthropic(model="claude-sonnet-4-6")
# model = ChatOpenAI(model="gpt-4o")  # swap to OpenAI by uncommenting this line
# ---------------------------------------------------------------------------

VALID_SEARCH_TYPES = {"web", "academic", "both"}


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_json_response(text: str):
    """Code-fence-tolerant JSON parser for LLM responses."""
    cleaned = _strip_code_fences(text)
    return json.loads(cleaned)


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return response.content


def _normalize_arxiv_categories(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(cat).strip() for cat in raw if isinstance(cat, str) and str(cat).strip()]


def _normalize_layer(layer: dict) -> dict:
    search_type = str(layer.get("search_type", "both")).strip().lower()
    if search_type not in VALID_SEARCH_TYPES:
        search_type = "both"
    return {
        "layer_name": layer.get("layer_name", "Untitled Layer"),
        "rationale": layer.get("rationale", ""),
        "search_type": search_type,
        "arxiv_categories": _normalize_arxiv_categories(layer.get("arxiv_categories")),
    }


# --- 1. input_node --------------------------------------------------------------

def input_node(state: dict) -> dict:
    ui.show_welcome()

    mode_choice = input(
        "Are you providing your own research plan/papers, or starting from a topic? [own/topic]: "
    ).strip().lower()
    import_mode = mode_choice.startswith("o")

    goal = input("\nWhat is your research goal? ").strip()

    if not import_mode:
        return {"goal": goal, "import_mode": False, "current_stage": "planning"}

    print("\nEnter your research plan as layer names, one per line.")
    print("Press Enter on a blank line when done.")
    layer_names = []
    while True:
        line = input("  layer name: ").strip()
        if not line:
            break
        layer_names.append(line)

    plan = [
        {
            "layer_name": name,
            "rationale": "User-provided layer.",
            "search_type": "both",
            "arxiv_categories": [],
        }
        for name in layer_names
    ]

    return {
        "goal": goal,
        "import_mode": True,
        "plan": plan,
        "current_stage": "planning",
    }


# --- 2. generate_plan_node -------------------------------------------------------

def generate_plan_node(state: dict) -> dict:
    if state.get("import_mode"):
        return {}

    system_prompt = (
        "You are a research planning assistant. Given a research goal, produce a structured "
        "layered research plan. Each layer should represent a distinct knowledge area required "
        "to achieve the goal. For each layer specify: a name, a 2-3 sentence rationale explaining "
        "why this layer is necessary, a search_type (web, academic, or both) based on whether "
        "the topic is practical/applied (web), theoretically grounded (academic), or both, and "
        "arxiv_categories: 1-3 valid arXiv category codes (e.g. cs.AI, cs.CL, cs.LG, cs.CV, "
        "physics.optics, q-bio.NC, math.OC, econ.EM, stat.ML) that best scope THIS layer's actual "
        "topic. Pick categories based on the layer's real subject matter, not any assumed domain "
        "for the overall goal. If no category meaningfully scopes the layer, return an empty list "
        "for arxiv_categories so the search stays unscoped."
    )
    user_prompt = (
        f"Goal: {state.get('goal', '')}\n\n"
        "Produce a research plan as a JSON array of layers. Each layer must be an object with "
        "keys: layer_name (string), rationale (string), search_type (\"web\", \"academic\", or "
        "\"both\"), arxiv_categories (array of 0-3 arXiv category code strings). Return ONLY the "
        "JSON array, no prose, no code fences."
    )

    error_log = list(state.get("error_log", []))
    try:
        raw = _call_llm(system_prompt, user_prompt)
        parsed = parse_json_response(raw)
        plan = [_normalize_layer(layer) for layer in parsed]
    except Exception as exc:
        error_log.append(f"generate_plan_node: failed to parse plan JSON: {exc}")
        plan = [_normalize_layer({
            "layer_name": "General Background",
            "rationale": "Fallback layer created because the planning LLM call failed to parse.",
            "search_type": "both",
        })]

    ui.show_plan(plan)
    return {"plan": plan, "current_stage": "planning", "error_log": error_log}


# --- 3. plan_review_node ---------------------------------------------------------

def plan_review_node(state: dict) -> dict:
    while True:
        ui.show_plan_menu()
        choice = input("Choice: ").strip().lower()

        if choice == "f":
            feedback = input("Enter your feedback on the plan: ").strip()
            return {"feedback": feedback, "current_stage": "plan_feedback"}
        if choice == "a":
            return {"current_stage": "paper_discovery"}
        if choice == "g":
            return {"current_stage": "researching"}
        if choice == "v":
            ui.show_full_plan(state.get("plan", []))
            continue

        print("Invalid choice. Please enter f, v, a, or g.")


# --- 4. revise_plan_node ---------------------------------------------------------

def revise_plan_node(state: dict) -> dict:
    system_prompt = (
        "You are a research planning assistant. Revise the given research plan based on user "
        "feedback. Return the revised plan as JSON. Also return a change_log array of strings "
        "explaining what you changed and why. Preserve each layer's arxiv_categories where still "
        "appropriate, or regenerate them (1-3 valid arXiv category codes scoped to that layer's "
        "actual topic, or an empty list if no category fits) if the layer's topic changed."
    )
    user_prompt = (
        f"Current plan:\n{json.dumps(state.get('plan', []), indent=2)}\n\n"
        f"User feedback:\n{state.get('feedback', '')}\n\n"
        "Return ONLY a JSON object of the form: "
        '{"plan": [{"layer_name": str, "rationale": str, "search_type": "web"|"academic"|"both", '
        '"arxiv_categories": [str, ...]}], '
        '"change_log": [str, ...]}. No prose, no code fences.'
    )

    error_log = list(state.get("error_log", []))
    change_log = list(state.get("plan_change_log", []))
    plan = state.get("plan", [])

    try:
        raw = _call_llm(system_prompt, user_prompt)
        parsed = parse_json_response(raw)
        plan = [_normalize_layer(layer) for layer in parsed.get("plan", [])]
        change_log = change_log + list(parsed.get("change_log", []))
    except Exception as exc:
        error_log.append(f"revise_plan_node: failed to parse revision JSON: {exc}")
        change_log = change_log + [f"Revision failed to parse; plan left unchanged ({exc})."]

    ui.show_plan_changes(change_log)
    ui.show_plan(plan)

    return {
        "plan": plan,
        "plan_change_log": change_log,
        "current_stage": "planning",
        "error_log": error_log,
    }


# --- 5. discover_papers_node ------------------------------------------------------

def _paper_from_web(item: dict, layer_name: str) -> dict:
    return {
        "title": item.get("title", "Untitled"),
        "url": item.get("url", ""),
        "authors": "",
        "venue": "Web",
        "verified": True,
        "peer_reviewed": False,
        "relevance": f"Found via web search for layer '{layer_name}'.",
    }


def _is_relevant_paper(title: str, abstract: str, layer_name: str, rationale: str) -> bool:
    """Ask the LLM whether a candidate paper is genuinely on-topic for a layer.

    Category filtering only excludes wrong subject fields; this catches papers that
    share vocabulary with the layer but are actually about something else. Lets
    exceptions from the LLM call propagate so callers can fail open.
    """
    system_prompt = (
        "You are a research relevance filter. Given a research layer's topic and rationale, "
        "decide whether a candidate paper is genuinely relevant to that specific topic, as "
        "opposed to merely sharing keywords or belonging to the same broad subject field."
    )
    user_prompt = (
        f"Layer: {layer_name}\n"
        f"Rationale: {rationale}\n\n"
        f"Candidate paper title: {title}\n"
        f"Candidate paper abstract: {abstract}\n\n"
        "Is this paper genuinely relevant to the layer's topic? Return ONLY a JSON object of the "
        'form {"relevant": true} or {"relevant": false}. No prose, no code fences.'
    )
    raw = _call_llm(system_prompt, user_prompt)
    try:
        parsed = parse_json_response(raw)
        if isinstance(parsed, dict) and "relevant" in parsed:
            return bool(parsed["relevant"])
    except Exception:
        pass
    return _strip_code_fences(raw).strip().lower().startswith(("true", "yes"))


def _paper_from_academic(item: dict, layer_name: str, error_log: list) -> dict:
    title = item.get("title", "Untitled")
    authors = item.get("authors", "")

    lookup = semantic_scholar_lookup(title, authors)
    if lookup.get("error"):
        error_log.append(f"discover_papers_node: {lookup['error']}")

    verified = bool(lookup.get("exists"))
    venue = lookup.get("venue") or "arXiv"
    peer_reviewed = bool(lookup.get("peer_reviewed"))

    return {
        "title": title,
        "url": item.get("url", ""),
        "authors": authors,
        "venue": venue,
        "verified": verified,
        "peer_reviewed": peer_reviewed,
        "relevance": f"Found via academic search for layer '{layer_name}'.",
    }


def discover_papers_node(state: dict) -> dict:
    goal = state.get("goal", "")
    papers = dict(state.get("papers", {}))
    error_log = list(state.get("error_log", []))

    for layer in state.get("plan", []):
        layer_name = layer.get("layer_name", "Untitled Layer")
        search_type = layer.get("search_type", "both")
        query = f"{layer_name} {goal}"

        found = []

        if search_type in ("web", "both"):
            web_results = tavily_search(query)
            for item in web_results:
                if item.get("error"):
                    error_log.append(f"discover_papers_node: {item['error']}")
                    continue
                found.append(_paper_from_web(item, layer_name))

        if search_type in ("academic", "both"):
            categories = layer.get("arxiv_categories", [])
            academic_results = arxiv_search(query, categories=categories)
            for item in academic_results:
                if item.get("error"):
                    error_log.append(f"discover_papers_node: {item['error']}")
                    continue

                title = item.get("title", "Untitled")
                try:
                    relevant = _is_relevant_paper(
                        title, item.get("abstract", ""), layer_name, layer.get("rationale", "")
                    )
                except Exception as exc:
                    error_log.append(
                        f"discover_papers_node: relevance check failed for '{title}' "
                        f"(keeping paper): {exc}"
                    )
                    relevant = True

                if not relevant:
                    error_log.append(
                        f"discover_papers_node: filtered out as not relevant to "
                        f"layer '{layer_name}': '{title}'"
                    )
                    continue

                found.append(_paper_from_academic(item, layer_name, error_log))

        papers[layer_name] = found
        ui.show_discovery_progress(layer_name, len(found))

    ui.show_papers(papers)
    return {"papers": papers, "current_stage": "paper_review", "error_log": error_log}


# --- 6. paper_review_node ---------------------------------------------------------

def _remove_paper_interactive(papers: dict) -> dict:
    """Prompt for a layer then a paper index within it and drop that paper directly,
    without going through revise_papers_node's LLM-based feedback parsing."""
    layer_names = list(papers.keys())
    if not layer_names:
        print("No papers to remove.")
        return papers

    print("\nLayers:")
    for i, name in enumerate(layer_names, start=1):
        print(f"  [{i}] {name} ({len(papers[name])} paper(s))")

    layer_choice = input("Layer number (blank to cancel): ").strip()
    if not layer_choice:
        return papers
    if not layer_choice.isdigit() or not (1 <= int(layer_choice) <= len(layer_names)):
        print("Invalid layer number.")
        return papers

    layer_name = layer_names[int(layer_choice) - 1]
    layer_papers = papers[layer_name]
    if not layer_papers:
        print(f"No papers in layer '{layer_name}'.")
        return papers

    print(f"\nPapers in '{layer_name}':")
    for i, paper in enumerate(layer_papers, start=1):
        print(f"  [{i}] {paper.get('title', 'Untitled')}")

    paper_choice = input("Paper number to remove (blank to cancel): ").strip()
    if not paper_choice:
        return papers
    if not paper_choice.isdigit() or not (1 <= int(paper_choice) <= len(layer_papers)):
        print("Invalid paper number.")
        return papers

    removed = layer_papers.pop(int(paper_choice) - 1)
    print(f"Removed: {removed.get('title', 'Untitled')}")
    return papers


def paper_review_node(state: dict) -> dict:
    papers = dict(state.get("papers", {}))

    while True:
        ui.show_paper_menu()
        choice = input("Choice: ").strip().lower()

        if choice == "f":
            feedback = input("Enter your feedback on the papers: ").strip()
            return {"papers": papers, "feedback": feedback, "current_stage": "paper_feedback"}
        if choice == "a":
            return {"papers": papers, "current_stage": "researching"}
        if choice == "v":
            ui.show_full_papers(papers)
            continue
        if choice == "r":
            papers = _remove_paper_interactive(papers)
            continue

        print("Invalid choice. Please enter f, v, r, or a.")


# --- 7. revise_papers_node ---------------------------------------------------------

def revise_papers_node(state: dict) -> dict:
    system_prompt = (
        "You are a research assistant. Revise the paper list based on user feedback. You may "
        "add, remove, or modify papers. For added papers, set verified=false until checked. "
        "Return revised papers as JSON and a change_log array."
    )
    user_prompt = (
        f"Current papers by layer:\n{json.dumps(state.get('papers', {}), indent=2)}\n\n"
        f"User feedback:\n{state.get('feedback', '')}\n\n"
        "Return ONLY a JSON object of the form: "
        '{"papers": {"<layer_name>": [{"title": str, "url": str, "authors": str, "venue": str, '
        '"verified": bool, "peer_reviewed": bool, "relevance": str}, ...]}, '
        '"change_log": [str, ...]}. Keep the same layer names as keys. No prose, no code fences.'
    )

    error_log = list(state.get("error_log", []))
    change_log = list(state.get("paper_change_log", []))
    papers = state.get("papers", {})

    try:
        raw = _call_llm(system_prompt, user_prompt)
        parsed = parse_json_response(raw)
        papers = parsed.get("papers", papers)
        change_log = change_log + list(parsed.get("change_log", []))
    except Exception as exc:
        error_log.append(f"revise_papers_node: failed to parse revision JSON: {exc}")
        change_log = change_log + [f"Revision failed to parse; papers left unchanged ({exc})."]

    ui.show_paper_changes(change_log)
    ui.show_papers(papers)

    return {
        "papers": papers,
        "paper_change_log": change_log,
        "current_stage": "paper_review",
        "error_log": error_log,
    }


# --- 8. generate_notes_node -------------------------------------------------------

def _parse_notes_sections(raw: str, note_sections: list[str]) -> dict:
    parsed = parse_json_response(raw)
    return {section: str(parsed.get(section, "")).strip() for section in note_sections}


def generate_notes_node(state: dict) -> dict:
    goal = state.get("goal", "")
    notes = dict(state.get("notes", {}))
    error_log = list(state.get("error_log", []))
    note_sections = state.get("note_sections") or DEFAULT_NOTE_SECTIONS

    system_prompt = (
        "You are a research assistant helping an AI engineer. Extract structured notes focused "
        f"on what is useful for the following goal: {goal}. Ignore content not relevant to this goal."
    )

    section_list = ", ".join(note_sections)
    section_keys = ", ".join(f'"{s}"' for s in note_sections)

    for layer_name, paper_list in state.get("papers", {}).items():
        layer_notes = []

        for paper in paper_list:
            title = paper.get("title", "Untitled")
            url = paper.get("url", "")

            text = fetch_paper_text(url)
            if text.startswith("ERROR:"):
                error_log.append(f"generate_notes_node: {text}")
                continue

            user_prompt = (
                f"Paper title: {title}\n\nPaper text:\n{text}\n\n"
                f"Produce notes in exactly these {len(note_sections)} sections: {section_list}.\n\n"
                f"Return ONLY a JSON object whose keys are exactly: {section_keys}. "
                "Each value is a string of notes for that section. No prose, no code fences."
            )

            try:
                raw = _call_llm(system_prompt, user_prompt)
                sections = _parse_notes_sections(raw, note_sections)
            except Exception as exc:
                error_log.append(f"generate_notes_node: failed to parse notes for '{title}': {exc}")
                continue

            layer_notes.append({
                "title": title,
                "url": url,
                "authors": paper.get("authors", ""),
                "venue": paper.get("venue", ""),
                "verified": paper.get("verified", False),
                "peer_reviewed": paper.get("peer_reviewed", False),
                "sections": sections,
            })
            ui.show_notes_progress(layer_name, title)

        notes[layer_name] = layer_notes

    return {"notes": notes, "error_log": error_log}


# --- 9. write_documents_node -------------------------------------------------------

def write_documents_node(state: dict) -> dict:
    output_dir = state.get("output_dir") or "research_agent/output/"
    note_sections = state.get("note_sections") or DEFAULT_NOTE_SECTIONS
    error_log = list(state.get("error_log", []))

    for layer_name, papers_notes in state.get("notes", {}).items():
        if not papers_notes:
            continue
        path = write_docx(layer_name, papers_notes, output_dir, note_sections)
        if path.startswith("ERROR:"):
            error_log.append(f"write_documents_node: {path}")
            continue
        ui.show_output_path(layer_name, path)

    ui.show_complete(output_dir, error_log)
    return {"current_stage": "done", "error_log": error_log}
