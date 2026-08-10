# AI Research Agent — Project Overview

## What it is

A LangGraph-based agent that automates the early, grunt-work phase of research: given a topic, it breaks the topic into a structured plan of subtopics, searches the web and arXiv for relevant sources per subtopic, and produces a set of structured Word documents (key claims, tools, techniques, limitations, ideas, open questions) — with the person reviewing and steering the plan and the source list at two checkpoints along the way, rather than the agent running fully unsupervised.

## How the pipeline flows

The agent is built as a graph of nine steps (a LangGraph `StateGraph`), where a single shared state object carries all the accumulated data — goal, plan, papers, notes — through every step in sequence:

1. **Input** — capture the research goal (and, if the user already has their own plan, import it directly).
2. **Generate plan** — an LLM breaks the goal into layered subtopics, each with a rationale and a search strategy (web, academic, or both).
3. **Plan review (human checkpoint)** — the person reviews the plan and either approves it, sends it back for revision with feedback, or skips straight to research.
4. **Discover papers** — for each subtopic, search the web (Tavily) and arXiv, scoped to the right subject categories.
5. **Paper review (human checkpoint)** — review the discovered sources, remove ones that aren't useful, or send feedback for revision.
6. **Generate notes** — an LLM extracts structured notes per paper, per subtopic.
7. **Write documents** — one Word document per subtopic, containing all its papers' notes.

The two human checkpoints are the core design choice: the agent proposes, the person disposes, at both the planning stage and the sourcing stage, before any time is spent on the expensive step (note extraction).

## Hardening done so far

Several real bugs and design gaps have been found and fixed since the first working version:

- **Source quality** — arXiv searches were pulling in physics/math papers for AI-research topics because the query wasn't scoped to subject categories. Fixed with category scoping, plus a second LLM-based relevance filter that catches papers that share vocabulary with a topic but aren't actually about it (category scoping alone can't catch that — it only excludes wrong fields, not off-topic papers within the right field).
- **Metadata loss** — paper authors/venue/peer-review status were being silently dropped between discovery and the final document, so every source showed as "Unknown authors — Unknown venue" regardless of what was actually found.
- **Hardcoded analysis lens** — the note-taking template had a fixed section ("Relevance to Cloud-Agnostic Agents") baked in from the project's original narrower use case, forcing every paper through that lens regardless of the actual research topic. Now configurable per project.
- **No persistence** — the very first version held all state in memory only; closing the terminal lost everything. Now each project has its own file-backed checkpoint database, so work resumes exactly where it left off. This required catching a subtle LangGraph bug where naive resumption silently restarts the whole pipeline from the beginning and duplicates work — fixed with an explicit check of whether a project has pending work before deciding to resume vs. start fresh.
- **Multi-project support** — projects are now namespaced (`data/<user>/<project>/...`), so multiple research efforts can be tracked and resumed independently instead of everything overwriting one shared output folder.

## Testing

30 automated unit tests currently cover the search/filtering logic — category-scoped query construction, the relevance filter's response parsing, and the "fail open" behavior when a relevance check errors out (a paper is kept rather than silently dropped if the check itself breaks). These run fully mocked, in milliseconds, with no live API calls or cost.

A separate opt-in "live" test tier exists for real-world validation against actual arXiv/LLM responses — including an adversarial test case (a deliberately ambiguous topic that's historically produced off-topic results) to confirm the relevance filter earns its keep. This isn't run by default since it costs real API calls.

**Not yet covered:** the newer project/persistence code — the project registry, and especially the resume-vs-fresh-start logic, which was verified by hand but doesn't yet have an automated regression test. This is the next planned addition to the test suite.

## Where it's headed

The tool is currently a terminal/command-line interface. The planned direction is a visual, interactive one:

- Instead of a static text plan, the research plan becomes an actual diagram built conversationally with an LLM — click into a subtopic to edit, delete, or drill into it, without regenerating the whole plan each time.
- Paper discovery becomes visual too: click into a subtopic node to see the sources found for it specifically, with a surface/medium/deep dial controlling how many sources get pulled per topic.
- Document output changes come after those two land.
- Further out: a second user (a non-technical collaborator), and a "chat with your saved research" mode — asking questions against previously gathered research for a project, powered by a per-project search index. The storage layout already reserves space for this so it isn't a rebuild later.

## Summary for a one-line status

Core pipeline works and is hardened against several real bugs (source quality, data loss, hardcoded assumptions, lack of persistence). Now supports multiple independent, resumable research projects. 30 passing automated tests cover the search/filtering logic; test coverage for the persistence layer is the next planned addition, followed by a shift from CLI to an interactive visual interface for plan-building and source review.
