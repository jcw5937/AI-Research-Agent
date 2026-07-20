"""Terminal display functions for the research agent. Stdlib only."""

WIDTH = 78


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _print_rule(char: str = "-") -> None:
    print(_rule(char))


def show_welcome() -> None:
    print()
    print("=" * WIDTH)
    print("  RESEARCH AGENT".center(WIDTH))
    print("  Layered research planning, paper discovery, and structured notes".center(WIDTH))
    print("=" * WIDTH)
    print()


def show_plan(plan: list[dict]) -> None:
    print()
    _print_rule("=")
    print("RESEARCH PLAN")
    _print_rule("=")
    for i, layer in enumerate(plan, start=1):
        categories = layer.get("arxiv_categories") or []
        cat_display = ", ".join(categories) if categories else "unscoped"
        print(f"\n[{i}] {layer.get('layer_name', 'Untitled Layer')}  (search: {layer.get('search_type', '?')})")
        print(f"    arxiv categories: {cat_display}")
        print(f"    {layer.get('rationale', '')}")
    _print_rule("=")
    print()


def show_full_plan(plan: list[dict]) -> None:
    print()
    _print_rule("=")
    print("FULL RESEARCH PLAN (all layers)")
    _print_rule("=")
    show_plan(plan)


def show_plan_changes(change_log: list[str]) -> None:
    print()
    _print_rule("-")
    print("PLAN REVISIONS")
    _print_rule("-")
    if not change_log:
        print("  (no changes recorded)")
    for change in change_log:
        print(f"  * {change}")
    _print_rule("-")
    print()


def show_plan_menu() -> None:
    print()
    _print_rule("-")
    print("PLAN REVIEW")
    print("  [f] Give feedback")
    print("  [v] View full plan")
    print("  [a] Approve plan -> continue to paper discovery")
    print("  [g] Go -> skip straight to research")
    _print_rule("-")


def show_discovery_progress(layer_name: str, count: int) -> None:
    print(f"  -> [{layer_name}] found {count} paper(s)")


def show_papers(papers: dict) -> None:
    print()
    _print_rule("=")
    print("DISCOVERED PAPERS")
    _print_rule("=")
    for layer_name, paper_list in papers.items():
        print(f"\n{layer_name}")
        if not paper_list:
            print("  (no papers found)")
            continue
        for i, paper in enumerate(paper_list, start=1):
            verified = "verified" if paper.get("verified") else "unverified"
            peer = "peer-reviewed" if paper.get("peer_reviewed") else "not peer-reviewed"
            print(f"  [{i}] {paper.get('title', 'Untitled')}")
            print(f"      {paper.get('venue', 'Unknown venue')} | {verified} | {peer}")
    _print_rule("=")
    print()


def show_full_papers(papers: dict) -> None:
    print()
    _print_rule("=")
    print("FULL PAPER LIST")
    _print_rule("=")
    for layer_name, paper_list in papers.items():
        print(f"\n{layer_name}")
        if not paper_list:
            print("  (no papers found)")
            continue
        for i, paper in enumerate(paper_list, start=1):
            verified = "verified" if paper.get("verified") else "unverified"
            peer = "peer-reviewed" if paper.get("peer_reviewed") else "not peer-reviewed"
            print(f"  [{i}] {paper.get('title', 'Untitled')}")
            print(f"      authors: {paper.get('authors', 'Unknown')}")
            print(f"      venue:   {paper.get('venue', 'Unknown venue')} | {verified} | {peer}")
            print(f"      url:     {paper.get('url', '')}")
            print(f"      relevance: {paper.get('relevance', '(none provided)')}")
    _print_rule("=")
    print()


def show_paper_changes(change_log: list[str]) -> None:
    print()
    _print_rule("-")
    print("PAPER LIST REVISIONS")
    _print_rule("-")
    if not change_log:
        print("  (no changes recorded)")
    for change in change_log:
        print(f"  * {change}")
    _print_rule("-")
    print()


def show_paper_menu() -> None:
    print()
    _print_rule("-")
    print("PAPER REVIEW")
    print("  [f] Give feedback on papers")
    print("  [v] View full paper list")
    print("  [a] Approve papers -> begin research")
    _print_rule("-")


def show_notes_progress(layer_name: str, title: str) -> None:
    print(f"  -> [{layer_name}] notes generated for: {title}")


def show_output_path(layer_name: str, path: str) -> None:
    print(f"  -> [{layer_name}] document written: {path}")


def show_complete(output_dir: str, error_log: list[str]) -> None:
    print()
    _print_rule("=")
    print("RESEARCH COMPLETE")
    _print_rule("=")
    print(f"  Output directory: {output_dir}")
    if error_log:
        print(f"\n  {len(error_log)} error(s) occurred:")
        for err in error_log:
            print(f"    * {err}")
    else:
        print("\n  No errors encountered.")
    _print_rule("=")
    print()
