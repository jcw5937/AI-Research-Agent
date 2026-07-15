"""Tool functions used by the research agent's nodes.

Every tool catches its own exceptions. On failure, list-returning tools
return a single-item list with an "error" key, dict-returning tools return
a dict with an "error" key (plus safe defaults for the rest of the schema),
and string-returning tools return a string starting with "ERROR:". Nodes
are responsible for detecting these and logging to error_log.
"""

import re

import arxiv
import httpx
from bs4 import BeautifulSoup

from writer import build_layer_document

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(v\d+)?|[a-z-]+/\d{7}(v\d+)?)", re.IGNORECASE)
HTTP_TIMEOUT = 15.0
USER_AGENT = "research-agent/1.0"


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily. Returns [{'title', 'url', 'snippet'}, ...]."""
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        search = TavilySearchResults(max_results=max_results)
        raw_results = search.invoke({"query": query})

        results = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })
        return results
    except Exception as exc:
        return [{"error": f"tavily_search failed: {exc}"}]


def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv. Returns [{'title', 'url', 'authors', 'abstract', 'published'}, ...]."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for result in client.results(search):
            authors = ", ".join(a.name for a in result.authors)
            results.append({
                "title": result.title,
                "url": result.entry_id,
                "authors": authors,
                "abstract": result.summary,
                "published": result.published.isoformat() if result.published else "",
            })
        return results
    except Exception as exc:
        return [{"error": f"arxiv_search failed: {exc}"}]


def semantic_scholar_lookup(title: str, authors: str) -> dict:
    """Verify a paper via Semantic Scholar. Returns {'exists', 'peer_reviewed', 'venue', 'year'}."""
    default = {"exists": False, "peer_reviewed": False, "venue": "", "year": None}
    try:
        resp = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "fields": "title,venue,year,externalIds", "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        papers = data.get("data", [])
        if not papers:
            return default

        paper = papers[0]
        venue = (paper.get("venue") or "").strip()
        peer_reviewed = bool(venue) and venue.lower() != "arxiv"

        return {
            "exists": True,
            "peer_reviewed": peer_reviewed,
            "venue": venue,
            "year": paper.get("year"),
        }
    except Exception as exc:
        return {**default, "error": f"semantic_scholar_lookup failed: {exc}"}


def _extract_arxiv_id(url: str) -> str | None:
    match = ARXIV_ID_RE.search(url)
    return match.group(1) if match else None


def _crude_pdf_text_extract(pdf_bytes: bytes) -> str:
    """Best-effort text extraction from raw PDF bytes without a PDF library.

    Pulls text drawn via Tj/TJ operators out of PDF content streams. This is
    lossy (misses compressed streams) but degrades gracefully when it can't
    find anything usable.
    """
    try:
        raw = pdf_bytes.decode("latin-1", errors="ignore")
        chunks = re.findall(r"\(((?:[^()\\]|\\.)*)\)\s*T[jJ]", raw)
        text = " ".join(chunks)
        text = text.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


def _fetch_arxiv_text(url: str, max_words: int) -> str:
    arxiv_id = _extract_arxiv_id(url)
    if not arxiv_id:
        raise ValueError(f"could not extract arxiv id from url: {url}")

    abs_resp = httpx.get(
        f"https://arxiv.org/abs/{arxiv_id}",
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
    )
    abs_resp.raise_for_status()
    soup = BeautifulSoup(abs_resp.text, "html.parser")

    title_tag = soup.find("h1", class_="title")
    title = title_tag.get_text(strip=True).replace("Title:", "").strip() if title_tag else ""

    authors_tag = soup.find("div", class_="authors")
    authors = authors_tag.get_text(strip=True).replace("Authors:", "").strip() if authors_tag else ""

    abstract_tag = soup.find("blockquote", class_="abstract")
    abstract = abstract_tag.get_text(strip=True).replace("Abstract:", "").strip() if abstract_tag else ""

    body_text = ""
    try:
        pdf_resp = httpx.get(
            f"https://arxiv.org/pdf/{arxiv_id}",
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )
        pdf_resp.raise_for_status()
        body_text = _crude_pdf_text_extract(pdf_resp.content)
    except Exception:
        body_text = ""

    combined = f"Title: {title}\nAuthors: {authors}\nAbstract: {abstract}\n\n{body_text}".strip()
    words = combined.split()
    return " ".join(words[:max_words])


def _fetch_generic_text(url: str, max_words: int) -> str:
    resp = httpx.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    return " ".join(words[:max_words])


def fetch_paper_text(url: str, max_words: int = 6000) -> str:
    """Fetch and extract text content for a paper URL, truncated to max_words."""
    try:
        if "arxiv.org" in url:
            return _fetch_arxiv_text(url, max_words)
        return _fetch_generic_text(url, max_words)
    except Exception as exc:
        return f"ERROR: fetch_paper_text failed for {url}: {exc}"


def write_docx(layer_name: str, papers_notes: list[dict], output_dir: str) -> str:
    """Write a Word document of notes for one layer. Returns the output path."""
    try:
        return build_layer_document(layer_name, papers_notes, output_dir)
    except Exception as exc:
        return f"ERROR: write_docx failed for {layer_name}: {exc}"
