"""Word document generation for research notes."""

import os

from docx import Document
from docx.shared import Pt

NOTE_SECTIONS = [
    "Key Claims",
    "Tools",
    "Techniques",
    "Limitations",
    "Relevance to Cloud-Agnostic Agents",
    "Ideas",
    "Questions Raised",
]


def build_layer_document(layer_name: str, papers_notes: list[dict], output_dir: str) -> str:
    """Build a single Word document for a layer's papers and return the output path."""
    os.makedirs(output_dir, exist_ok=True)

    doc = Document()
    doc.add_heading(layer_name, level=0)

    for paper in papers_notes:
        title = paper.get("title", "Untitled Paper")
        authors = paper.get("authors", "Unknown authors")
        venue = paper.get("venue", "Unknown venue")
        sections = paper.get("sections", {})

        doc.add_heading(title, level=1)

        meta = doc.add_paragraph()
        meta_run = meta.add_run(f"{authors} — {venue}")
        meta_run.italic = True
        meta_run.font.size = Pt(10)

        for section_name in NOTE_SECTIONS:
            doc.add_heading(section_name, level=2)
            body_text = sections.get(section_name, "").strip() or "No information extracted."
            doc.add_paragraph(body_text)

    safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in layer_name).strip()
    safe_name = safe_name.replace(" ", "_") or "layer"
    file_path = os.path.join(output_dir, f"{safe_name}.docx")
    doc.save(file_path)
    return file_path
