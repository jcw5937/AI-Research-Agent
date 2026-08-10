"""User/project registry and per-project storage layout.

Single-user today. DEFAULT_USER_SLUG is the only hardcoded user identity in the
codebase -- every lookup here takes user_slug as a parameter, so wiring in real
multi-user auth later means changing DEFAULT_USER_SLUG's call sites, not this
module's logic.

Storage layout: data/<user_slug>/projects.json is the registry; each project
gets data/<user_slug>/<project_slug>/{checkpoints.db, output/, vectorstore/}.
vectorstore/ is reserved for a future chat-with-saved-research feature and is
otherwise unused today.
"""

import json
import os
import re
from datetime import datetime, timezone

DEFAULT_USER_SLUG = "jake"

DEFAULT_NOTE_SECTIONS = [
    "Key Claims",
    "Tools",
    "Techniques",
    "Limitations",
    "Relevance to Cloud-Agnostic Agents",
    "Ideas",
    "Questions Raised",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "project"


def user_dir(user_slug: str = DEFAULT_USER_SLUG) -> str:
    return os.path.join(DATA_DIR, user_slug)


def projects_registry_path(user_slug: str = DEFAULT_USER_SLUG) -> str:
    return os.path.join(user_dir(user_slug), "projects.json")


def project_dir(project_slug: str, user_slug: str = DEFAULT_USER_SLUG) -> str:
    return os.path.join(user_dir(user_slug), project_slug)


def checkpoints_db_path(project_slug: str, user_slug: str = DEFAULT_USER_SLUG) -> str:
    return os.path.join(project_dir(project_slug, user_slug), "checkpoints.db")


def output_dir_path(project_slug: str, user_slug: str = DEFAULT_USER_SLUG) -> str:
    return os.path.join(project_dir(project_slug, user_slug), "output")


def vectorstore_dir_path(project_slug: str, user_slug: str = DEFAULT_USER_SLUG) -> str:
    return os.path.join(project_dir(project_slug, user_slug), "vectorstore")


def thread_id_for(project_slug: str, user_slug: str = DEFAULT_USER_SLUG) -> str:
    return f"{user_slug}:{project_slug}"


def list_projects(user_slug: str = DEFAULT_USER_SLUG) -> list[dict]:
    """Return registered projects for a user, newest first. [] if none/unreadable."""
    path = projects_registry_path(user_slug)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            projects = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return sorted(projects, key=lambda p: p.get("created_at", ""), reverse=True)


def get_project(project_slug: str, user_slug: str = DEFAULT_USER_SLUG) -> dict | None:
    for project in list_projects(user_slug):
        if project.get("project_slug") == project_slug:
            return project
    return None


def ensure_project_layout(project_slug: str, user_slug: str = DEFAULT_USER_SLUG) -> None:
    os.makedirs(project_dir(project_slug, user_slug), exist_ok=True)
    os.makedirs(output_dir_path(project_slug, user_slug), exist_ok=True)
    os.makedirs(vectorstore_dir_path(project_slug, user_slug), exist_ok=True)


def register_project(
    project_slug: str,
    display_name: str,
    goal: str,
    note_sections: list[str],
    user_slug: str = DEFAULT_USER_SLUG,
    created_at: str | None = None,
) -> dict:
    """Add or update a project entry in the user's registry and ensure its on-disk layout exists."""
    os.makedirs(user_dir(user_slug), exist_ok=True)
    ensure_project_layout(project_slug, user_slug)

    projects = [p for p in list_projects(user_slug) if p.get("project_slug") != project_slug]
    entry = {
        "project_slug": project_slug,
        "display_name": display_name,
        "goal": goal,
        "note_sections": note_sections,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    projects.append(entry)

    with open(projects_registry_path(user_slug), "w") as f:
        json.dump(projects, f, indent=2)

    return entry
