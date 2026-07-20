import sys
from pathlib import Path

RESEARCH_AGENT_DIR = Path(__file__).parent.parent / "research_agent"
if str(RESEARCH_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_AGENT_DIR))
