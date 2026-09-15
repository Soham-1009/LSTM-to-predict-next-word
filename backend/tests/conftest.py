"""Pytest configuration: ensure the `backend/` directory is on sys.path so
imports like `from app.main import app` work regardless of where pytest
is invoked from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
