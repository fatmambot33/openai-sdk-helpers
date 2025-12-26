"""Fixtures for pytest."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local source takes precedence over any installed version
ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))
