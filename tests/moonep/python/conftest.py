from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "integrations" / "moonep_torch"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

