"""Pytest bootstrap: put the repo root on sys.path so tests can `import src.*` and
`import scripts.*` regardless of how deep under the tree they live (top-level tests/
or a stage plugin's own tests/ dir). Auto-loaded by pytest from the rootdir."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
