"""Put the tests/ dir itself on sys.path so test modules can `import fixes_common`
(shared helpers) by bare name. The repo-root conftest already adds the project root
for `import src.*` / `import scripts.*`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
