"""Put this framework-tests dir on sys.path so its test modules can `import fixes_common`
(shared helpers) by bare name. The repo-root conftest already adds the project root for
`import src.*` / `import scripts.*`. Framework tests live here under `src/`; each model's
tests live with the model (`models/<model>/.../tests/`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
