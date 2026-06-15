"""RDMCA models — the training scenarios, OUTSIDE the framework (`src/` is framework only).

Each subpackage is one self-contained model: its stage plugins, any model-specific
faculties (e.g. cognition's `mood/`), its `uses/` consumers (chat, agent, …) and its
`experiments/`. The framework discovers a model by name through the registry
(`src.plugins`, driven by `cfg["model_name"]`); deleting a model's folder removes it
whole without touching the framework.
"""
