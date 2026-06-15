# RDMCA — Relevance-Driven Modular Cognitive Architecture
"""The RDMCA FRAMEWORK (task-agnostic). Everything used to build/train a model in
general — backend, model, modalities, data, training, the continual-learning runtime
(`src/`) plus the plugin system (`src/plugins/`). The models themselves live OUTSIDE
`src/`, in top-level `models/`; the framework discovers them by name and never imports
one directly."""

__version__ = "0.1.0"
