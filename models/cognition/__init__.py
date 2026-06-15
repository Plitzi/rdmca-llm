"""Model: **cognition** — the conversational/agentic LLM curriculum.

Ten stages that grow a foundational cognitive model: language → perception →
abstraction → causal → reasoning → memory → ethics/BCF (freeze point), then the
behavioral LoRA sectors tools → MCP → skills. This is the default training model
(`cfg["model_name"]`, registry default). Each stage is a self-contained plugin under this
package; the registry discovers them automatically. Its experiments (hypothesis probes
like continual_learning) live in `experiments/` within this package.
"""
