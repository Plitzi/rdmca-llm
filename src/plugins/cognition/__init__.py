"""Domain: **cognition** — the conversational/agentic LLM curriculum.

Ten stages that grow a foundational cognitive model: language → perception →
abstraction → causal → reasoning → memory → ethics/BCF (freeze point), then the
behavioral LoRA sectors tools → MCP → skills. This is the default training domain
(`cfg["domain"]`, registry default). Each stage is a self-contained plugin under this
package; the registry discovers them automatically.
"""
