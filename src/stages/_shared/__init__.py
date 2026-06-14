"""Helpers shared by more than one stage's data sources (readability gate,
real+synthetic blending, system personas, the hermes tool-call parser, and the
tiered dictionary bank). Kept here so no single stage "owns" cross-cutting utility
code, and so the per-stage sources.py modules stay focused on their own data."""
