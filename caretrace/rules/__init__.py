"""Rules package — legacy pyDatalog entry point.

The canonical rules engine now lives at `src.rules.rules_agent` (plain
Python). This package is kept for import compatibility with anything
that still references `caretrace.rules`; add new rule modules here only
if they complement Alex's engine (e.g., medication safety rules).
"""
