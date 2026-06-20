# visualization/__init__.py
"""
Only theme.py is in active use — it is imported directly by all pages.
This package init is intentionally minimal; the lazy-import pattern used
elsewhere in the codebase is unnecessary here since theme.py has no heavy
dependencies (pure constants and dict builders, no plotly/matplotlib import
side effects worth deferring).
"""
