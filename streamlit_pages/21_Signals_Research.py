"""Consolidated signal and ranking workspace for ETIL."""

from __future__ import annotations

import importlib

import pandas as pd
import streamlit as st


_WORKSPACES = {
    "EVQTI signal validation": "streamlit_pages.06_EVQTI_Ranking",
    "Composite transition index": "streamlit_pages.10_ETII",
    "Multi-factor research": "streamlit_pages.11_Research_Intelligence",
}


def render(df: pd.DataFrame) -> None:
    """Render a single research workspace to preserve response time and analytic focus."""
    latest = int(df["year"].max())
    st.markdown("## Signals & Rankings")
    st.caption(
        f"Cross-country research signals for the {latest} historical panel. "
        "Use validation diagnostics before interpreting rankings as forward-looking evidence."
    )
    st.markdown("---")

    selection = st.radio(
        "Research workflow",
        list(_WORKSPACES),
        horizontal=True,
        label_visibility="collapsed",
    )
    module = importlib.import_module(_WORKSPACES[selection])
    module.render(df, embedded=True)
