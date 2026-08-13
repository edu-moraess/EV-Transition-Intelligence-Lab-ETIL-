"""Consolidated modelling workspace for ETIL."""

from __future__ import annotations

import importlib

import pandas as pd
import streamlit as st


_WORKSPACES = {
    "Diffusion curves": "streamlit_pages.03_Diffusion_Engine",
    "Regime analysis": "streamlit_pages.04_Regime_Detection",
    "Probabilistic forecast": "streamlit_pages.05_Forecast_Lab",
    "Scenario analysis": "streamlit_pages.09_Scenarios",
}


def render(df: pd.DataFrame) -> None:
    """Render one computational workspace at a time to avoid duplicate model work."""
    latest = int(df["year"].max())
    st.markdown("## Model Lab")
    st.caption(
        f"{df.attrs.get('dataset', 'IEA')} · {df.attrs.get('vehicle_mode', 'Cars')} · "
        f"Historical coverage through {latest}. Select a model family to run its dedicated workflow."
    )
    st.markdown("---")

    selection = st.radio(
        "Analytical workflow",
        list(_WORKSPACES),
        horizontal=True,
        label_visibility="collapsed",
    )
    module = importlib.import_module(_WORKSPACES[selection])
    module.render(df, embedded=True)
