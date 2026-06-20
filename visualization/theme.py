# visualization/theme.py
"""
Centralised design system — ETIL.

Rules:
  - All colours defined here, imported by all pages
  - No fixed font colours → inherit from Streamlit theme (dark + light safe)
  - Legend background transparent → no white box in dark mode
  - Gridlines use rgba neutral → visible in both modes
  - Backgrounds always rgba(0,0,0,0) → Streamlit controls the page colour
"""

# ── Brand palette ─────────────────────────────────────────────────────────────
NAVY        = "#1f3a6e"
NAVY_LIGHT  = "#2e5090"
NAVY_PALE   = "#c8d4e8"
RED_DARK    = "#8b1a1a"
RED_MED     = "#c0392b"
GREEN_DARK  = "#1a4731"
GREEN_MED   = "#27ae60"
AMBER       = "#b7770d"
SLATE       = "#4a4a4a"
WHITE       = "#ffffff"
BORDER      = "rgba(128,128,128,0.2)"

# ── Colour sequences ──────────────────────────────────────────────────────────
SEQ_MAP = [
    [0.0,  "#f8f9fb"],
    [0.15, "#c8d4e8"],
    [0.35, "#8aa2c8"],
    [0.60, "#4d6ea8"],
    [0.80, "#2e5090"],
    [1.0,  "#1f3a6e"],
]

QUINTIL_COLORS = ["#c0392b", "#e67e22", "#f1c40f", "#2ecc71", "#1f3a6e"]

DIFF_COLORS = {
    "Bass":     NAVY,
    "Logistic": "#2e7d32",
    "Gompertz": AMBER,
}


def base_layout(height: int = 340, title: str = "") -> dict:
    """
    Base Plotly layout — dark/light mode safe.
    No fixed colours on text or backgrounds.
    """
    return dict(
        title=dict(text=title, font=dict(size=13), x=0),
        height=height,
        margin=dict(l=0, r=8, t=36 if title else 8, b=0),
        font=dict(family="Inter, Arial, sans-serif", size=11),
        xaxis=dict(
            gridcolor="rgba(128,128,128,0.12)",
            gridwidth=0.5,
            linecolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.2)",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.12)",
            gridwidth=0.5,
            linecolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.2)",
            tickfont=dict(size=10),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(128,128,128,0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
        hoverlabel=dict(
            bordercolor="rgba(128,128,128,0.3)",
            font=dict(size=11),
        ),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="rgba(128,128,128,0.6)",
            activecolor="rgba(31,58,110,0.9)",
        ),
    )


def bar_color_signed(values) -> list:
    """Navy for positive values, dark red for negative."""
    return [NAVY if v >= 0 else RED_DARK for v in values]


def chart_config() -> dict:
    """
    Plotly `config=` dict for st.plotly_chart — makes the modebar
    (camera/zoom/pan toolbar) transparent instead of the default solid
    grey box, which otherwise sits on top of the legend and chart title
    and is visible in both light and dark mode.

    Usage:
        st.plotly_chart(fig, use_container_width=True, config=chart_config())
    """
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d", "lasso2d", "autoScale2d",
            "hoverClosestCartesian", "hoverCompareCartesian",
            "toggleSpikelines",
        ],
        "displayModeBar": "hover",  # only appears on hover, doesn't sit static over the legend
    }


def modebar_layout_patch() -> dict:
    """
    Layout-level modebar styling — makes the toolbar background and
    button icons transparent/neutral so no grey box appears even while
    visible on hover, in both light and dark Streamlit themes.

    Merge into any layout dict, e.g.:
        fig.update_layout(**base_layout(300), **modebar_layout_patch())
    """
    return {
        "modebar": dict(
            bgcolor="rgba(0,0,0,0)",
            color="rgba(128,128,128,0.6)",
            activecolor="rgba(31,58,110,0.9)",
        ),
    }
