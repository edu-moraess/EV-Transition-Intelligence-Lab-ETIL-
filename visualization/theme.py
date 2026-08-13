# visualization/theme.py
"""
Centralised design system — ETIL.

Rules:
  - All colours defined here, imported by all pages
  - No fixed font colours → inherit from Streamlit theme (dark + light safe)
  - Legend background transparent → no white box in dark mode
  - Gridlines use rgba neutral → visible in both modes
  - Backgrounds always rgba(0,0,0,0) → Streamlit controls the page colour
  - Palette is saturated/vibrant (vs the earlier muted "private bank" tones)
    while still reading cleanly on a light background — no neon-on-black,
    since the app stays on Streamlit's native light theme by design.
"""

# ── Brand palette ─────────────────────────────────────────────────────────────
NAVY        = "#1551d1"   # vibrant royal blue (was muted #1f3a6e)
NAVY_LIGHT  = "#3b82f6"   # bright electric blue accent
NAVY_PALE   = "#bfdbfe"
RED_DARK    = "#dc2626"   # saturated red (was muted #8b1a1a)
RED_MED     = "#ef4444"
GREEN_DARK  = "#15803d"
GREEN_MED   = "#16a34a"   # vivid green (was muted #27ae60)
AMBER       = "#f59e0b"   # vibrant amber/gold (was muted #b7770d) — echoes the
                           # gold-miners yellow used as an accent colour
GOLD        = "#eab308"   # high-impact gold accent for highlight lines/markers
SLATE       = "#475569"
WHITE       = "#ffffff"
BORDER      = "rgba(100,116,139,0.25)"

# ── Colour sequences ──────────────────────────────────────────────────────────
SEQ_MAP = [
    [0.0,  "#eff6ff"],
    [0.15, "#bfdbfe"],
    [0.35, "#60a5fa"],
    [0.60, "#3b82f6"],
    [0.80, "#1d4ed8"],
    [1.0,  "#1551d1"],
]

QUINTIL_COLORS = ["#dc2626", "#f59e0b", "#eab308", "#16a34a", "#1551d1"]

DIFF_COLORS = {
    "Bass":     NAVY,
    "Logistic": "#15803d",
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
    Plotly `config=` dict for st.plotly_chart — completely hides the
    modebar (camera/zoom/pan toolbar). It previously sat as a solid grey
    box on top of the legend and chart title in both light and dark mode;
    hiding it entirely removes the visual clutter project-wide.

    Usage:
        st.plotly_chart(fig, use_container_width=True, config=chart_config())
    """
    return {
        "displaylogo": False,
        "displayModeBar": False,  # never shown, not even on hover
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
