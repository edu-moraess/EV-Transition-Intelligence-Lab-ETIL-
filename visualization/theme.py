"""Shared visual system for ETIL research views."""

NAVY = "#0F3B5E"
NAVY_LIGHT = "#3A6D8C"
NAVY_PALE = "#DCE6F1"
RED_DARK = "#A84442"
RED_MED = "#C75A57"
GREEN_DARK = "#2E6A55"
GREEN_MED = "#3F8068"
AMBER = "#A56F16"
GOLD = "#C18A2A"
SLATE = "#556575"
WHITE = "#FFFFFF"
BORDER = "rgba(70, 88, 106, 0.20)"

SEQ_MAP = [
    [0.0, "#EEF3F8"],
    [0.25, "#CBDCEB"],
    [0.55, "#7FA7C1"],
    [0.80, "#3A6D8C"],
    [1.0, "#0F3B5E"],
]

QUINTIL_COLORS = [RED_DARK, AMBER, "#B1A56A", GREEN_MED, NAVY]

DIFF_COLORS = {
    "Bass": NAVY,
    "Logistic": GREEN_DARK,
    "Gompertz": AMBER,
}


def base_layout(height: int = 340, title: str = "") -> dict:
    """Return a restrained Plotly layout suitable for analytical comparison."""
    return {
        "title": {"text": title, "font": {"size": 13, "family": "Inter, Arial, sans-serif"}, "x": 0},
        "height": height,
        "margin": {"l": 2, "r": 10, "t": 36 if title else 8, "b": 4},
        "font": {"family": "Inter, Arial, sans-serif", "size": 11, "color": "#334155"},
        "xaxis": {
            "gridcolor": "rgba(70, 88, 106, 0.10)",
            "gridwidth": 0.5,
            "linecolor": "rgba(70, 88, 106, 0.28)",
            "showline": True,
            "zeroline": False,
            "tickfont": {"size": 10},
        },
        "yaxis": {
            "gridcolor": "rgba(70, 88, 106, 0.10)",
            "gridwidth": 0.5,
            "linecolor": "rgba(70, 88, 106, 0.28)",
            "showline": False,
            "zeroline": False,
            "tickfont": {"size": 10},
        },
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "showlegend": True,
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "font": {"size": 10},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        "hoverlabel": {"bordercolor": "rgba(70, 88, 106, 0.30)", "font": {"size": 11}},
    }


def bar_color_signed(values) -> list:
    """Use restrained positive and negative colours for signed series."""
    return [NAVY if value >= 0 else RED_DARK for value in values]


def chart_config() -> dict:
    """Remove non-analytical toolbar controls while retaining responsive charts."""
    return {"displaylogo": False, "displayModeBar": False, "responsive": True}


def modebar_layout_patch() -> dict:
    """Keep the Plotly toolbar neutral if a page opts into displaying it."""
    return {
        "modebar": {
            "bgcolor": "rgba(0,0,0,0)",
            "color": "rgba(70, 88, 106, 0.60)",
            "activecolor": NAVY,
        }
    }
