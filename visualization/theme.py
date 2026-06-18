# visualization/theme.py
"""
Paleta institucional quant — light mode.

Inspiração: Bloomberg Terminal (sem o dark), Goldman Sachs Research,
Bridgewater Associates reports. Austero, legível, denso de informação.

Regras:
  - Azul navy (#1f3a6e) para elementos primários e positivos
  - Vermelho escuro (#8b1a1a) para negativos / alertas
  - Cinza neutro (#4a4a4a) para texto secundário
  - Fundo branco / off-white (#f8f9fb)
  - Sem gradientes chamativos — só quando semanticamente útil
  - Sequência divergente: vermelho → branco → azul (para heatmaps)
  - Sequência linear: cinzas → azul navy (para rankings)
"""

# ── Cores base ────────────────────────────────────────────────────────────
NAVY        = "#1f3a6e"
NAVY_LIGHT  = "#2e5090"
NAVY_PALE   = "#d0daea"
RED_DARK    = "#8b1a1a"
RED_MED     = "#c0392b"
GREEN_DARK  = "#1a4731"
GREEN_MED   = "#27ae60"
AMBER       = "#b7770d"
SLATE       = "#4a4a4a"
SLATE_LIGHT = "#8a8a8a"
WHITE       = "#ffffff"
OFF_WHITE   = "#f8f9fb"
BORDER      = "#dce3ed"

# ── Sequências de cor ─────────────────────────────────────────────────────

# Barras de ranking — do mais fraco ao mais forte (cinza → navy)
SEQ_NAVY = [
    "#c8d4e8", "#a8bbd8", "#8aa2c8", "#6b88b8",
    "#4d6ea8", "#335796", "#1f3a6e",
]

# Divergente IC — vermelho (negativo) → branco → azul (positivo)
DIV_RdBu = [
    RED_DARK, "#c0392b", "#e8a0a0", "#f5d5d5",
    WHITE,
    "#c8d4e8", "#8aa2c8", NAVY_LIGHT, NAVY,
]

# Quintis — de fraco a forte
QUINTIL_COLORS = ["#c0392b", "#e67e22", "#f1c40f", "#2ecc71", "#1f3a6e"]

# Mapa coroplético — branco → navy
SEQ_MAP = [
    [0.0,  "#f8f9fb"],
    [0.15, "#d0daea"],
    [0.35, "#8aa2c8"],
    [0.60, "#4d6ea8"],
    [0.80, "#2e5090"],
    [1.0,  "#1f3a6e"],
]

# ── Layout Plotly base ────────────────────────────────────────────────────

def base_layout(height: int = 340, title: str = "") -> dict:
    """Layout Plotly mínimo e limpo — sem fundo fixo."""
    return dict(
        title=dict(text=title, font=dict(size=13, color=SLATE), x=0),
        height=height,
        margin=dict(l=0, r=8, t=36 if title else 8, b=0),
        font=dict(family="Inter, Arial, sans-serif", size=11, color=SLATE),
        xaxis=dict(
            gridcolor=BORDER, gridwidth=0.5,
            linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor=BORDER, gridwidth=0.5,
            linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(size=10),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=BORDER, borderwidth=1,
            font=dict(size=10),
        ),
        hoverlabel=dict(
            bgcolor=WHITE, bordercolor=BORDER,
            font=dict(size=11, color=SLATE),
        ),
    )


def bar_color_signed(values) -> list:
    """Azul navy para positivo, vermelho escuro para negativo."""
    import numpy as np
    return [NAVY if v >= 0 else RED_DARK for v in values]
