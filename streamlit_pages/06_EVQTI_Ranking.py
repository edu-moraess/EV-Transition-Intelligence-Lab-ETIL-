# streamlit_pages/06_EVQTI_Ranking.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from models.scoring.evqti import EVQTI
from visualization.theme import (
    base_layout, bar_color_signed,
    NAVY, NAVY_LIGHT, NAVY_PALE, RED_DARK, RED_MED,
    GREEN_MED, AMBER, SLATE, BORDER,
    SEQ_NAVY, DIV_RdBu, QUINTIL_COLORS, SEQ_MAP
)


def _metric_card(col, label, value, sub="", color=None):
    col.metric(label, value, sub)


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "")
    st.title("🏆 EVQTI Ranking")
    st.markdown(
        "**EV Quant Transition Index v3** — sinal anti-leakage calibrado por IC empírico"
    )
    st.caption(
        f"Dataset: {dataset} | Features: ev_stock_share(t-1) · ev_sales_share(t-1) | "
        f"Pesos calibrados por |IC| | Target: crescimento EV em t+1"
    )
    st.markdown("---")

    with st.spinner("Calculando EVQTI v3..."):
        try:
            model = EVQTI()
            model.fit(df)
            rdf   = model.get_ranking()
            ic_df = model.get_ic_stats()
            dec   = model.get_decile_stats()
            s     = model.get_summary()
        except Exception as e:
            st.error(f"Erro no EVQTI: {e}")
            import traceback; st.code(traceback.format_exc())
            return

    # ── KPIs ─────────────────────────────────────────────────────────────
    st.subheader("📐 Validação Estatística do Fator")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rank IC médio",        f"{s.get('ic_mean',0):+.3f}",
              help="Spearman entre EVQTI(t-1) e crescimento(t+1). |IC|>0.05 = útil; >0.10 = bom.")
    c2.metric("IR (IC/σ)",            f"{s.get('ir',0):.2f}",
              help="Information Ratio. IR>0.5 = sinal consistente.")
    c3.metric("% anos IC+",           f"{s.get('pct_positive',0):.0f}%",
              help="Proporção de anos com IC positivo.")
    c4.metric("% anos IC sig.",       f"{s.get('pct_sig',0):.0f}%",
              help="Anos com p-value < 0.05.")
    c5.metric("Spread Q5-Q1",         f"{s.get('spread_q5_q1',0):.1f}pp",
              help="Diferença de crescimento futuro médio entre quintil 5 e quintil 1.")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # ── IC por ano ───────────────────────────────────────────────────────
    with col_a:
        st.markdown("##### IC e Rank IC por ano")
        if not ic_df.empty:
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Bar(
                x=ic_df["year"],
                y=ic_df["rank_ic"],
                marker_color=bar_color_signed(ic_df["rank_ic"]),
                text=ic_df["rank_ic"].map(lambda v: f"{v:+.3f}"),
                textposition="outside",
                textfont=dict(size=9),
                name="Rank IC",
                hovertemplate="Ano: %{x}<br>Rank IC: %{y:.3f}<br>n=%{customdata}",
                customdata=ic_df["n"],
            ))
            fig_ic.add_hline(y=0.05,  line_dash="dot",
                             line_color=NAVY_PALE, line_width=1,
                             annotation_text="IC=0.05",
                             annotation_font_color=SLATE,
                             annotation_font_size=9)
            fig_ic.add_hline(y=-0.05, line_dash="dot",
                             line_color=NAVY_PALE, line_width=1)
            lay = base_layout(height=300)
            lay.update({"yaxis": {**lay.get("yaxis",{}),
                                   "title": "Rank IC (Spearman)",
                                   "range": [-0.45, 0.60]}})
            fig_ic.update_layout(**lay)
            st.plotly_chart(fig_ic, use_container_width=True)

            # t-stat tabela compacta
            with st.expander("Tabela IC por ano"):
                show = ic_df[["year","rank_ic","t_stat","p_value","n","significant"]].copy()
                show.columns = ["Ano","Rank IC","t-stat","p-value","n","Sig."]
                st.dataframe(show.sort_values("Ano", ascending=False),
                             use_container_width=True, hide_index=True)

    # ── Quintis ──────────────────────────────────────────────────────────
    with col_b:
        st.markdown("##### Análise de Quintis — Monotônico = Fator Válido")
        st.caption("Crescimento YoY futuro (t+1) por quintil de EVQTI (t). Deve ser crescente.")
        if not dec.empty:
            fig_dec = go.Figure()
            fig_dec.add_trace(go.Bar(
                x=dec["decile_label"],
                y=dec["avg_growth"],
                error_y=dict(
                    type="data",
                    array=dec["se"].values,
                    visible=True,
                    color=SLATE,
                    thickness=1.2,
                    width=4,
                ),
                marker_color=QUINTIL_COLORS[:len(dec)],
                text=dec["avg_growth"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                textfont=dict(size=9, color=SLATE),
                name="Crescimento médio futuro",
                hovertemplate="%{x}<br>Crescimento: %{y:.1f}%<br>SE: ±%{error_y.array:.1f}pp<extra></extra>",
            ))
            lay2 = base_layout(height=300)
            lay2.update({"yaxis": {**lay2.get("yaxis",{}),
                                    "title": "Crescimento YoY médio (%) em t+1"},
                          "showlegend": False})
            fig_dec.update_layout(**lay2)
            st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown("---")

    # ── Ranking horizontal ───────────────────────────────────────────────
    st.subheader("📋 Ranking de Países")
    n_show = st.slider("Países a exibir", 10, len(rdf), min(30, len(rdf)))
    top_n  = rdf.head(n_show).sort_values("evqti_score")

    # Cor proporcional ao score — navy gradient
    norm = (top_n["evqti_score"] - top_n["evqti_score"].min()) / \
           (top_n["evqti_score"].max() - top_n["evqti_score"].min() + 1e-9)

    fig_rank = go.Figure(go.Bar(
        x=top_n["evqti_score"],
        y=top_n["country"],
        orientation="h",
        marker=dict(
            color=norm,
            colorscale=[[0, "#c8d4e8"], [0.5, "#4d6ea8"], [1.0, NAVY]],
            showscale=False,
        ),
        text=top_n["evqti_score"].map(lambda v: f"{v:.3f}"),
        textposition="outside",
        textfont=dict(size=9, color=SLATE),
        hovertemplate="%{y}<br>Score: %{x:.4f}<extra></extra>",
    ))
    lay3 = base_layout(height=max(380, n_show * 22), title=f"EVQTI — Top {n_show}")
    lay3.update({"xaxis": {**lay3.get("xaxis",{}), "title": "Score EVQTI"},
                  "yaxis": {**lay3.get("yaxis",{}), "title": ""},
                  "margin": {**lay3.get("margin",{}), "r": 80},
                  "showlegend": False})
    fig_rank.update_layout(**lay3)
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("---")

    # ── Scatter EVQTI × CAGR × Vendas ────────────────────────────────────
    st.subheader("🔍 EVQTI × CAGR × Escala de Mercado")
    df_l   = df[df["year"] == df["year"].max()]
    df_meta = df_l.groupby("country").agg(
        cagr=("cagr","mean"),
        ev_sales=("ev_sales","sum"),
        growth_rate=("growth_rate","mean")
    ).reset_index()
    sc_df = rdf.merge(df_meta, on="country", how="left")
    sc_df["ev_sales_log"] = np.log1p(sc_df["ev_sales"].fillna(0))
    sc_df["cagr"]         = sc_df["cagr"].fillna(sc_df["cagr"].median())

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=sc_df["evqti_score"],
        y=sc_df["cagr"],
        mode="markers",
        marker=dict(
            size=sc_df["ev_sales_log"] * 3,
            color=sc_df["evqti_score"],
            colorscale=[[0, "#c8d4e8"], [0.5, "#4d6ea8"], [1.0, NAVY]],
            showscale=True,
            colorbar=dict(title="EVQTI", thickness=10, len=0.6),
            line=dict(width=0.5, color="white"),
            opacity=0.85,
        ),
        text=sc_df["country"],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "EVQTI: %{x:.3f}<br>"
            "CAGR: %{y:.1f}%<extra></extra>"
        ),
    ))
    lay4 = base_layout(height=420)
    lay4.update({"xaxis": {**lay4.get("xaxis",{}), "title": "Score EVQTI"},
                  "yaxis": {**lay4.get("yaxis",{}), "title": "CAGR (%)"},
                  "showlegend": False})
    fig_sc.update_layout(**lay4)
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # ── Tabela completa ───────────────────────────────────────────────────
    st.subheader("Tabela completa")
    full = rdf.merge(df_meta[["country","cagr","ev_sales","growth_rate"]], on="country", how="left")
    full["evqti_score"] = full["evqti_score"].round(4)
    full["cagr"]        = full["cagr"].fillna(0).round(1)
    full["ev_sales"]    = full["ev_sales"].fillna(0).round(0).astype("Int64")
    full["growth_rate"] = full["growth_rate"].fillna(0).round(1)
    full.columns        = ["País","Rank","Score EVQTI","CAGR (%)","Vendas EV","YoY (%)"]
    st.dataframe(full, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Metodologia EVQTI v3"):
        st.markdown("""
**Lógica econômica:** países em estágio early-stage de difusão EV (baixa penetração atual)
concentram o maior crescimento futuro — lei de Bass aplicada ao cross-section de países.

**Construção do fator (anti-leakage):**

| Feature (t-1) | Direção | Peso | IC empírico |
|---|---|---|---|
| ev_stock_share | Negativo (early-stage) | 48.5% | −0.259 |
| ev_sales_share | Negativo (early-stage) | 51.5% | −0.275 |

*growth_rate foi excluída: IC = −0.037, p = 0.36 (não significativo).*

**Pipeline de normalização:**
1. Winsorização 1%-99% nas features raw
2. Lags (t-1) criados sobre os valores winsorizados e re-winsorizados
3. Z-score cross-section por ano
4. Score final = -(w₁·z₁ + w₂·z₂) — sinal negativo inverte early-stage
5. Target winsorizados em 5%-95%

**Interpretação do IC:**
- |IC| < 0.05 = sem sinal
- |IC| 0.05–0.10 = sinal fraco mas útil
- |IC| > 0.10 = sinal robusto ← **EVQTI v3: IC = +0.238**
- IR > 0.5 = consistente ← **IR = 0.99**

**Fonte:** IEA Global EV Outlook 2025/2026
        """)
