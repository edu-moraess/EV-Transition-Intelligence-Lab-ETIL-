# streamlit_pages/06_EVQTI_Ranking.py
"""
EVQTI Ranking — fator quant com validação estatística completa.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.scoring.evqti import EVQTI


def render(df: pd.DataFrame):
    dataset = df.attrs.get("dataset", "")
    st.title("🏆 EVQTI Ranking")
    st.markdown(
        "**EV Quant Transition Index v2** — fator anti-leakage com IC, Rank IC e análise de quintis"
    )
    st.caption(f"Dataset: {dataset} | Sinal calculado em t-1 | Target: crescimento t+1")
    st.markdown("---")

    with st.spinner("Ajustando fator EVQTI..."):
        try:
            model = EVQTI()
            model.fit(df)
            rdf       = model.get_ranking()
            ic_df     = model.get_ic_stats()
            decile_df = model.get_decile_stats()
        except Exception as e:
            st.error(f"Erro no EVQTI: {e}")
            import traceback; st.code(traceback.format_exc())
            return

    # ── IC Summary KPIs ───────────────────────────────────────────────────
    st.subheader("📐 Validação Estatística do Fator")
    if not ic_df.empty:
        ic_mean   = ic_df["rank_ic"].mean()
        ic_pos    = (ic_df["rank_ic"] > 0).mean() * 100
        ic_sig    = ic_df["significant"].mean() * 100
        ir        = ic_mean / (ic_df["rank_ic"].std() + 1e-9)  # Information Ratio
        spread_df = decile_df.copy() if not decile_df.empty else pd.DataFrame()
        spread    = (
            spread_df["avg_growth"].iloc[-1] - spread_df["avg_growth"].iloc[0]
            if len(spread_df) >= 2 else np.nan
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rank IC médio",      f"{ic_mean:+.3f}", help="Spearman entre EVQTI e crescimento futuro. |IC|>0.05 = sinal útil")
        c2.metric("% ICs positivos",    f"{ic_pos:.0f}%",  help="Proporção de anos com IC positivo")
        c3.metric("% ICs significativos", f"{ic_sig:.0f}%", help="Anos com p-value < 0.05")
        c4.metric("Spread Q5-Q1",       f"{spread:.1f}pp" if not np.isnan(spread) else "N/A",
                  help="Diferença de crescimento médio entre quintil superior e inferior")

        # IC por ano
        st.markdown("##### IC e Rank IC por ano")
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Bar(
            x=ic_df["year"], y=ic_df["rank_ic"],
            name="Rank IC",
            marker_color=[
                "#2ECC71" if v > 0 else "#E74C3C"
                for v in ic_df["rank_ic"]
            ],
            text=ic_df["rank_ic"].map(lambda v: f"{v:+.3f}"),
            textposition="outside",
        ))
        fig_ic.add_hline(y=0.05,  line_dash="dot", line_color="gray", annotation_text="IC=0.05")
        fig_ic.add_hline(y=-0.05, line_dash="dot", line_color="gray")
        fig_ic.update_layout(
            xaxis_title="Ano", yaxis_title="Rank IC (Spearman)",
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_ic, use_container_width=True)

    st.markdown("---")

    # ── Decile Analysis ───────────────────────────────────────────────────
    if not decile_df.empty:
        st.subheader("📊 Análise de Quintis — Monotônico = Fator Válido")
        st.caption("Crescimento médio (t+1) por quintil de EVQTI (t). Deve ser crescente da esquerda para direita.")
        fig_dec = go.Figure()
        fig_dec.add_trace(go.Bar(
            x=decile_df["decile_label"],
            y=decile_df["avg_growth"],
            error_y=dict(type="data", array=decile_df["std_growth"] / np.sqrt(decile_df["n"]), visible=True),
            marker_color=px.colors.sequential.Viridis[::2][:len(decile_df)],
            text=decile_df["avg_growth"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig_dec.update_layout(
            xaxis_title="Quintil EVQTI", yaxis_title="Crescimento médio YoY (%) em t+1",
            height=320, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown("---")

    # ── Ranking ────────────────────────────────────────────────────────────
    st.subheader("📋 Ranking de Países por EVQTI")
    n_show = st.slider("Países a exibir", 10, len(rdf), min(30, len(rdf)))
    top_n  = rdf.head(n_show).sort_values("evqti_score")

    fig_rank = px.bar(
        top_n, x="evqti_score", y="country", orientation="h",
        color="evqti_score", color_continuous_scale="RdYlGn",
        text=top_n["evqti_score"].map(lambda v: f"{v:.3f}"),
        labels={"evqti_score": "Score EVQTI", "country": ""},
    )
    fig_rank.update_traces(textposition="outside")
    fig_rank.update_layout(
        title=f"EVQTI — Top {n_show} países",
        height=max(400, n_show * 22),
        margin=dict(l=0, r=80, t=40, b=0),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("---")

    # ── Scatter EVQTI × CAGR × Vendas ──────────────────────────────────────
    st.subheader("🔍 EVQTI × CAGR × Vendas")
    df_latest = df[df["year"] == df["year"].max()]
    df_meta   = df_latest.groupby("country").agg(
        cagr=("cagr","mean"), ev_sales=("ev_sales","sum"), growth_rate=("growth_rate","mean")
    ).reset_index()
    scatter_df = rdf.merge(df_meta, on="country", how="left")
    scatter_df["ev_sales_log"] = np.log1p(scatter_df["ev_sales"].fillna(0))
    scatter_df["cagr"]         = scatter_df["cagr"].fillna(scatter_df["cagr"].median())

    fig_sc = px.scatter(
        scatter_df, x="evqti_score", y="cagr",
        size="ev_sales_log", color="evqti_score",
        hover_name="country",
        color_continuous_scale="RdYlGn",
        size_max=45,
        labels={"evqti_score": "Score EVQTI", "cagr": "CAGR (%)", "ev_sales_log": "Vendas (log)"},
    )
    fig_sc.update_layout(
        height=440, margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # ── Tabela completa ───────────────────────────────────────────────────
    st.subheader("📋 Tabela completa")
    full = rdf.merge(df_meta[["country","cagr","ev_sales","growth_rate"]], on="country", how="left")
    full["evqti_score"] = full["evqti_score"].round(3)
    full["cagr"]        = full["cagr"].fillna(0).round(1)
    full["ev_sales"]    = full["ev_sales"].fillna(0).round(0).astype("Int64")
    full["growth_rate"] = full["growth_rate"].fillna(0).round(1)
    full.columns        = ["País","Rank","Score EVQTI","CAGR (%)","Vendas EV","YoY (%)"]
    st.dataframe(full, use_container_width=True, hide_index=True)

    # ── Metodologia ────────────────────────────────────────────────────────
    with st.expander("ℹ️ Metodologia EVQTI v2"):
        st.markdown("""
**Lógica econômica:**
Países em estágio early-stage de difusão (baixa penetração + momentum positivo)
concentram o maior crescimento futuro — conforme a lei de difusão de Bass.

**Construção do fator (anti-leakage):**
- Sinal calculado com features **defasadas (t-1)**: ev_stock_share, ev_sales_share, growth_rate
- Target: crescimento de ev_sales em **t+1**
- Winsorização 1%-99% nas features
- Z-score cross-section + ponderação calibrada pelo IC empírico

**Pesos calibrados:**
| Feature | Direção | Peso |
|---|---|---|
| ev_stock_share (t-1) | Negativo (early-stage) | 45% |
| ev_sales_share (t-1) | Negativo (early-stage) | 35% |
| growth_rate (t-1)    | Positivo (momentum)    | 20% |

**Validação:**
- IC (Pearson sobre ranks) e Rank IC (Spearman) por ano
- Análise de quintis com barras de erro (erro padrão)
- Spread Q5-Q1 mede separação do fator

**Fonte:** IEA Global EV Outlook 2025/2026
        """)
