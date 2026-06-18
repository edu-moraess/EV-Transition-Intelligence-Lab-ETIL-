# streamlit_pages/02_Country_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analytics.country_analysis import CountryAnalyzer
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel
from visualization.theme import (
    base_layout, bar_color_signed,
    NAVY, NAVY_LIGHT, RED_DARK, GREEN_MED, AMBER, SLATE, BORDER
)

_DIFF_COLORS = {"Bass": NAVY, "Logistic": "#2e7d32", "Gompertz": AMBER}


def render(df: pd.DataFrame):
    st.title("🔬 Country Lab")
    st.caption("Análise quantitativa detalhada por país")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    selected  = st.selectbox("País", countries,
                              index=countries.index("Norway") if "Norway" in countries else 0)
    if not selected: return

    try:
        analyzer = CountryAnalyzer(df, selected)
    except ValueError as e:
        st.error(str(e)); return

    data   = analyzer.get_data()
    latest = data.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Último ano", str(int(latest["year"])))
    c2.metric("Vendas EV", f"{latest['ev_sales']:,.0f}")
    gr   = latest.get("growth_rate", np.nan)
    cagr = latest.get("cagr", np.nan)
    c3.metric("YoY", f"{gr:.1f}%" if not np.isnan(gr) else "N/A",
              delta=f"{gr:.1f}%" if not np.isnan(gr) else None)
    c4.metric("CAGR", f"{cagr:.1f}%" if not np.isnan(cagr) else "N/A")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### Vendas EV anuais")
        fig = go.Figure(go.Bar(
            x=data["year"], y=data["ev_sales"]/1e3,
            marker_color=NAVY,
            hovertemplate="%{x}: %{y:.1f}k unidades<extra></extra>",
        ))
        lay = base_layout(280)
        lay.update({"yaxis":{**lay.get("yaxis",{}),"title":"k unidades"}, "showlegend":False})
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("##### Crescimento YoY (%)")
        gr_vals = data["growth_rate"].fillna(0)
        fig2 = go.Figure(go.Bar(
            x=data["year"], y=gr_vals,
            marker_color=bar_color_signed(gr_vals),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        lay2 = base_layout(280)
        lay2.update({"yaxis":{**lay2.get("yaxis",{}),"title":"%"}, "showlegend":False})
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, use_container_width=True)

    if "ev_stock_share" in data.columns and data["ev_stock_share"].notna().any():
        st.markdown("##### EV Stock Share (%)")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=data["year"], y=data["ev_stock_share"],
            mode="lines+markers",
            line=dict(color=NAVY, width=2),
            marker=dict(size=5, color=NAVY),
            fill="tozeroy", fillcolor=f"rgba(31,58,110,0.08)",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        lay3 = base_layout(220)
        lay3.update({"yaxis":{**lay3.get("yaxis",{}),"title":"Share (%)"}, "showlegend":False})
        fig3.update_layout(**lay3)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("🧬 Modelos de Difusão")
    with st.spinner("Ajustando..."):
        results = analyzer.fit_diffusion_models()

    t   = np.arange(len(data), dtype=float)
    S   = data["ev_sales"].cumsum().values
    t_x = np.arange(0, len(t)+8, dtype=float)
    y_x = np.arange(int(data["year"].min()), int(data["year"].min())+len(t_x))

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=data["year"].values, y=S/1e3,
        mode="markers+lines", name="Observado",
        line=dict(color=SLATE, width=1.5, dash="dot"),
        marker=dict(size=5, color=SLATE),
    ))

    param_cols = st.columns(3)
    for i, (name, model_cls) in enumerate(
        [("Bass",BassDiffusion()), ("Logistic",LogisticGrowth()), ("Gompertz",GompertzModel())]
    ):
        try:
            model_cls.fit(t, S)
            pred = model_cls.predict(t_x)
            fig_d.add_trace(go.Scatter(
                x=y_x, y=pred/1e3, mode="lines", name=name,
                line=dict(color=_DIFF_COLORS[name], width=2, dash="dash"),
            ))
            with param_cols[i]:
                st.success(f"**{name}**")
                for k, v in model_cls.get_params().items():
                    st.caption(f"{k} = {v:.4f}")
        except Exception:
            with param_cols[i]:
                st.warning(f"**{name}** não convergiu")

    fig_d.add_vline(x=int(data["year"].max()), line_dash="dot",
                    line_color=SLATE, line_width=1,
                    annotation_text="Hoje", annotation_font_color=SLATE,
                    annotation_font_size=10)
    lay_d = base_layout(360)
    lay_d.update({"xaxis":{**lay_d.get("xaxis",{}),"title":"Ano"},
                   "yaxis":{**lay_d.get("yaxis",{}),"title":"k unidades (cumulativo)"}})
    fig_d.update_layout(**lay_d)
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")
    st.subheader("🔄 Regimes de Adoção")
    with st.spinner("Detectando..."):
        regimes = analyzer.detect_regimes()

    if "error" in regimes:
        st.warning(regimes["error"])
    else:
        states = regimes["states"]
        rm = {0:"Emerging",1:"Expansion",2:"Acceleration",3:"Maturity"}
        rc = {0:SLATE, 1:NAVY_LIGHT, 2:GREEN_MED, 3:AMBER}
        reg_names = [rm.get(s,f"R{s}") for s in states]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=data["year"].values, y=data["ev_sales"].values/1e3,
            mode="lines", line=dict(color=SLATE, width=1.5), name="Vendas EV",
        ))
        for s in sorted(set(states)):
            mask = np.array(states)==s
            fig_r.add_trace(go.Scatter(
                x=data["year"].values[mask], y=data["ev_sales"].values[mask]/1e3,
                mode="markers", name=rm.get(s,f"R{s}"),
                marker=dict(size=9, color=rc.get(s,"#888"),
                            line=dict(width=1, color="white")),
            ))
        lay_r = base_layout(300)
        lay_r.update({"xaxis":{**lay_r.get("xaxis",{}),"title":"Ano"},
                       "yaxis":{**lay_r.get("yaxis",{}),"title":"k unidades"}})
        fig_r.update_layout(**lay_r)
        st.plotly_chart(fig_r, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Ano": data["year"].values,
            "Regime": reg_names,
            "Vendas (k)": (data["ev_sales"].values/1e3).round(1),
            "YoY (%)": data["growth_rate"].fillna(0).round(1).values
        }), use_container_width=True, hide_index=True)
