# streamlit_pages/02_Country_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analytics.country_analysis import CountryAnalyzer
from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel


def render(df: pd.DataFrame):
    st.title("🔬 Country Lab")
    st.caption("Análise quantitativa detalhada por país")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    selected  = st.selectbox(
        "País", countries,
        index=countries.index("Norway") if "Norway" in countries else 0
    )
    if not selected:
        return

    try:
        analyzer = CountryAnalyzer(df, selected)
    except ValueError as e:
        st.error(str(e)); return

    data   = analyzer.get_data()
    latest = data.iloc[-1]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Último ano", str(int(latest["year"])))
    c2.metric("Vendas EV", f"{latest['ev_sales']:,.0f}")
    gr = latest.get("growth_rate", np.nan)
    c3.metric("YoY", f"{gr:.1f}%" if not np.isnan(gr) else "N/A", delta=f"{gr:.1f}%" if not np.isnan(gr) else None)
    cagr_v = latest.get("cagr", np.nan)
    c4.metric("CAGR", f"{cagr_v:.1f}%" if not np.isnan(cagr_v) else "N/A")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📈 Vendas EV históricas")
        fig = px.bar(data, x="year", y="ev_sales",
                     labels={"ev_sales": "Vendas EV", "year": "Ano"},
                     color_discrete_sequence=["#636EFA"])
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📉 Crescimento YoY (%)")
        gr_vals = data["growth_rate"].fillna(0)
        fig2    = px.bar(data, x="year", y="growth_rate",
                         color="growth_rate",
                         color_continuous_scale="RdYlGn",
                         labels={"growth_rate": "YoY (%)", "year": "Ano"})
        fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Stock share
    if "ev_stock_share" in data.columns and data["ev_stock_share"].notna().any():
        st.subheader("🔋 EV Stock Share (%)")
        fig3 = px.area(data, x="year", y="ev_stock_share",
                       labels={"ev_stock_share": "Stock Share (%)", "year": "Ano"},
                       color_discrete_sequence=["#00CC96"])
        fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=240)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("🧬 Modelos de Difusão")
    with st.spinner("Ajustando modelos..."):
        results = analyzer.fit_diffusion_models()

    t   = data.index.values.astype(float)
    S   = data["ev_sales"].cumsum().values
    t_ext   = np.arange(0, len(t)+8, dtype=float)
    yrs_ext = np.arange(int(data["year"].min()), int(data["year"].min())+len(t_ext))

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=data["year"].values, y=S/1e3,
        mode="markers+lines", name="Observado",
        line=dict(width=2), marker=dict(size=6)
    ))
    colors_d = {"Bass": "#636EFA", "Logistic": "#00CC96", "Gompertz": "#AB63FA"}
    _mods    = {"Bass": BassDiffusion(), "Logistic": LogisticGrowth(), "Gompertz": GompertzModel()}
    params_cols = st.columns(3)
    for i, (name, model) in enumerate(_mods.items()):
        try:
            model.fit(t, S)
            fig_d.add_trace(go.Scatter(
                x=yrs_ext, y=model.predict(t_ext)/1e3,
                mode="lines", name=name,
                line=dict(color=colors_d[name], dash="dash", width=2)
            ))
            with params_cols[i]:
                st.success(f"**{name}**")
                for k, v in model.get_params().items():
                    st.caption(f"{k} = {v:.4f}")
        except Exception:
            with params_cols[i]:
                st.warning(f"**{name}** não convergiu")

    fig_d.add_vline(x=int(data["year"].max()), line_dash="dot", annotation_text="Hoje")
    fig_d.update_layout(
        xaxis_title="Ano", yaxis_title="Adoção cumulativa (k)",
        height=360, margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")
    st.subheader("🔄 Detecção de Regimes")
    with st.spinner("Detectando regimes..."):
        regimes = analyzer.detect_regimes()

    if "error" in regimes:
        st.warning(regimes["error"])
    else:
        states = regimes["states"]
        regime_map = {0:"Emerging",1:"Expansion",2:"Acceleration",3:"Maturity"}
        reg_names  = [regime_map.get(s, f"R{s}") for s in states]
        regime_df  = pd.DataFrame({"Ano": data["year"].values, "Regime": reg_names,
                                    "Vendas (k)": (data["ev_sales"].values/1e3).round(1)})
        fig_reg = px.scatter(regime_df, x="Ano", y="Vendas (k)", color="Regime",
                             color_discrete_sequence=px.colors.qualitative.Plotly,
                             size=[10]*len(regime_df))
        fig_reg.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_reg, use_container_width=True)
        st.dataframe(regime_df, use_container_width=True, hide_index=True)
