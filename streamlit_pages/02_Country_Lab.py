# streamlit_pages/02_Country_Lab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from analytics.country_analysis import CountryAnalyzer

_BG = "#161b22"
_GRID = "#21262d"
_TEXT = "#e6edf3"
_DIM = "#8b949e"


def _layout(title=""):
    return dict(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_TEXT), title=title,
        margin=dict(l=0,r=0,t=36,b=0),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID)
    )


def render(df: pd.DataFrame):
    st.title("🔬 Country Lab")
    st.markdown("Análise quantitativa detalhada por país")
    st.markdown("---")

    countries = sorted(df["country"].unique())
    selected  = st.selectbox("Selecione um país", countries, index=countries.index("Norway") if "Norway" in countries else 0)

    if not selected:
        return

    try:
        analyzer = CountryAnalyzer(df, selected)
    except ValueError as e:
        st.error(str(e)); return

    data = analyzer.get_data()
    latest = data.iloc[-1]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Último ano", str(int(latest["year"])))
    c2.metric("Vendas EV", f"{latest['ev_sales']:,.0f}")
    c3.metric("CAGR", f"{latest['cagr']:.1f}%" if not np.isnan(latest.get('cagr', np.nan)) else "N/A")
    gr = latest.get("growth_rate", np.nan)
    c4.metric("Crescimento YoY", f"{gr:.1f}%" if not np.isnan(gr) else "N/A", delta=f"{gr:.1f}%" if not np.isnan(gr) else None)

    st.markdown("---")

    # Histórico de vendas + growth rate
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📈 Série histórica — Vendas EV")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=data["year"], y=data["ev_sales"]/1e3,
                             name="Vendas (k)", marker_color="#388bfd"))
        fig.update_layout(**_layout(), yaxis_title="Mil unidades", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📉 Taxa de crescimento anual (%)")
        fig2 = go.Figure()
        gr_vals = data["growth_rate"].fillna(0)
        colors = ["#3fb950" if v >= 0 else "#f85149" for v in gr_vals]
        fig2.add_trace(go.Bar(x=data["year"], y=gr_vals, name="YoY %",
                              marker_color=colors))
        fig2.update_layout(**_layout(), yaxis_title="%", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Penetração (ev_stock_share se disponível)
    if "ev_stock_share" in data.columns and data["ev_stock_share"].notna().any():
        st.subheader("🔋 Penetração EV — Stock Share (%)")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=data["year"], y=data["ev_stock_share"],
            mode="lines+markers", line=dict(color="#d2a8ff", width=2),
            fill="tozeroy", fillcolor="rgba(210,168,255,0.1)", name="EV Stock Share"
        ))
        fig3.update_layout(**_layout(), yaxis_title="%", height=280)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Modelos de difusão
    st.subheader("🧬 Modelos de Difusão")
    with st.spinner("Ajustando modelos..."):
        results = analyzer.fit_diffusion_models()

    cols = st.columns(3)
    model_colors = {"Bass": "#388bfd", "Logistic": "#3fb950", "Gompertz": "#d2a8ff"}
    for i, (model_name, res) in enumerate(results.items()):
        with cols[i]:
            if res["fitted"]:
                params = res["params"]
                param_str = " · ".join([f"{k}={v:.3f}" for k,v in params.items()])
                st.success(f"**{model_name}** convergiu")
                st.caption(param_str)
            else:
                st.warning(f"**{model_name}** não convergiu")

    # Plot curvas
    t = data.index.values
    S = data["ev_sales"].cumsum().values
    fig_diff = go.Figure()
    fig_diff.add_trace(go.Scatter(x=data["year"].values, y=S/1e3,
                                   mode="markers+lines", name="Observado",
                                   line=dict(color="#e6edf3", width=1.5)))
    from models.diffusion import BassDiffusion, LogisticGrowth, GompertzModel
    _models = {"Bass": BassDiffusion(), "Logistic": LogisticGrowth(), "Gompertz": GompertzModel()}
    for name, model in _models.items():
        try:
            model.fit(t, S)
            t_ext = np.arange(0, len(t)+8)
            years_ext = np.arange(data["year"].min(), data["year"].min()+len(t_ext))
            fig_diff.add_trace(go.Scatter(
                x=years_ext, y=model.predict(t_ext)/1e3,
                mode="lines", name=name,
                line=dict(color=model_colors[name], dash="dash")
            ))
        except:
            pass
    fig_diff.update_layout(**_layout("Curvas de Difusão — Adoção Cumulativa"),
                            yaxis_title="Mil unidades (cumulativo)", height=360)
    st.plotly_chart(fig_diff, use_container_width=True)

    st.markdown("---")

    # Detecção de regimes
    st.subheader("🔄 Detecção de Regimes")
    with st.spinner("Detectando regimes..."):
        regimes = analyzer.detect_regimes()

    if "error" in regimes:
        st.warning(regimes["error"])
    else:
        states = regimes["states"]
        regime_map = {0:"Emerging",1:"Expansion",2:"Acceleration",3:"Maturity"}
        regime_colors = {0:"#8b949e",1:"#388bfd",2:"#3fb950",3:"#d2a8ff"}
        regime_names = [regime_map.get(s, f"R{s}") for s in states]

        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(
            x=data["year"].values, y=data["ev_sales"].values/1e3,
            mode="lines", name="Vendas EV", line=dict(color="#e6edf3", width=2)
        ))
        for s in sorted(set(states)):
            mask = np.array(states) == s
            years_s = data["year"].values[mask]
            vals_s  = data["ev_sales"].values[mask]/1e3
            fig_reg.add_trace(go.Scatter(
                x=years_s, y=vals_s, mode="markers",
                marker=dict(size=10, color=regime_colors.get(s,"#888"), line=dict(width=1,color="#0d1117")),
                name=regime_map.get(s,f"R{s}")
            ))
        fig_reg.update_layout(**_layout("Regimes de Adoção EV"),
                               yaxis_title="Mil unidades", height=320)
        st.plotly_chart(fig_reg, use_container_width=True)

        regime_df = pd.DataFrame({"Ano": data["year"].values, "Regime": regime_names})
        st.dataframe(regime_df, use_container_width=True, hide_index=True)
 