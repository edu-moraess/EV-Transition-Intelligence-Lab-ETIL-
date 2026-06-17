# streamlit_pages/02_Country_Lab.py
"""
Página: Country Lab - análise detalhada por país.
"""
import streamlit as st
import pandas as pd
from analytics.country_analysis import CountryAnalyzer
from visualization.charts import Charts

def render(df_processed):
    st.title("🔬 Country Lab")
    countries = sorted(df_processed["country"].unique())
    selected = st.selectbox("Selecione um país", countries)

    if selected:
        analyzer = CountryAnalyzer(df_processed, selected)
        data = analyzer.get_data()

        st.subheader(f"📈 Dados históricos: {selected}")
        st.line_chart(data.set_index("year")["ev_sales"])

        # Métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("Último ano", data["year"].max())
        col2.metric("Vendas no último ano", f"{data['ev_sales'].iloc[-1]:,.0f}")
        col3.metric("CAGR", f"{data['cagr'].iloc[-1]:.2f}%" if "cagr" in data else "N/A")

        # Modelos de difusão
        st.subheader("Modelos de Difusão")
        results = analyzer.fit_diffusion_models()
        for model_name, res in results.items():
            if res["fitted"]:
                st.write(f"{model_name}: {res['params']}")
            else:
                st.warning(f"{model_name}: não convergiu.")

        # Regimes
        st.subheader("Detecção de Regimes")
        regimes = analyzer.detect_regimes()
        if "error" in regimes:
            st.error(regimes["error"])
        else:
            st.write("Estados (0=Emerging, 1=Expansion, 2=Acceleration, 3=Maturity):")
            st.write(regimes["states"])