# app.py
import streamlit as st

st.set_page_config(
    page_title="EV Transition Intelligence Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("ETIL Navigation")
st.sidebar.markdown("Selecione uma página:")

st.title("EV Transition Intelligence Lab (ETIL)")
st.markdown("""
**Tagline:** *A quantitative framework for modeling the dynamics, regimes, and future paths of global electric vehicle adoption.*
""")

st.write("Use o menu lateral para navegar entre as páginas.")
