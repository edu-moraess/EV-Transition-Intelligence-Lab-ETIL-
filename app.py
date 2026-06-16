import streamlit as st
import importlib

# Lista de páginas com nome amigável e módulo correspondente
PAGES = {
    "Global Monitor": "streamlit_pages.01_Global_Monitor",
    "Country Lab": "streamlit_pages.02_Country_Lab",
    "Diffusion Engine": "streamlit_pages.03_Diffusion_Engine",
    "Regime Detection": "streamlit_pages.04_Regime_Detection",
    "Forecast Lab": "streamlit_pages.05_Forecast_Lab",
    "EVQTI Ranking": "streamlit_pages.06_EVQTI_Ranking",
}

def main():
    st.sidebar.title("Navegação")
    selection = st.sidebar.radio("Ir para:", list(PAGES.keys()))

    # Carrega dinamicamente o módulo
    module_name = PAGES[selection]
    page_module = importlib.import_module(module_name)

    # Executa a função app() da página
    page_module.app()

if __name__ == "__main__":
    main()