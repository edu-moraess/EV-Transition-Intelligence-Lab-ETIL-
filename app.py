import streamlit as st

# Importa cada página do pacote streamlit_pages
from streamlit_pages import (
    Global_Monitor,
    Country_Lab,
    Diffusion_Engine,
    Regime_Detection,
    Forecast_Lab,
    EVQTI_Ranking
)

# Dicionário de páginas
PAGES = {
    "Global Monitor": Global_Monitor,
    "Country Lab": Country_Lab,
    "Diffusion Engine": Diffusion_Engine,
    "Regime Detection": Regime_Detection,
    "Forecast Lab": Forecast_Lab,
    "EVQTI Ranking": EVQTI_Ranking,
}

def main():
    st.sidebar.title("Navegação")
    selection = st.sidebar.radio("Ir para:", list(PAGES.keys()))

    page = PAGES[selection]
    page.app()  # cada página precisa ter uma função app()

if __name__ == "__main__":
    main()