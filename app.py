import streamlit as st

# Importa cada página com alias para nomes amigáveis
from streamlit_pages import (
    _01_Global_Monitor as Global_Monitor,
    _02_Country_Lab as Country_Lab,
    _03_Diffusion_Engine as Diffusion_Engine,
    _04_Regime_Detection as Regime_Detection,
    _05_Forecast_Lab as Forecast_Lab,
    _06_EVQTI_Ranking as EVQTI_Ranking
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