# streamlit_pages/06_EVQTI_Ranking.py
import streamlit as st
import pandas as pd
from ingestion.loader import DataLoader
from preprocessing.cleaning import DataCleaner
from preprocessing.features import FeatureEngineer
from models.scoring.evqti import EVQTI

st.title("🏆 EVQTI Ranking")

loader = DataLoader()
df = loader.load_from_csv("EV_Data_Explorer_2026_GEVO_clean.csv")
df = DataCleaner(df).clean()
df = FeatureEngineer(df).engineer()

# Calcular índice
evqti = EVQTI(df)
scores = evqti.compute_scores()

st.subheader("Ranking Global")
st.dataframe(scores.sort_values("EVQTI", ascending=False))