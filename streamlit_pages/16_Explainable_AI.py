# streamlit_pages/16_Explainable_AI.py
"""Explainable AI with SHAP."""
import streamlit as st
import matplotlib.pyplot as plt
import shap
from utils.advanced_analytics import perform_shap_explain
import pandas as pd

def render(df: pd.DataFrame):
    st.title("🤖 Explainable AI")
    st.markdown("Understand what drives EV sales predictions with SHAP.")

    if df.empty:
        st.warning("No data.")
        return

    target_options = ["ev_sales", "ev_stock", "penetration"]
    target = st.selectbox("Target variable", [t for t in target_options if t in df.columns])
    if not target:
        st.error("No valid target column.")
        return

    try:
        explainer, shap_values, X_test, feature_importance, features = perform_shap_explain(df, target)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return

    st.subheader("Feature Importance (Global)")
    imp_df = pd.DataFrame({"Feature": features, "Importance": feature_importance}).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    st.subheader("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    st.subheader("SHAP Dependence Plot")
    selected_feature = st.selectbox("Select feature for dependence plot", features)
    shap.dependence_plot(selected_feature, shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    st.subheader("Force Plot (First observation)")
    idx = 0
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[idx,:], X_test.iloc[idx,:], matplotlib=True)
    st.pyplot(force_plot)
    plt.clf()

    st.markdown("""**Interpretation:**  
    The force plot shows how each feature pushes the prediction away from the base value.  
    Red arrows increase the predicted EV sales, blue arrows decrease them.""")