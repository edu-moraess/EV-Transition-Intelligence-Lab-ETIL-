# streamlit_pages/16_Explainable_AI.py
"""Explainable AI — Professional Edition."""
import streamlit as st
import matplotlib.pyplot as plt
import shap
from utils.advanced_analytics import perform_shap_explain
import pandas as pd
import numpy as np

def render(df: pd.DataFrame):
    st.title("🤖 Explainable AI")
    st.markdown("Understand what drives EV sales predictions using SHAP values and feature importance.")

    if df.empty:
        st.warning("No data.")
        return

    # Target selection
    target_options = ["ev_sales", "ev_stock", "penetration"]
    target = st.selectbox("Target variable", [t for t in target_options if t in df.columns])
    if not target:
        st.error("No valid target column.")
        return

    # Train model and get SHAP
    with st.spinner("Training model and computing SHAP values..."):
        try:
            explainer, shap_values, X_test, importance, features = perform_shap_explain(df, target)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            return

    # ── Feature Importance ────────────────────────────────────────────────────
    st.subheader("📊 Global Feature Importance")
    imp_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values("Importance", ascending=True)
    fig1, ax1 = plt.subplots()
    ax1.barh(imp_df["Feature"], imp_df["Importance"], color="skyblue")
    ax1.set_xlabel("Importance")
    ax1.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig1)
    plt.close(fig1)

    # ── SHAP Summary ──────────────────────────────────────────────────────────
    st.subheader("🔍 SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    # ── Dependence Plot ───────────────────────────────────────────────────────
    st.subheader("📈 SHAP Dependence Plot")
    selected_feature = st.selectbox("Select feature", features)
    shap.dependence_plot(selected_feature, shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    # ── Individual Explanation ────────────────────────────────────────────────
    st.subheader("💧 Waterfall Plot (Single Prediction)")
    idx = st.slider("Select observation index", 0, len(X_test)-1, 0)
    try:
        if hasattr(shap, "waterfall_plot"):
            shap.waterfall_plot(shap.Explanation(values=shap_values[idx],
                                                 base_values=explainer.expected_value,
                                                 data=X_test.iloc[idx],
                                                 feature_names=features))
        else:
            shap.force_plot(explainer.expected_value, shap_values[idx,:], X_test.iloc[idx,:], matplotlib=True, show=False)
        st.pyplot(bbox_inches='tight')
        plt.clf()
    except Exception as e:
        st.warning(f"Could not generate waterfall plot: {e}")

    # ── Natural Language Explanation ──────────────────────────────────────────
    st.subheader("📝 Text Explanation")
    # Identify top positive/negative features for the selected instance
    vals = shap_values[idx]
    top_pos_idx = np.argsort(vals)[-3:][::-1]
    top_neg_idx = np.argsort(vals)[:3]
    explanation = f"For this observation, the predicted {target} is pushed **higher** by: "
    explanation += ", ".join([f"{features[i]} (SHAP +{vals[i]:.2f})" for i in top_pos_idx])
    explanation += ". **Lower** by: "
    explanation += ", ".join([f"{features[i]} (SHAP {vals[i]:.2f})" for i in top_neg_idx])
    st.write(explanation)
    st.info("SHAP values represent the contribution of each feature to the prediction relative to the average prediction.")