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
    try:
        shap.summary_plot(shap_values, X_test, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.warning(f"Could not generate summary plot: {e}")

    # ── Dependence Plot ───────────────────────────────────────────────────────
    st.subheader("📈 SHAP Dependence Plot")
    selected_feature = st.selectbox("Select feature", features)
    try:
        shap.dependence_plot(selected_feature, shap_values, X_test, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.warning(f"Could not generate dependence plot: {e}")

    # ── Individual Explanation ────────────────────────────────────────────────
    st.subheader("💧 Waterfall Plot (Single Prediction)")
    idx = st.slider("Select observation index", 0, len(X_test)-1, 0)

    try:
        # Build Explanation object correctly
        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[idx].values,   # <- must be a 1D array
            feature_names=features
        )
        shap.waterfall_plot(exp, max_display=10)
        fig = plt.gcf()
        fig.tight_layout()
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.warning(f"Could not generate waterfall plot: {e}")

    # ── Natural Language Explanation ──────────────────────────────────────────
    st.subheader("📝 Text Explanation")
    # Identify top positive/negative features for the selected instance
    vals = shap_values[idx]
    # Sort indices by magnitude (descending)
    sorted_idx = np.argsort(np.abs(vals))[::-1]
    top_positive = []
    top_negative = []
    for i in sorted_idx:
        if vals[i] > 0.01 * np.std(vals):   # ignore tiny contributions
            top_positive.append((features[i], vals[i]))
        elif vals[i] < -0.01 * np.std(vals):
            top_negative.append((features[i], vals[i]))
        if len(top_positive) >= 3 and len(top_negative) >= 3:
            break

    explanation = ""
    if top_positive:
        explanation += "This prediction is **increased** by: "
        explanation += ", ".join([f"{feat} (+{val:.1f})" for feat, val in top_positive[:3]])
        explanation += ". "
    if top_negative:
        explanation += "It is **decreased** by: "
        explanation += ", ".join([f"{feat} ({val:.1f})" for feat, val in top_negative[:3]])
        explanation += "."
    if not explanation:
        explanation = "All features have negligible impact on this prediction."

    st.write(explanation)
    st.info("SHAP values represent the contribution of each feature to the prediction relative to the average prediction.")