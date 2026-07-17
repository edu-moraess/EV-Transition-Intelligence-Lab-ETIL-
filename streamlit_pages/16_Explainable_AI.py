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
    plt.close(fig)

    st.subheader("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    st.subheader("SHAP Dependence Plot")
    selected_feature = st.selectbox("Select feature for dependence plot", features)
    shap.dependence_plot(selected_feature, shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    st.subheader("Waterfall Plot (First observation)")
    try:
        # Use the new waterfall API if available (SHAP ≥0.42), else fallback to force plot
        if hasattr(shap, "waterfall_plot"):
            shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                                 base_values=explainer.expected_value,
                                                 data=X_test.iloc[0],
                                                 feature_names=features))
        else:
            # Fallback to force plot with matplotlib (no initjs needed)
            shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
        st.pyplot(bbox_inches='tight')
        plt.clf()
    except Exception as e:
        st.warning(f"Could not generate individual explanation plot: {e}")

    st.markdown("""**Interpretation:**  
    The waterfall/force plot shows how each feature pushes the prediction away from the base value.  
    Red/positive contributions increase the predicted EV sales, blue/negative contributions decrease them.""")