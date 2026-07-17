# streamlit_pages/13_Clustering.py
"""Country Clustering page."""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.advanced_analytics import country_clustering, auto_name_clusters
from sklearn.decomposition import PCA
import numpy as np

def render(df: pd.DataFrame):
    st.title("🧩 Country Clustering")
    st.markdown("Segment countries by EV adoption, infrastructure, and economic indicators.")

    if df.empty:
        st.warning("No data.")
        return

    method = st.selectbox("Clustering Algorithm", ["kmeans", "dbscan", "hdbscan"], index=0)
    try:
        df_clust = country_clustering(df, method=method)
    except Exception as e:
        st.error(f"Clustering failed: {e}")
        return

    # Visualisation
    st.subheader("PCA Projection")
    feature_cols = ["ev_sales", "ev_stock", "penetration", "growth_rate",
                    "charging_points_per_100k", "gdp_per_capita", "population"]
    available = [c for c in feature_cols if c in df_clust.columns]
    latest = df_clust.sort_values("year").groupby("country").last().reset_index()
    X = latest[available].fillna(latest[available].median())
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plot_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    plot_df["country"] = latest["country"]
    plot_df["cluster"] = latest["cluster"]
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", hover_name="country",
                     title="Countries in 2D PCA Space")
    st.plotly_chart(fig, use_container_width=True)

    # Cluster statistics
    st.subheader("Cluster Profiles")
    names = auto_name_clusters(df_clust)
    st.write("**Suggested Names:**", names)
    cluster_stats = latest.groupby("cluster").agg(
        Count=("country", "count"),
        Avg_Penetration=("penetration", "mean"),
        Avg_GDP=("gdp_per_capita", "mean"),
    ).reset_index()
    st.dataframe(cluster_stats)

    # 3D PCA (if plotly supports)
    if len(available) >= 3:
        pca3 = PCA(n_components=3)
        comp3 = pca3.fit_transform(X)
        fig3d = px.scatter_3d(x=comp3[:,0], y=comp3[:,1], z=comp3[:,2],
                              color=latest["cluster"], hover_name=latest["country"],
                              title="3D PCA")
        st.plotly_chart(fig3d, use_container_width=True)