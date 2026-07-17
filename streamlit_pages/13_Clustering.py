# streamlit_pages/13_Clustering.py
"""Country Clustering — Professional Edition."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.advanced_analytics import country_clustering, auto_name_clusters
from sklearn.decomposition import PCA
import numpy as np

def render(df: pd.DataFrame):
    st.title("🧩 Country Clustering")
    st.markdown("Segment countries by EV adoption, infrastructure, and economic indicators using unsupervised learning.")

    if df.empty:
        st.warning("No data.")
        return

    # Clustering controls
    method = st.selectbox("Algorithm", ["kmeans", "dbscan", "hdbscan"], index=0, help="K-Means with automatic k selection, DBSCAN, or HDBSCAN (if installed).")
    with st.spinner("Running clustering..."):
        try:
            df_clust = country_clustering(df, method=method)
        except Exception as e:
            st.error(f"Clustering failed: {e}")
            return

    latest = df_clust.sort_values("year").groupby("country").last().reset_index()
    # Generate cluster names
    cluster_names = auto_name_clusters(df_clust)

    # ── Cluster Map ───────────────────────────────────────────────────────────
    st.subheader("🗺️ Cluster World Map")
    # Map cluster labels to names
    latest["cluster_name"] = latest["cluster"].map(cluster_names)
    fig_map = px.choropleth(
        latest,
        locations="country",
        locationmode="country names",
        color="cluster_name",
        title="Country Clusters",
        hover_name="country",
        hover_data={"cluster": True, "cluster_name": False},
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig_map, use_container_width=True)

    # ── PCA Projection ────────────────────────────────────────────────────────
    st.subheader("📉 PCA Projection (2D)")
    feature_cols = ["ev_sales", "ev_stock", "penetration", "growth_rate",
                    "charging_points_per_100k", "gdp_per_capita", "population"]
    available = [c for c in feature_cols if c in df_clust.columns]
    X = latest[available].fillna(latest[available].median())
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plot_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    plot_df["country"] = latest["country"]
    plot_df["cluster"] = latest["cluster_name"]
    fig_pca = px.scatter(
        plot_df, x="PC1", y="PC2", color="cluster",
        hover_name="country", title="Countries in 2D PCA Space",
        template="plotly_white"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── 3D PCA (optional) ─────────────────────────────────────────────────────
    if len(available) >= 3:
        st.subheader("🧊 3D PCA")
        pca3 = PCA(n_components=3)
        comp3 = pca3.fit_transform(X)
        fig3d = px.scatter_3d(
            x=comp3[:, 0], y=comp3[:, 1], z=comp3[:, 2],
            color=latest["cluster_name"], hover_name=latest["country"],
            title="3D PCA Projection"
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # ── Cluster Profiles ──────────────────────────────────────────────────────
    st.subheader("📊 Cluster Profiles")
    # Compute aggregated statistics
    agg_metrics = {}
    if "penetration" in latest.columns:
        agg_metrics["Avg Penetration (%)"] = ("penetration", "mean")
    if "growth_rate" in latest.columns:
        agg_metrics["Avg Growth Rate (%)"] = ("growth_rate", "mean")
    if "gdp_per_capita" in latest.columns:
        agg_metrics["Avg GDP/capita"] = ("gdp_per_capita", "mean")
    agg_metrics["Country Count"] = ("country", "count")

    if agg_metrics:
        profile = latest.groupby("cluster_name").agg(**agg_metrics).round(2)
        profile.columns = [col.replace("_", " ").title() for col in profile.columns]
        st.dataframe(profile, use_container_width=True)
    else:
        st.info("No aggregation metrics available.")

    # Cluster descriptions
    with st.expander("ℹ️ Cluster Interpretation", expanded=True):
        for cluster, name in cluster_names.items():
            st.markdown(f"**{name}**")
            st.write(f"Countries: {latest[latest['cluster_name'] == name]['country'].tolist()}")