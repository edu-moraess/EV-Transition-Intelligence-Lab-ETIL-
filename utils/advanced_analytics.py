# utils/advanced_analytics.py
"""
Advanced analytics utilities for ETIL v2.0.

Provides:
- ETRI calculation
- Country clustering (K‑Means, DBSCAN, HDBSCAN)
- Scenario projection
- SHAP explainer wrapper
- Anomaly detection suite
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import logging

logger = logging.getLogger(__name__)


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a series or an empty series with the same index, avoiding KeyError."""
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index, name=col)


def calculate_etri(df: pd.DataFrame) -> pd.DataFrame:
    """
    EV Transition Risk Index (0–100, higher = better transition readiness).
    Adapts automatically to missing columns.
    """
    vars_ideal = {
        "ev_stock_share": 0.20,
        "ev_sales_share": 0.15,
        "charging_points_per_100k": 0.15,
        "gdp_per_capita": 0.10,
        "population": 0.05,
        "co2_emissions": 0.10,
        "oil_dependence": 0.10,
        "growth_rate": 0.10,
        "cagr": 0.05,
        "momentum": 0.00,
    }
    available = {}
    for var in vars_ideal:
        if var in df.columns and df[var].notna().sum() > 0:
            available[var] = vars_ideal[var]

    if not available:
        logger.warning("No valid variables for ETRI. Returning neutral scores.")
        df["ETRI"] = 50.0
        return df

    total_w = sum(available.values())
    scaled_weights = {k: v / total_w for k, v in available.items()}

    def normalize_positive(series):
        s = series.dropna()
        if s.nunique() <= 1:
            return (series - s.min()) / (s.max() - s.min() + 1e-9) if s.max() != s.min() else pd.Series(0.5, index=series.index)
        return (series - s.min()) / (s.max() - s.min())

    def normalize_negative(series):
        s = series.dropna()
        if s.nunique() <= 1:
            return (s.max() - series) / (s.max() - s.min() + 1e-9) if s.max() != s.min() else pd.Series(0.5, index=series.index)
        return (s.max() - series) / (s.max() - s.min())

    score = pd.Series(0.0, index=df.index)
    for var, w in scaled_weights.items():
        if var == "ev_stock_share":
            score += w * normalize_positive(df["ev_stock_share"].astype(float))
        elif var == "ev_sales_share":
            score += w * normalize_positive(df["ev_sales_share"].astype(float))
        elif var == "charging_points_per_100k":
            score += w * normalize_positive(df["charging_points_per_100k"].astype(float))
        elif var == "gdp_per_capita":
            score += w * normalize_positive(df["gdp_per_capita"].astype(float))
        elif var == "population":
            score += w * normalize_positive(df["population"].astype(float))
        elif var == "co2_emissions":
            score += w * normalize_negative(df["co2_emissions"].astype(float))
        elif var == "oil_dependence":
            score += w * normalize_negative(df["oil_dependence"].astype(float))
        elif var == "growth_rate":
            score += w * normalize_positive(df["growth_rate"].astype(float))
        elif var == "cagr":
            score += w * normalize_positive(df["cagr"].astype(float))

    df["ETRI"] = score * 100

    bins = [0, 20, 40, 60, 80, 101]
    labels = ["Critical", "High", "Moderate", "Low", "Very Low"]
    df["ETRI_category"] = pd.cut(df["ETRI"], bins=bins, labels=labels, include_lowest=True)
    return df


def country_clustering(df: pd.DataFrame, method: str = "kmeans", n_clusters: int = None,
                       dbscan_eps: float = 0.5, dbscan_min_samples: int = 5) -> pd.DataFrame:
    feature_cols = [
        "ev_sales", "ev_stock", "penetration", "growth_rate",
        "charging_points_per_100k", "gdp_per_capita", "population"
    ]
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError("No clustering features available.")

    df_latest = df.sort_values("year").groupby("country").last().reset_index()
    X = df_latest[available].copy()
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "kmeans":
        if n_clusters is None:
            best_k = 2
            best_sil = -1
            for k in range(2, min(10, len(X) - 1)):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            n_clusters = best_k
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif method == "dbscan":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model.fit_predict(X_scaled)
    elif method == "hdbscan":
        try:
            import hdbscan
            model = hdbscan.HDBSCAN(min_cluster_size=dbscan_min_samples, gen_min_span_tree=True)
            labels = model.fit_predict(X_scaled)
        except ImportError:
            raise ImportError("hdbscan package not installed.")
    else:
        raise ValueError("Unknown clustering method.")

    df_latest["cluster"] = labels.astype(str)
    cluster_map = df_latest.set_index("country")["cluster"].to_dict()
    df["cluster"] = df["country"].map(cluster_map)
    return df


def auto_name_clusters(df_clustered: pd.DataFrame) -> dict:
    latest = df_clustered.sort_values("year").groupby("country").last().reset_index()

    agg_dict = {}
    if "ev_sales" in latest.columns:
        agg_dict["ev_sales_mean"] = ("ev_sales", "mean")
    if "penetration" in latest.columns:
        agg_dict["penetration_mean"] = ("penetration", "mean")
    if "gdp_per_capita" in latest.columns:
        agg_dict["gdp_mean"] = ("gdp_per_capita", "mean")
    if "growth_rate" in latest.columns:
        agg_dict["growth_mean"] = ("growth_rate", "mean")

    if not agg_dict:
        counts = latest["cluster"].value_counts().to_dict()
        return {str(c): f"Cluster {c} ({cnt} countries)" for c, cnt in counts.items()}

    cluster_profiles = latest.groupby("cluster").agg(**agg_dict)

    names = {}
    for cluster, row in cluster_profiles.iterrows():
        penetration = row.get("penetration_mean", np.nan)
        growth = row.get("growth_mean", np.nan)
        gdp = row.get("gdp_mean", np.nan)

        if not np.isnan(penetration) and penetration > 30:
            names[cluster] = "Early Leaders"
        elif not np.isnan(growth) and growth > 20:
            names[cluster] = "Fast Growing"
        elif not np.isnan(gdp) and gdp < 5000:
            names[cluster] = "Emerging Markets"
        else:
            names[cluster] = "Late Adopters"
    return names


def project_scenario(df: pd.DataFrame, adjustments: dict) -> pd.DataFrame:
    """
    Adjust base year data with scenario parameters and project
    EV sales/stock/penetration using simple elasticity assumptions.
    Only modifies columns that exist in the dataset.
    """
    base_year = df["year"].max()
    base = df[df["year"] == base_year].copy()

    oil_elasticity = -0.02
    lithium_elasticity = -0.015
    interest_elasticity = -0.03
    incentive_elasticity = 0.05
    carbon_tax_elasticity = 0.04
    gdp_elasticity = 0.1

    factors = pd.Series(1.0, index=base.index)
    if "oil_price_change" in adjustments:
        factors *= 1 + oil_elasticity * adjustments["oil_price_change"] / 100
    if "lithium_price_change" in adjustments:
        factors *= 1 + lithium_elasticity * adjustments["lithium_price_change"] / 100
    if "interest_rate_change" in adjustments:
        factors *= 1 + interest_elasticity * adjustments["interest_rate_change"]
    if "incentive_change" in adjustments:
        factors *= 1 + incentive_elasticity * adjustments["incentive_change"] / 100
    if "carbon_tax_change" in adjustments:
        factors *= 1 + carbon_tax_elasticity * adjustments["carbon_tax_change"] / 100
    if "gdp_growth_change" in adjustments:
        factors *= 1 + gdp_elasticity * adjustments["gdp_growth_change"] / 100

    # Only update columns that exist
    if "ev_sales" in base.columns:
        base["ev_sales"] = base["ev_sales"] * factors
    if "ev_stock" in base.columns:
        base["ev_stock"] = base["ev_stock"] * factors
    if "penetration" in base.columns:
        base["penetration"] = base["penetration"] * factors.clip(0, None)
    return base


def perform_shap_explain(df: pd.DataFrame, target_col: str = "ev_sales"):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import shap

    feature_cols = [
        "gdp_per_capita", "population", "charging_points_per_100k",
        "ev_stock_share", "growth_rate", "cagr", "momentum"
    ]
    available = [c for c in feature_cols if c in df.columns]
    if target_col not in df.columns or not available:
        raise ValueError("Required columns missing for Explainable AI.")

    df_model = df.dropna(subset=available + [target_col]).copy()
    X = df_model[available]
    y = df_model[target_col]

    if len(X) < 20:
        raise ValueError("Not enough data for model training.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values, X_test, model.feature_importances_, available


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "ev_sales", "ev_stock", "penetration", "growth_rate",
        "charging_points_per_100k", "gdp_per_capita", "population"
    ]
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return df.assign(is_anomaly=False)

    df_latest = df.sort_values("year").groupby("country").last().reset_index()
    X = df_latest[available].fillna(df_latest[available].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.1, random_state=42)
    iso_flag = iso.fit_predict(X_scaled)
    lof = LocalOutlierFactor(contamination=0.1, novelty=False)
    lof_flag = lof.fit_predict(X_scaled)
    svm = OneClassSVM(nu=0.1)
    svm_flag = svm.fit_predict(X_scaled)

    flags = np.column_stack([iso_flag, lof_flag, svm_flag])
    anomaly_score = ((flags == -1).sum(axis=1) >= 2)
    df_latest["is_anomaly"] = anomaly_score
    return df.merge(df_latest[["country", "is_anomaly"]], on="country", how="left")