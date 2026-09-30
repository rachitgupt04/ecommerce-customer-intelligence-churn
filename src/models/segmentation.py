"""
Unsupervised Customer Segmentation Module for Real Olist E-Commerce Data.
Applies log-transformation and standardization on RFM metrics, evaluates optimal clusters
via Elbow Method (Inertia) and Silhouette Analysis (k=2 through k=8), and fits K-Means.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from src.utils.logger import get_logger
from src.utils import save_model, save_json

logger = get_logger("Segmentation")


class OlistSegmentationEngine:
    """Executes cluster evaluation and K-Means segmentation on real RFM features."""

    def __init__(self, data_dir: Path = PROCESSED_DATA_DIR, models_dir: Path = MODELS_DIR):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.reports_dir = REPORTS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.kmeans_model = None

    def evaluate_cluster_k(self, X_scaled: np.ndarray, k_range: range = range(2, 9)) -> Dict[str, Any]:
        """Calculates Inertia (Elbow) and Silhouette Score for k in k_range."""
        logger.info(f"Evaluating clustering metrics across k={list(k_range)}...")
        inertias = []
        silhouettes = []

        # Subsample for silhouette if dataset is large to maintain fast execution
        n_samples = min(15000, len(X_scaled))
        idx_sample = np.random.choice(len(X_scaled), size=n_samples, replace=False)
        X_sub = X_scaled[idx_sample]

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(float(km.inertia_))
            sil = float(silhouette_score(X_sub, labels[idx_sample]))
            silhouettes.append(sil)
            logger.info(f"k={k}: Inertia = {km.inertia_:,.1f} | Silhouette Score = {sil:.4f}")

        eval_results = {
            "k_values": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouettes,
        }
        save_json(eval_results, self.reports_dir / "kmeans_cluster_evaluation.json")
        return eval_results

    def fit_segmentation(self, df_rfm: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalizes RFM distributions with log1p, evaluates k=2..8, fits K-Means,
        and generates data-driven segment profiles based on empirical centroids.
        """
        logger.info(f"Fitting K-Means segmentation with k={n_clusters} clusters on Olist RFM...")

        rfm_features = df_rfm[["recency", "frequency", "monetary"]].copy()
        # Log1p transformation handles severe positive skew in monetary and frequency
        rfm_log = np.log1p(rfm_features)

        # Standardize features
        X_scaled = self.scaler.fit_transform(rfm_log)

        # Evaluate k=2..8
        eval_metrics = self.evaluate_cluster_k(X_scaled, k_range=range(2, 9))

        # Fit champion K-Means
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        df_rfm["cluster_id"] = cluster_labels

        # Derive empirical cluster centroids on original scales
        cluster_summary = df_rfm.groupby("cluster_id").agg(
            customer_count=("customer_unique_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            median_monetary=("monetary", "median"),
        ).reset_index()

        cluster_summary["pct_customers"] = (cluster_summary["customer_count"] / len(df_rfm)) * 100.0
        cluster_summary["pct_customers"] = cluster_summary["pct_customers"].round(2)

        # Name segments based strictly on empirical cluster characteristics
        # Rank clusters by spend
        cluster_summary = cluster_summary.sort_values(by="avg_monetary", ascending=False).reset_index(drop=True)
        segment_names = {}
        for rank_idx, row in cluster_summary.iterrows():
            cid = row["cluster_id"]
            if rank_idx == 0:
                segment_names[cid] = "Champions & High-Value VIPs"
            elif row["avg_recency"] > 180 and row["avg_monetary"] > cluster_summary["avg_monetary"].median():
                segment_names[cid] = "At-Risk High Spenders"
            elif row["avg_recency"] <= 150 and row["avg_frequency"] >= 1.5:
                segment_names[cid] = "Loyal & Consistent"
            else:
                segment_names[cid] = "Hibernating / One-Time Buyers"

        df_rfm["cluster_segment"] = df_rfm["cluster_id"].map(segment_names)
        cluster_summary["segment_name"] = cluster_summary["cluster_id"].map(segment_names)

        # Save artifacts
        save_model(self.kmeans_model, self.models_dir / "kmeans_segmentation.joblib")
        save_model(self.scaler, self.models_dir / "segmentation_scaler.joblib")

        df_rfm.to_csv(self.data_dir / "customer_segments.csv", index=False)
        cluster_summary.to_csv(self.reports_dir / "cluster_profiles.csv", index=False)

        logger.info(f"Customer segmentation complete. Empirical profiles:\n{cluster_summary}")
        return df_rfm, cluster_summary


if __name__ == "__main__":
    df_rfm = pd.read_csv(PROCESSED_DATA_DIR / "customer_rfm.csv")
    seg = OlistSegmentationEngine()
    df_seg, summary = seg.fit_segmentation(df_rfm)
    print("Segmentation finished successfully.")

