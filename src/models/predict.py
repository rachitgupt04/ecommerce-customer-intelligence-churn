"""
Inference and Customer Risk Intelligence Engine.
Executes real-time single-customer scoring with local SHAP attributions and
generates prioritized batch risk scoring tables for CRM integration.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.utils.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RISK_THRESHOLDS,
)
from src.utils.logger import get_logger
from src.utils import load_model
from src.models.explain import OlistModelExplainer
from src.analytics.recommendations import RetentionRecommendationEngine

logger = get_logger("PredictionPipeline")


class OlistPredictor:
    """Production scoring pipeline with model explanations and prescriptive recommendations."""

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
        reports_dir: Path = REPORTS_DIR,
    ):
        self.models_dir = models_dir
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads trained champion model and initializes explainability engine."""
        model_path = self.models_dir / "champion_churn_model.joblib"
        if model_path.exists():
            self.model = load_model(model_path)
            self.explainer = OlistModelExplainer(
                models_dir=self.models_dir,
                reports_dir=self.reports_dir,
            )
            logger.info("Champion model and Explainer loaded successfully.")
        else:
            logger.warning(f"Champion model not found at {model_path}. Train the model before running predictions.")

    def predict_single_customer(self, customer_features: pd.Series) -> Dict[str, Any]:
        """Runs inference, risk tiering, local SHAP explanation, and recommendation for one customer."""
        if self.model is None:
            self._load_artifacts()
            if self.model is None:
                raise RuntimeError("Model artifact is missing. Run model training first.")

        X_df = pd.DataFrame([customer_features[FEATURE_COLUMNS]])
        churn_prob = float(self.model.predict_proba(X_df)[0, 1])
        risk_tier = RetentionRecommendationEngine.get_risk_tier(churn_prob)

        # Local explainability
        top_factors = self.explainer.explain_instance(customer_features, top_n=4) if self.explainer else []
        lead_factor_desc = f"{top_factors[0]['feature']} ({top_factors[0]['direction']})" if top_factors else "High inactivity"

        # Segment & Spend
        segment = customer_features.get("cluster_segment", customer_features.get("rfm_segment", "Standard"))
        monetary = float(customer_features.get("monetary_total", customer_features.get("monetary", 0.0)))
        recency = float(customer_features.get("recency_days", customer_features.get("recency", 0.0)))
        frequency = int(customer_features.get("order_frequency", customer_features.get("frequency", 1)))

        # Prescriptive recommendation
        recommendation = RetentionRecommendationEngine.generate_recommendation(
            segment=segment,
            risk_level=risk_tier,
            churn_prob=churn_prob,
            monetary=monetary,
            recency=recency,
            top_risk_factor=lead_factor_desc,
        )

        customer_id = customer_features.get("customer_unique_id", customer_features.get("customer_id", "N/A"))

        return {
            "customer_id": customer_id,
            "customer_unique_id": customer_id,
            "churn_probability": round(churn_prob, 4),
            "risk_tier": risk_tier,
            "segment": segment,
            "monetary": monetary,
            "lifetime_value": monetary,
            "recency_days": recency,
            "recency": recency,
            "frequency": frequency,
            "important_factors": top_factors,
            "recommendation": recommendation,
        }

    def generate_batch_risk_table(self) -> pd.DataFrame:
        """
        Generates production customer risk table across all active observation customers.
        Merges RFM segments and features, saves to reports/customer_risk_scoring.csv.
        """
        logger.info("Generating full batch customer risk table for CRM export...")
        feat_path = self.processed_dir / "churn_features.csv"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature matrix not found at {feat_path}.")

        df_feat = pd.read_csv(feat_path)

        # Merge segment labels if available
        seg_path = self.processed_dir / "customer_segments.csv"
        if seg_path.exists():
            df_seg = pd.read_csv(seg_path)
            merge_cols = [c for c in ["customer_unique_id", "cluster_segment", "rfm_segment"] if c in df_seg.columns]
            df_feat = df_feat.merge(df_seg[merge_cols], on="customer_unique_id", how="left")
        else:
            df_feat["cluster_segment"] = "Standard"
            df_feat["rfm_segment"] = "Standard"

        # Run vector batch inference
        X = df_feat[FEATURE_COLUMNS]
        probs = self.model.predict_proba(X)[:, 1]

        df_risk = pd.DataFrame()
        df_risk["customer_unique_id"] = df_feat["customer_unique_id"]
        # Also provide customer_id alias for dashboard compatibility
        df_risk["customer_id"] = df_feat["customer_unique_id"]
        df_risk["churn_probability"] = np.round(probs, 4)
        df_risk["risk_tier"] = df_risk["churn_probability"].apply(RetentionRecommendationEngine.get_risk_tier)
        df_risk["segment"] = df_feat["cluster_segment"].fillna("Standard")
        df_risk["rfm_segment"] = df_feat["rfm_segment"].fillna("Standard")
        df_risk["lifetime_value"] = np.round(df_feat["monetary_total"], 2)
        df_risk["recency"] = np.round(df_feat["recency_days"], 1)
        df_risk["frequency"] = df_feat["order_frequency"].astype(int)
        df_risk["avg_order_value"] = np.round(df_feat["avg_order_value"], 2)

        # Sort by churn risk descending, then monetary descending
        df_risk.sort_values(by=["churn_probability", "lifetime_value"], ascending=[False, False], inplace=True)

        # Export to reports and processed dirs
        out_report = self.reports_dir / "customer_risk_scoring.csv"
        out_proc = self.processed_dir / "customer_risk_scoring.csv"
        df_risk.to_csv(out_report, index=False)
        df_risk.to_csv(out_proc, index=False)
        logger.info(f"Batch risk scoring complete. Processed {len(df_risk):,} customers -> {out_report}")

        return df_risk


if __name__ == "__main__":
    predictor = OlistPredictor()
    df_scored = predictor.generate_batch_risk_table()
    print("Scored preview:\n", df_scored.head())

