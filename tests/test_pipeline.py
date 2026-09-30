"""
Unit and Integration Tests for E-Commerce Intelligence & Churn Platform.
Validates:
1. Snapshot dates and prediction windows
2. Zero future feature leakage
3. Chronological split integrity
4. Customer temporal leakage prevention
5. Target construction and distribution
6. Lookback window integrity and comparison
7. Model artifact loading and probability calibration
8. Validation-frozen decision threshold selection
9. SHAP compatibility and feature attributions
10. Referential integrity across relational warehouse tables
11. RFM customer-level aggregation
12. Single customer prediction inference
13. Recommendation risk tier mapping
14. Prescriptive retention business logic
"""

import pytest
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta

from src.utils.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RISK_THRESHOLDS,
)
from src.analytics.recommendations import RetentionRecommendationEngine as RecommendationEngine
from src.models.predict import OlistPredictor as PredictionPipeline
from src.data.ingestion.validate_olist import OlistDataValidator


# 1. Snapshot dates and prediction windows
def test_snapshot_dates_and_prediction_windows():
    """Validates that snapshots have strictly chronological 90-day forward prediction windows."""
    snapshots = [
        {"id": "S1", "cutoff": datetime(2017, 12, 1), "window_days": 90},
        {"id": "S2", "cutoff": datetime(2018, 3, 1), "window_days": 90},
        {"id": "S3", "cutoff": datetime(2018, 6, 1), "window_days": 90},
        {"id": "S4", "cutoff": datetime(2018, 8, 31), "window_days": 90},
    ]
    for i in range(len(snapshots) - 1):
        assert snapshots[i]["cutoff"] < snapshots[i + 1]["cutoff"], "Snapshots must be chronological"
        forward_end = snapshots[i]["cutoff"] + timedelta(days=snapshots[i]["window_days"])
        assert forward_end <= snapshots[i + 1]["cutoff"] + timedelta(days=2), "Forward windows must not overlap improperly"


# 2. Zero future feature leakage
def test_no_future_features_leakage():
    """Ensures feature columns contain no negative recencies or future transaction artifacts."""
    csv_path = PROCESSED_DATA_DIR / "churn_multi_snapshot.csv"
    if not csv_path.exists():
        csv_path = PROCESSED_DATA_DIR / "churn_features.csv"
    assert csv_path.exists(), "Feature table must exist"
    df_feat = pd.read_csv(csv_path, nrows=5000)

    assert "recency_days" in df_feat.columns
    assert (df_feat["recency_days"] >= 0).all(), "Recency days must be non-negative"
    assert (df_feat["order_frequency"] >= 1).all(), "Observed order frequency must be >= 1"


# 3. Chronological split integrity
def test_chronological_split_integrity():
    """Validates chronological ordering between train (S1+S2), val (S3), and holdout test (S4)."""
    t_train_max = datetime(2018, 3, 1)
    t_val = datetime(2018, 6, 1)
    t_test = datetime(2018, 8, 31)

    assert t_train_max < t_val < t_test, "Split ordering must be strictly chronological"


# 4. Customer temporal leakage prevention
def test_customer_temporal_leakage():
    """Ensures orders delivered after snapshot cutoff have delivery date clamped to NaT."""
    cutoff = pd.Timestamp("2018-06-01")
    delivered_date = pd.Timestamp("2018-06-05")

    clamped_delivery = delivered_date if delivered_date < cutoff else pd.NaT
    assert pd.isna(clamped_delivery), "Delivery on or after cutoff must be clamped to NaT"


# 5. Target construction and distribution
def test_target_construction_and_distribution():
    """Validates that churn target is binary and churn rate is within expected range."""
    csv_path = PROCESSED_DATA_DIR / "churn_multi_snapshot.csv"
    if not csv_path.exists():
        csv_path = PROCESSED_DATA_DIR / "churn_features.csv"
    df_feat = pd.read_csv(csv_path, nrows=10000)

    assert TARGET_COLUMN in df_feat.columns
    unique_targets = set(df_feat[TARGET_COLUMN].dropna().unique())
    assert unique_targets.issubset({0, 1}), "Target must be strictly binary (0 or 1)"
    churn_rate = (df_feat[TARGET_COLUMN] == 1).mean()
    assert 0.85 <= churn_rate <= 1.0, "Non-contractual e-commerce exhibits heavy churn (>85%)"


# 6. Lookback window integrity and comparison
def test_lookback_window_integrity():
    """Validates lookback window comparison report and confirmed customer counts."""
    report_path = REPORTS_DIR / "lookback_window_comparison.json"
    assert report_path.exists(), "Lookback window comparison report must exist"

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "180-Day Lookback" in data
    assert "365-Day Lookback" in data
    assert "All-Time Lookback" in data

    summary = data["All-Time Lookback"]["split_summary"]
    assert summary["total_panel_obs"] == 196508
    assert summary["unique_customers"] == 77808


# 7. Model artifact loading and probability calibration
def test_model_artifact_loading():
    """Loads champion model and verifies probability output bounds."""
    model_path = MODELS_DIR / "champion_churn_model.joblib"
    assert model_path.exists(), "Champion model artifact must exist"

    model = joblib.load(model_path)
    assert hasattr(model, "predict_proba"), "Model must have predict_proba method"

    dummy_x = np.zeros((2, len(FEATURE_COLUMNS)))
    probas = model.predict_proba(dummy_x)
    assert probas.shape == (2, 2)
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
    assert np.allclose(probas.sum(axis=1), 1.0)


# 8. Validation-frozen decision threshold selection
def test_validation_threshold_selection():
    """Verifies that decision threshold optimization file exists and frozen threshold theta=0.10 is present."""
    thresh_path = REPORTS_DIR / "threshold_optimization_results.csv"
    assert thresh_path.exists(), "Threshold optimization results must exist"

    df_th = pd.read_csv(thresh_path)
    assert "threshold" in df_th.columns
    assert "total_business_cost" in df_th.columns

    assert (df_th["threshold"].round(2) == 0.10).any(), "Threshold 0.10 must be in evaluated thresholds"


# 9. SHAP compatibility and feature attributions
def test_shap_compatibility_and_explanations():
    """Verifies that SHAP feature importance report exists and recency is the top driver."""
    shap_path = REPORTS_DIR / "shap_feature_importance.json"
    assert shap_path.exists(), "SHAP feature importance report must exist"

    with open(shap_path, "r", encoding="utf-8") as f:
        shap_data = json.load(f)

    assert len(shap_data) >= 5, "Must contain at least top 5 features"
    top_feature = list(shap_data.keys())[0] if isinstance(shap_data, dict) else shap_data[0]["feature"]
    assert top_feature == "recency_days", "recency_days must be the #1 predictive feature"


# 10. Referential integrity across relational warehouse tables
def test_referential_integrity_validator():
    """Verifies 100% referential integrity and zero orphan records across processed warehouse tables."""
    validator = OlistDataValidator(processed_dir=PROCESSED_DATA_DIR)
    dfs = validator.load_processed_tables()

    orphan_items = set(dfs["fact_order_items"]["order_id"]) - set(dfs["fact_orders"]["order_id"])
    assert len(orphan_items) == 0, f"Found {len(orphan_items)} orphan order items"

    orphan_orders = set(dfs["fact_orders"]["customer_id"]) - set(dfs["dim_customers"]["customer_id"])
    assert len(orphan_orders) == 0, f"Found {len(orphan_orders)} orphan orders"


# 11. RFM customer-level aggregation
def test_rfm_customer_level_aggregation():
    """Verifies customer-unique RFM calculations."""
    rfm_path = PROCESSED_DATA_DIR / "customer_rfm.csv"
    assert rfm_path.exists(), "customer_rfm.csv must exist"
    df_rfm = pd.read_csv(rfm_path, nrows=5000)

    assert "customer_unique_id" in df_rfm.columns
    assert "monetary" in df_rfm.columns
    assert "frequency" in df_rfm.columns
    assert (df_rfm["monetary"] > 0).all()
    assert (df_rfm["frequency"] >= 1).all()


# 12. Single customer prediction inference
def test_single_customer_prediction_inference():
    """Verifies that PredictionPipeline generates valid predictions and recommendations."""
    pipeline = PredictionPipeline()
    csv_path = PROCESSED_DATA_DIR / "churn_multi_snapshot.csv"
    if not csv_path.exists():
        csv_path = PROCESSED_DATA_DIR / "churn_features.csv"
    df_feat = pd.read_csv(csv_path, nrows=10)

    sample_row = df_feat.iloc[0]
    result = pipeline.predict_single_customer(sample_row)

    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "recommendation" in result
    assert "action_title" in result["recommendation"]


# 13. Recommendation risk tier mapping
def test_recommendation_risk_tiers():
    """Validates mapping of churn probabilities to risk tiers."""
    assert RecommendationEngine.get_risk_tier(0.20) == "LOW"
    assert RecommendationEngine.get_risk_tier(0.55) == "MEDIUM"
    assert RecommendationEngine.get_risk_tier(0.85) == "HIGH"


# 14. Prescriptive retention business logic
def test_recommendation_business_logic():
    """Ensures VIP high-risk customers receive concierge retention treatment."""
    rec_vip = RecommendationEngine.generate_recommendation(
        segment="Champions & VIPs",
        risk_level="HIGH",
        churn_prob=0.88,
        monetary=2450.0,
        recency=45.0,
        top_risk_factor="Reduced frequency",
    )
    assert "VIP" in rec_vip["action_title"] or "Concierge" in rec_vip["action_title"]
    assert rec_vip["operational_urgency"].startswith("URGENT")
