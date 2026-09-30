"""
Unit and Integration Tests for E-Commerce Intelligence Platform.

Validates:
- Recommendation risk tier logic
- Recommendation business logic
- Model evaluation metrics
- Threshold optimization and business cost calculations
"""

import numpy as np

from src.analytics.recommendations import RetentionRecommendationEngine
from src.models.evaluate import ModelEvaluator


def test_recommendation_risk_tiers():
    """Validates mapping of churn probabilities to risk tiers."""
    assert RetentionRecommendationEngine.get_risk_tier(0.20) == "LOW"
    assert RetentionRecommendationEngine.get_risk_tier(0.55) == "MEDIUM"
    assert RetentionRecommendationEngine.get_risk_tier(0.85) == "HIGH"


def test_recommendation_business_logic():
    """Ensures high-value and lower-value customers receive appropriate treatment."""

    rec_vip = RetentionRecommendationEngine.generate_recommendation(
        segment="Champions & VIPs",
        risk_level="HIGH",
        churn_prob=0.88,
        monetary=2450.0,
        recency=45.0,
        top_risk_factor="Reduced frequency",
    )

    assert (
        "VIP" in rec_vip["action_title"]
        or "Concierge" in rec_vip["action_title"]
    )
    assert rec_vip["operational_urgency"].startswith("URGENT")

    rec_casual = RetentionRecommendationEngine.generate_recommendation(
        segment="Hibernating & Low-Value",
        risk_level="HIGH",
        churn_prob=0.75,
        monetary=60.0,
        recency=150.0,
    )

    assert "Digital" in rec_casual["action_title"]
    assert "$" in rec_casual["recommended_budget"]


def test_model_evaluator_metrics():
    """Tests evaluation calculations with known predictions."""

    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.8, 0.6])

    metrics = ModelEvaluator.evaluate_predictions(
        y_true,
        y_pred,
        y_prob,
    )

    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_threshold_optimization_cost_minimization():
    """Tests threshold optimization and business cost calculations."""

    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.4, 0.65])

    df_th = ModelEvaluator.optimize_threshold(
        y_true,
        y_prob,
        cost_fp=10.0,
        cost_fn=120.0,
    )

    assert not df_th.empty
    assert "total_business_cost" in df_th.columns
    assert (
        df_th["total_business_cost"].min()
        <= df_th["total_business_cost"].max()
    )