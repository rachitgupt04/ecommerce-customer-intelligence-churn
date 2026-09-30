"""
Model Evaluation and Business Scenario Threshold Optimization.
Computes technical classification metrics (ROC-AUC, PR-AUC, F1, Recall, Precision, Brier Score)
and simulates financial cost-benefit outcomes across probability thresholds.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from src.utils.logger import get_logger

logger = get_logger("ModelEvaluation")


class ModelEvaluator:
    """Computes technical performance and scenario-based economic cost trade-offs."""

    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculates standard classification and calibration metrics."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        }
        return metrics

    @staticmethod
    def optimize_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fp: float = 10.0,   # Cost of retention voucher/incentive per customer
        cost_fn: float = 120.0,  # Gross profit contribution lost when a churner is missed
        success_rate: float = 0.20,  # Percentage of contacted customers successfully retained
        threshold_steps: int = 50,
    ) -> pd.DataFrame:
        """
        Evaluates precision, recall, and expected scenario economics across varying decision thresholds.
        Identifies the threshold that minimizes net business loss or maximizes net saved profit.
        """
        thresholds = np.linspace(0.10, 0.95, threshold_steps)
        records: List[Dict[str, Any]] = []

        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Business cost calculation:
            # False Positive: Wasted retention spend on someone who stays anyway
            # False Negative: Missed churner resulting in lost customer LTV
            total_business_cost = (fp * cost_fp) + (fn * cost_fn)

            # Expected Campaign Economics:
            targeted = tp + fp
            campaign_cost = targeted * cost_fp
            expected_saved_customers = int(targeted * success_rate)
            gross_saved_margin = expected_saved_customers * cost_fn
            net_saved_value = gross_saved_margin - campaign_cost

            records.append({
                "threshold": round(float(th), 3),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "total_business_cost": round(float(total_business_cost), 2),
                "targeted_customers": int(targeted),
                "campaign_cost": round(float(campaign_cost), 2),
                "expected_saved_customers": int(expected_saved_customers),
                "net_saved_value": round(float(net_saved_value), 2),
            })

        df_thresh = pd.DataFrame(records)
        optimal_row = df_thresh.loc[df_thresh["total_business_cost"].idxmin()]
        logger.info(
            f"Optimal Threshold (Min Cost): {optimal_row['threshold']:.2f} "
            f"(Total Cost: ${optimal_row['total_business_cost']:,.2f}, "
            f"Recall: {optimal_row['recall']:.1%}, Precision: {optimal_row['precision']:.1%})"
        )
        return df_thresh

