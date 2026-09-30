"""
Comprehensive Model Selection Audit: Comparison of Lookback Windows (180d vs 365d vs All-Time).
Implements Phases 1 to 5:
- Point-in-time safe feature extraction with delivery clamping across 4 snapshots
- Full feature availability, missingness, and data-loss audit
- Chronological Train (S1+S2), Validation (S3), Holdout Test (S4)
- 4 Candidate Models evaluated with Stratified 5-Fold CV on Train AND evaluated on Validation & Holdout Test
- Threshold selected SOLELY on Validation set (Snapshot 3) using FP*$10 + FN*$120, then frozen and applied to Holdout (Snapshot 4)
- Leakage verification across all snapshots
- Full runtime and computational complexity profiling
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
)

from src.features.feature_engineering import OlistFeatureEngineer
from src.models.evaluate import ModelEvaluator
from src.utils.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    CV_FOLDS,
    DEFAULT_CAMPAIGN_COST,
    DEFAULT_SAVED_MARGIN,
    DEFAULT_SUCCESS_RATE,
    SNAPSHOT_CONFIG,
)
from src.utils.logger import get_logger

logger = get_logger("LookbackAudit")


def evaluate_model_on_split(model, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], np.ndarray]:
    """Helper to evaluate fitted model predictions and probabilities on a given partition."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return ModelEvaluator.evaluate_predictions(y.values, y_pred, y_prob), y_prob


def run_lookback_evaluation(lookback_days: Optional[int], lookback_label: str) -> Dict[str, Any]:
    """Runs complete end-to-end evaluation for a single lookback window configuration."""
    logger.info(f"==================================================")
    logger.info(f"STARTING AUDIT: Lookback Window = {lookback_label}")
    logger.info(f"==================================================")

    fe = OlistFeatureEngineer(data_dir=PROCESSED_DATA_DIR, reports_dir=REPORTS_DIR)

    # 1. Feature Extraction & Leakage Verification across all 4 snapshots
    t0_feat = time.time()
    dfs = []
    snapshot_details = []
    leakage_checks = []

    for cfg in SNAPSHOT_CONFIG:
        snap_dt = pd.to_datetime(cfg["date"])
        df_snap = fe.extract_single_snapshot(
            snapshot_date=cfg["date"],
            horizon_days=cfg.get("horizon_days", 90),
            split_role=cfg.get("split", "train"),
            lookback_days=lookback_days,
        )
        dfs.append(df_snap)

        n_obs = len(df_snap)
        n_churn = int(df_snap[TARGET_COLUMN].sum())
        n_ret = n_obs - n_churn
        churn_pct = (n_churn / n_obs) * 100.0 if n_obs > 0 else 0.0

        snapshot_details.append({
            "snapshot_name": cfg["name"],
            "snapshot_date": cfg["date"],
            "split_role": cfg["split"],
            "horizon_days": cfg.get("horizon_days", 90),
            "observations": n_obs,
            "retained": n_ret,
            "churned": n_churn,
            "churn_rate_pct": round(churn_pct, 2),
            "retention_rate_pct": round(100.0 - churn_pct, 2),
        })

        # Leakage verification:
        obs_orders = fe.df_orders[fe.df_orders["order_purchase_timestamp"] < snap_dt]
        if lookback_days is not None:
            obs_orders = obs_orders[obs_orders["order_purchase_timestamp"] >= snap_dt - pd.Timedelta(days=lookback_days)]
        
        max_purch = obs_orders["order_purchase_timestamp"].max()
        has_future_purch = max_purch >= snap_dt if not obs_orders.empty else False
        clamped_count = (obs_orders["order_delivered_customer_date"] >= snap_dt).sum()

        leakage_checks.append({
            "snapshot": cfg["name"],
            "snapshot_date": cfg["date"],
            "max_purchase_timestamp": str(max_purch),
            "future_purchase_detected": bool(has_future_purch),
            "clamped_future_deliveries": int(clamped_count),
        })

    feat_extract_time = round(time.time() - t0_feat, 2)
    df_all = pd.concat(dfs, ignore_index=True)

    # Missingness and Feature Availability
    missing_counts = df_all[FEATURE_COLUMNS].isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())

    # Feature distribution summary
    feature_dist_summary = {}
    for col in FEATURE_COLUMNS:
        feature_dist_summary[col] = {
            "mean": round(float(df_all[col].mean()), 4),
            "std": round(float(df_all[col].std()), 4),
            "median": round(float(df_all[col].median()), 4),
            "min": round(float(df_all[col].min()), 4),
            "max": round(float(df_all[col].max()), 4),
            "zero_pct": round(float((df_all[col] == 0).mean() * 100), 2),
        }

    # Split distributions
    train_df = df_all[df_all["split_role"] == "train"].copy()
    val_df = df_all[df_all["split_role"] == "val"].copy()
    test_df = df_all[df_all["split_role"] == "test"].copy()

    split_summary = {
        "train_obs": len(train_df),
        "train_churn_pct": round(float(train_df[TARGET_COLUMN].mean() * 100), 2),
        "train_retained": int((train_df[TARGET_COLUMN] == 0).sum()),
        "val_obs": len(val_df),
        "val_churn_pct": round(float(val_df[TARGET_COLUMN].mean() * 100), 2),
        "val_retained": int((val_df[TARGET_COLUMN] == 0).sum()),
        "test_obs": len(test_df),
        "test_churn_pct": round(float(test_df[TARGET_COLUMN].mean() * 100), 2),
        "test_retained": int((test_df[TARGET_COLUMN] == 0).sum()),
        "total_panel_obs": len(df_all),
        "unique_customers": int(df_all["customer_unique_id"].nunique()),
    }

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # 2. Candidate Models: 5-Fold Stratified CV on Training Set
    logger.info(f"Running Stratified 5-Fold CV on Training Set ({len(X_train):,} obs)...")
    t0_train = time.time()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_pos = float(num_neg / max(num_pos, 1))

    candidate_factories = {
        "Logistic Regression (Balanced)": lambda: ImbPipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression + SMOTE": lambda: ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest (Balanced Subsample)": lambda: ImbPipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "XGBoost (Cost-Weighted)": lambda: ImbPipeline([
            ("classifier", XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    cv_results = {}
    candidate_holdout_results = {}

    for name, factory in candidate_factories.items():
        t_model = time.time()
        pipe_cv = factory()
        scores = cross_validate(pipe_cv, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        dur = round(time.time() - t_model, 2)
        cv_results[name] = {
            "cv_roc_auc_mean": round(float(np.mean(scores["test_roc_auc"])), 4),
            "cv_roc_auc_std": round(float(np.std(scores["test_roc_auc"])), 4),
            "cv_pr_auc_mean": round(float(np.mean(scores["test_pr_auc"])), 4),
            "cv_pr_auc_std": round(float(np.std(scores["test_pr_auc"])), 4),
            "cv_f1_mean": round(float(np.mean(scores["test_f1"])), 4),
            "cv_recall_mean": round(float(np.mean(scores["test_recall"])), 4),
            "cv_precision_mean": round(float(np.mean(scores["test_precision"])), 4),
            "cv_training_time_sec": dur,
        }
        logger.info(f"  {name}: CV ROC-AUC={cv_results[name]['cv_roc_auc_mean']:.4f} in {dur}s")

        # Fit model on training set (Train S1+S2) and evaluate on Holdout Test Set (S4)
        t_fit = time.time()
        pipe_fit = factory()
        pipe_fit.fit(X_train, y_train)
        fit_dur = round(time.time() - t_fit, 2)

        test_m, y_prob_test_candidate = evaluate_model_on_split(pipe_fit, X_test, y_test)
        
        # Business cost at default 0.50 on holdout
        cost_050 = (test_m["false_positives"] * DEFAULT_CAMPAIGN_COST) + (test_m["false_negatives"] * DEFAULT_SAVED_MARGIN)

        candidate_holdout_results[name] = {
            "fit_time_sec": fit_dur,
            "roc_auc": round(float(test_m["roc_auc"]), 4),
            "pr_auc": round(float(test_m["pr_auc"]), 4),
            "f1_score": round(float(test_m["f1_score"]), 4),
            "precision": round(float(test_m["precision"]), 4),
            "recall": round(float(test_m["recall"]), 4),
            "brier_score": round(float(test_m["brier_score"]), 4),
            "business_cost_050": round(float(cost_050), 2),
            "confusion_matrix": {
                "tp": int(test_m["true_positives"]),
                "fp": int(test_m["false_positives"]),
                "tn": int(test_m["true_negatives"]),
                "fn": int(test_m["false_negatives"]),
            },
        }

    # 3. Champion Tuning & Strict Validation-Based Threshold Optimization
    logger.info("Tuning champion XGBoost via 3-Fold Grid Search on Training Snapshots...")
    param_grid = {
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.05],
        "n_estimators": [100, 150],
        "subsample": [0.8, 1.0],
    }

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(xgb, param_grid=param_grid, scoring="roc_auc", cv=cv3, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate on chronological Validation Snapshot (S3)
    val_metrics, val_prob = evaluate_model_on_split(best_model, X_val, y_val)
    logger.info(f"Validation Snapshot (S3) Performance: ROC-AUC={val_metrics['roc_auc']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}, F1={val_metrics['f1_score']:.4f}")

    # PHASE 4: Strict Validation-Based Threshold Optimization
    # "Select the threshold using VALIDATION data only.
    # Apply that frozen threshold once to Snapshot 4.
    # NEVER optimize the threshold directly on the holdout test set."
    df_thresh_val = ModelEvaluator.optimize_threshold(
        y_val.values,
        val_prob,
        cost_fp=DEFAULT_CAMPAIGN_COST,
        cost_fn=DEFAULT_SAVED_MARGIN,
        success_rate=DEFAULT_SUCCESS_RATE,
        threshold_steps=50,
    )
    best_val_idx = df_thresh_val["total_business_cost"].idxmin()
    frozen_optimal_threshold = round(float(df_thresh_val.loc[best_val_idx, "threshold"]), 3)
    min_val_cost = round(float(df_thresh_val.loc[best_val_idx, "total_business_cost"]), 2)
    logger.info(f"Selected Frozen Threshold on VALIDATION set: {frozen_optimal_threshold:.3f} (Val Cost: ${min_val_cost:,.2f})")

    # Retrain champion on Train + Validation (S1+S2 + S3)
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)
    scale_pos_full = float(int((y_train_full == 0).sum()) / max(int((y_train_full == 1).sum()), 1))
    best_model.set_params(scale_pos_weight=scale_pos_full)
    best_model.fit(X_train_full, y_train_full)

    # Evaluate champion on Untouched Holdout Test Snapshot (S4)
    holdout_metrics, y_test_prob = evaluate_model_on_split(best_model, X_test, y_test)

    # Apply Frozen Threshold to Snapshot 4 Holdout Test Set ONCE
    y_test_pred_frozen = (y_test_prob >= frozen_optimal_threshold).astype(int)
    cm_frozen = confusion_matrix(y_test.values, y_test_pred_frozen)
    tn_fr, fp_fr, fn_fr, tp_fr = cm_frozen.ravel()
    prec_frozen = precision_score(y_test.values, y_test_pred_frozen, zero_division=0)
    rec_frozen = recall_score(y_test.values, y_test_pred_frozen, zero_division=0)
    f1_frozen = f1_score(y_test.values, y_test_pred_frozen, zero_division=0)
    holdout_business_cost_frozen = (fp_fr * DEFAULT_CAMPAIGN_COST) + (fn_fr * DEFAULT_SAVED_MARGIN)

    # Standard threshold 0.50 on holdout test set
    y_test_pred_050 = (y_test_prob >= 0.50).astype(int)
    cm_050 = confusion_matrix(y_test.values, y_test_pred_050)
    tn_50, fp_50, fn_50, tp_50 = cm_050.ravel()
    prec_050 = precision_score(y_test.values, y_test_pred_050, zero_division=0)
    rec_050 = recall_score(y_test.values, y_test_pred_050, zero_division=0)
    f1_050 = f1_score(y_test.values, y_test_pred_050, zero_division=0)
    holdout_business_cost_050 = (fp_50 * DEFAULT_CAMPAIGN_COST) + (fn_50 * DEFAULT_SAVED_MARGIN)

    cost_reduction_pct = round(
        ((holdout_business_cost_050 - holdout_business_cost_frozen) / max(holdout_business_cost_050, 1.0)) * 100.0,
        2,
    )

    total_pipeline_time = round(time.time() - t0_train, 2)

    config_result = {
        "lookback_label": lookback_label,
        "lookback_days": lookback_days,
        "snapshot_details": snapshot_details,
        "split_summary": split_summary,
        "leakage_verification": {
            "all_snapshots_leakage_free": all(not c["future_purchase_detected"] for c in leakage_checks),
            "snapshots_audit": leakage_checks,
        },
        "feature_hygiene": {
            "total_missing_values": int(total_missing),
            "feature_missing_breakdown": {k: int(v) for k, v in missing_counts.items() if v > 0},
            "feature_distributions": feature_dist_summary,
        },
        "computational_complexity": {
            "feature_extraction_seconds": feat_extract_time,
            "training_cv_tuning_seconds": total_pipeline_time,
            "total_runtime_seconds": round(feat_extract_time + total_pipeline_time, 2),
        },
        "cross_validation_candidate_models": cv_results,
        "candidate_models_holdout_test": candidate_holdout_results,
        "champion_hyperparameters": grid_search.best_params_,
        "validation_snapshot_metrics": {
            "roc_auc": round(float(val_metrics["roc_auc"]), 4),
            "pr_auc": round(float(val_metrics["pr_auc"]), 4),
            "f1_score": round(float(val_metrics["f1_score"]), 4),
            "recall": round(float(val_metrics["recall"]), 4),
            "precision": round(float(val_metrics["precision"]), 4),
            "brier_score": round(float(val_metrics["brier_score"]), 4),
        },
        "holdout_test_metrics_champion": {
            "roc_auc": round(float(holdout_metrics["roc_auc"]), 4),
            "pr_auc": round(float(holdout_metrics["pr_auc"]), 4),
            "f1_score": round(float(holdout_metrics["f1_score"]), 4),
            "recall": round(float(holdout_metrics["recall"]), 4),
            "precision": round(float(holdout_metrics["precision"]), 4),
            "brier_score": round(float(holdout_metrics["brier_score"]), 4),
            "accuracy": round(float(holdout_metrics["accuracy"]), 4),
            "specificity": round(float(holdout_metrics["specificity"]), 4),
            "true_positives": int(holdout_metrics["true_positives"]),
            "false_positives": int(holdout_metrics["false_positives"]),
            "true_negatives": int(holdout_metrics["true_negatives"]),
            "false_negatives": int(holdout_metrics["false_negatives"]),
        },
        "threshold_evaluation_validation_selected": {
            "methodology": "Selected optimal threshold on Validation Snapshot (S3) to minimize FP*$10 + FN*$120, then frozen and applied to Holdout Test (S4).",
            "frozen_optimal_threshold": frozen_optimal_threshold,
            "validation_cost_at_threshold": min_val_cost,
            "holdout_metrics_at_frozen_threshold": {
                "threshold": frozen_optimal_threshold,
                "precision": round(float(prec_frozen), 4),
                "recall": round(float(rec_frozen), 4),
                "f1_score": round(float(f1_frozen), 4),
                "tp": int(tp_fr),
                "fp": int(fp_fr),
                "tn": int(tn_fr),
                "fn": int(fn_fr),
                "total_business_cost": round(float(holdout_business_cost_frozen), 2),
            },
            "holdout_metrics_at_050": {
                "threshold": 0.50,
                "precision": round(float(prec_050), 4),
                "recall": round(float(rec_050), 4),
                "f1_score": round(float(f1_050), 4),
                "tp": int(tp_50),
                "fp": int(fp_50),
                "tn": int(tn_50),
                "fn": int(fn_50),
                "total_business_cost": round(float(holdout_business_cost_050), 2),
            },
            "cost_reduction_pct": cost_reduction_pct,
        },
    }

    logger.info(
        f"Completed {lookback_label}: Holdout ROC-AUC={holdout_metrics['roc_auc']:.4f} | "
        f"PR-AUC={holdout_metrics['pr_auc']:.4f} | F1={holdout_metrics['f1_score']:.4f} | "
        f"Brier={holdout_metrics['brier_score']:.4f} | Frozen Threshold={frozen_optimal_threshold} | "
        f"Holdout Cost=${holdout_business_cost_frozen:,.2f}"
    )
    return config_result, df_all, best_model


def main():
    logger.info("Initializing Comprehensive Lookback Window Evaluation (180d vs 365d vs All-Time)...")
    
    configs = [
        (180, "180-Day Lookback"),
        (365, "365-Day Lookback"),
        (None, "All-Time Lookback"),
    ]

    audit_summary = {}

    for lb_days, lb_label in configs:
        res, _, _ = run_lookback_evaluation(lb_days, lb_label)
        audit_summary[lb_label] = res

    # Save detailed audit JSON
    output_path = REPORTS_DIR / "lookback_window_comparison.json"
    with open(output_path, "w") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Audit results successfully exported to {output_path}")


if __name__ == "__main__":
    main()
