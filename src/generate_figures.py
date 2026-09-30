"""
Generates and saves publication-grade static figures for README documentation and technical reports.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from src.config import (
    REPORTS_DIR,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FEATURE_COLUMNS,
)
from src.utils import get_logger, load_model

logger = get_logger("FigureGenerator")


def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Load Data
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "holdout_test_set.csv")
    model = load_model(MODELS_DIR / "champion_churn_model.joblib")
    
    y_test = test_df["is_churned"].values
    X_test = test_df[FEATURE_COLUMNS]
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # -------------------------------------------------------------
    # 1. ROC and Precision-Recall Curves
    # -------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2.5, label="XGBoost (AUC = 0.952)")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color="#2ca02c", lw=2.5, label="XGBoost (PR-AUC = 0.994)")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_pr_curve.png", dpi=300)
    plt.close()
    logger.info("Saved roc_pr_curve.png")

    # -------------------------------------------------------------
    # 2. Confusion Matrix Heatmap
    # -------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"])
    plt.title("Holdout Test Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    logger.info("Saved confusion_matrix.png")

    # -------------------------------------------------------------
    # 3. Decision Threshold vs Expected Business Cost
    # -------------------------------------------------------------
    df_thresh = pd.read_csv(REPORTS_DIR / "threshold_optimization_results.csv")
    opt_idx = df_thresh["total_business_cost"].idxmin()
    opt_row = df_thresh.loc[opt_idx]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_thresh["threshold"], df_thresh["total_business_cost"], color="#d62728", lw=2.5, label="Total Business Cost ($)")
    ax1.axvline(x=opt_row["threshold"], color="black", linestyle="--", label=f"Optimal Threshold = {opt_row['threshold']:.2f}")
    ax1.set_xlabel("Probability Decision Threshold", fontweight="bold")
    ax1.set_ylabel("Total Financial Cost ($)", color="#d62728", fontweight="bold")
    
    ax2 = ax1.twinx()
    ax2.plot(df_thresh["threshold"], df_thresh["recall"], color="#2ca02c", linestyle=":", lw=2, label="Recall")
    ax2.plot(df_thresh["threshold"], df_thresh["precision"], color="#1f77b4", linestyle="-.", lw=2, label="Precision")
    ax2.set_ylabel("Metric Score", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Cost-Benefit Threshold Optimization Curve", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_optimization.png", dpi=300)
    plt.close()
    logger.info("Saved threshold_optimization.png")

    # -------------------------------------------------------------
    # 4. Model CV Comparison Bar Chart
    # -------------------------------------------------------------
    with open(REPORTS_DIR / "model_cv_comparison.json", "r") as f:
        cv_data = json.load(f)
    df_cv = pd.DataFrame(cv_data).T.reset_index().rename(columns={"index": "Model"})

    plt.figure(figsize=(11, 5))
    bar_width = 0.25
    x = np.arange(len(df_cv))
    plt.bar(x - bar_width, df_cv["cv_roc_auc_mean"], width=bar_width, label="CV ROC-AUC", color="#1f77b4")
    plt.bar(x, df_cv["cv_pr_auc_mean"], width=bar_width, label="CV PR-AUC", color="#2ca02c")
    plt.bar(x + bar_width, df_cv["cv_f1_mean"], width=bar_width, label="CV F1-Score", color="#ff7f0e")
    plt.xticks(x, df_cv["Model"], rotation=15, ha="right")
    plt.ylabel("Cross-Validation Score")
    plt.ylim(0.85, 1.02)
    plt.title("5-Fold Cross-Validation Performance Across Candidate Models", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_cv_comparison.png", dpi=300)
    plt.close()
    logger.info("Saved model_cv_comparison.png")

    # -------------------------------------------------------------
    # 5. Customer Segments RFM Scatter
    # -------------------------------------------------------------
    df_seg = pd.read_csv(PROCESSED_DATA_DIR / "customer_segments.csv")
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=df_seg.sample(min(2000, len(df_seg)), random_state=42),
        x="recency",
        y="monetary",
        hue="cluster_segment",
        palette="tab10",
        alpha=0.7,
        s=40
    )
    plt.yscale("log")
    plt.title("Customer Segments: Recency vs Monetary Spend (Log Scale)", fontweight="bold")
    plt.xlabel("Recency (Days Since Last Order)")
    plt.ylabel("Total Net Spend ($ - Log Scale)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rfm_customer_clusters.png", dpi=300)
    plt.close()
    logger.info("Saved rfm_customer_clusters.png")

    # -------------------------------------------------------------
    # 6. Monthly Net Revenue & MoM Growth
    # -------------------------------------------------------------
    df_monthly = pd.read_csv(REPORTS_DIR / "monthly_revenue_trends.csv")
    plt.figure(figsize=(11, 5))
    plt.plot(df_monthly["order_month"], df_monthly["net_revenue"], marker="o", lw=2.5, color="#1f77b4")
    plt.xticks(rotation=45)
    plt.title("E-Commerce Monthly Net Revenue Trajectory", fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Net Revenue ($)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_sales_trend.png", dpi=300)
    plt.close()
    logger.info("Saved monthly_sales_trend.png")

    # -------------------------------------------------------------
    # 7. SHAP Global Feature Importance
    # -------------------------------------------------------------
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test.iloc[:500])
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_test.iloc[:500], plot_type="bar", show=False)
        plt.title("SHAP Global Feature Importance (Tuned XGBoost)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary_plot.png", dpi=300)
        plt.close()
        logger.info("Saved shap_summary_plot.png")
    except Exception as e:
        logger.warning(f"Could not generate SHAP plot: {e}")

    logger.info(f"All figures generated and saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
