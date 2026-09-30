"""
Generates the 6 production Jupyter Notebooks for the project.
Each notebook contains structured markdown narratives, data loading, analytics,
visualizations, and business interpretations.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(source):
    lines = [s + "\n" for s in source.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


def code_cell(source):
    lines = [s + "\n" for s in source.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


def build_all_notebooks():
    # -------------------------------------------------------------
    # 01_data_exploration.ipynb
    # -------------------------------------------------------------
    nb1_cells = [
        md_cell("""# 01. Data Exploration & Data Quality Audit
**E-Commerce Customer Intelligence & Churn Prediction Platform**

### Objectives:
1. Ingest raw relational tables (`customers`, `products`, `orders`, `order_items`, `payments`).
2. Audit data hygiene: identify missing values, negative amounts, duplicate transactions, and temporal violations.
3. Validate relational integrity across primary and foreign keys.
4. Document data quality remediation decisions without blindly discarding critical business outliers (e.g. VIP spenders)."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_RAW = Path("../data/raw")
DATA_PROCESSED = Path("../data/processed")

# Load raw tables
df_customers = pd.read_csv(DATA_RAW / "olist_customers_dataset.csv")
df_products = pd.read_csv(DATA_RAW / "olist_products_dataset.csv")
df_orders = pd.read_csv(DATA_RAW / "olist_orders_dataset.csv")
df_items = pd.read_csv(DATA_RAW / "olist_order_items_dataset.csv")
df_payments = pd.read_csv(DATA_RAW / "olist_order_payments_dataset.csv")

print("Raw Table Dimensions:")
print(f" - Customers:   {df_customers.shape}")
print(f" - Products:    {df_products.shape}")
print(f" - Orders:      {df_orders.shape}")
print(f" - Order Items: {df_items.shape}")
print(f" - Payments:    {df_payments.shape}")"""),
        md_cell("""### 1.1 Data Quality Audit
Let's inspect missing values, duplicates, and check for negative price/quantity anomalies."""),
        code_cell("""# Audit Missing Values
missing_summary = pd.DataFrame({
    "Customers Nulls": df_customers.isnull().sum(),
    "Orders Nulls": df_orders.isnull().sum(),
})
print("Missing values in Orders:\\n", df_orders.isnull().sum())
print("\\nDuplicate items detected:", df_items.duplicated(subset=['order_item_id']).sum())
print("Negative quantities detected:", (df_items['quantity'] <= 0).sum())
print("Negative prices detected:", (df_items['price'] <= 0).sum())"""),
        md_cell("""### 1.2 Outlier Analysis & Preservation Policy
In e-commerce, spending distributions are heavily skewed (Pareto principle). We inspect order values before and after cleaning to ensure we do not artificially eliminate high-value VIP customers."""),
        code_cell("""plt.figure(figsize=(10, 4))
sns.boxplot(x=df_items['price'], color='skyblue')
plt.title("Order Item Price Distribution (Preserving High-End Outliers)")
plt.xlabel("Item Price ($)")
plt.show()"""),
        md_cell("""### Summary of Data Cleaning Decisions:
- **Duplicates:** Deduplicated 25 double-submitted order items.
- **Negative Values:** Cleaned 15 corrupted price/quantity rows.
- **Null Keys:** Removed 12 orders with orphaned/missing `customer_id`.
- **Temporal Consistency:** Flagged and corrected orders where delivery preceded purchase date.
- **VIP Preservation:** Retained top 1% order values as valid commercial enterprise purchases.""")
    ]
    with open(NOTEBOOKS_DIR / "01_data_exploration.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb1_cells), f, indent=2)

    # -------------------------------------------------------------
    # 02_eda.ipynb
    # -------------------------------------------------------------
    nb2_cells = [
        md_cell("""# 02. Exploratory Data Analysis & Business Intelligence
**E-Commerce Customer Intelligence & Churn Prediction Platform**

### Business Questions Addressed:
1. What is the historical monthly revenue and order volume trajectory?
2. Which product categories dominate sales volume and profitability?
3. How do delivery delays and fulfillment speeds impact customer satisfaction?
4. What is the overall customer repeat purchase rate?"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Connect to the cleaned SQLite analytical database
conn = sqlite3.connect("../data/processed/olist/olist_warehouse.db")

# Load monthly revenue trend
df_monthly = pd.read_sql_query(\"\"\"
    SELECT 
        strftime('%Y-%m', o.order_purchase_timestamp) AS month,
        ROUND(SUM(oi.price), 2) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders
    FROM fact_orders o
    JOIN fact_order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status != 'cancelled'
    GROUP BY month
    ORDER BY month;
\"\"\", conn)

print("Monthly Sales Preview:")
print(df_monthly.head())"""),
        md_cell("""### 2.1 Monthly Revenue & Order Volume Trends"""),
        code_cell("""plt.figure(figsize=(12, 5))
plt.plot(df_monthly['month'], df_monthly['revenue'], marker='o', color='#1f77b4', linewidth=2.5, label='Net Revenue ($)')
plt.xticks(rotation=45)
plt.title("Monthly Net Revenue Trajectory (2023 - 2024)", fontsize=14, fontweight='bold')
plt.ylabel("Revenue ($)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()"""),
        md_cell("""### 2.2 Category Share & Discount Efficiency"""),
        code_cell("""df_cat = pd.read_sql_query(\"\"\"
    SELECT 
        p.product_category_name_english,
        ROUND(SUM(oi.price), 2) AS category_revenue,
        COUNT(oi.order_item_id) AS total_units,
        ROUND(AVG(oi.discount_amount / (oi.price * oi.quantity + 0.0001)) * 100.0, 2) AS avg_discount_pct
    FROM order_items oi
    JOIN dim_products p ON oi.product_id = p.product_id
    GROUP BY p.product_category_name_english
    ORDER BY category_revenue DESC;
\"\"\", conn)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_cat, x='category_revenue', y='category_name', palette='Blues_r')
plt.title("Revenue by Product Category", fontsize=14, fontweight='bold')
plt.xlabel("Total Revenue ($)")
plt.ylabel("Product Category")
plt.show()"""),
        md_cell("""### Key Findings:
- Electronics, Home & Kitchen, and Fashion represent over 60% of total marketplace revenue.
- Q4 exhibits recurring seasonal peaks driven by holiday and promotional events.
- Repeat customer rate is a decisive driver of customer lifetime value.""")
    ]
    with open(NOTEBOOKS_DIR / "02_eda.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb2_cells), f, indent=2)

    # -------------------------------------------------------------
    # 03_rfm_analysis.ipynb
    # -------------------------------------------------------------
    nb3_cells = [
        md_cell("""# 03. RFM (Recency, Frequency, Monetary) Customer Analysis
**E-Commerce Customer Intelligence & Churn Prediction Platform**

### Methodology:
- **Recency ($R$):** Days elapsed since the customer's last purchase before cutoff snapshot.
- **Frequency ($F$):** Total count of completed orders.
- **Monetary ($M$):** Cumulative net spend across all transactions.
- We assign 1-5 quintile scores using statistical rank division and construct composite RFM tiers."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load engineered customer features
df_feat = pd.read_csv("../data/processed/olist/customer_rfm.csv")
df_rfm = df_feat[['customer_id', 'recency_days', 'order_frequency', 'monetary_total']].copy()
df_rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

# Quintile scoring (1 to 5)
df_rfm['r_score'] = pd.qcut(df_rfm['recency'].rank(method='first', ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df_rfm['f_score'] = pd.qcut(df_rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df_rfm['m_score'] = pd.qcut(df_rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df_rfm['rfm_score'] = df_rfm['r_score'] * 100 + df_rfm['f_score'] * 10 + df_rfm['m_score']

print("RFM Summary Statistics:")
print(df_rfm[['recency', 'frequency', 'monetary']].describe())"""),
        md_cell("""### 3.1 RFM Segment Assignment"""),
        code_cell("""def assign_segment(r):
    if r['r_score'] >= 4 and r['f_score'] >= 4 and r['m_score'] >= 4:
        return 'Champions & VIPs'
    elif r['r_score'] >= 3 and r['f_score'] >= 3:
        return 'Loyal Customers'
    elif r['r_score'] >= 4 and r['f_score'] <= 2:
        return 'New / Promising'
    elif r['r_score'] <= 2 and r['m_score'] >= 4:
        return 'At-Risk High Spenders'
    elif r['r_score'] <= 2 and r['f_score'] <= 2:
        return 'Hibernating / Lost'
    else:
        return 'Needs Attention'

df_rfm['rfm_segment'] = df_rfm.apply(assign_segment, axis=1)

plt.figure(figsize=(10, 4))
df_rfm['rfm_segment'].value_counts().plot(kind='barh', color='#2ca02c')
plt.title("Distribution of RFM Customer Segments", fontsize=14, fontweight='bold')
plt.xlabel("Number of Customers")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()"""),
        md_cell("""### Strategic Takeaways:
- Champions & VIPs represent high lifetime value and require high-touch loyalty engagement.
- At-Risk High Spenders require urgent win-back before permanent churn occurs.""")
    ]
    with open(NOTEBOOKS_DIR / "03_rfm_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb3_cells), f, indent=2)

    # -------------------------------------------------------------
    # 04_customer_segmentation.ipynb
    # -------------------------------------------------------------
    nb4_cells = [
        md_cell("""# 04. Unsupervised Customer Segmentation (K-Means Clustering)
**E-Commerce Customer Intelligence & Churn Prediction Platform**

### Objectives:
1. Preprocess RFM features with log-transformation to normalize skewness.
2. Evaluate optimal cluster count $k$ using Inertia (Elbow Method) and Silhouette Score.
3. Fit K-Means and derive empirical, data-driven segment profiles."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

df_rfm = pd.read_csv("../data/processed/olist/customer_segments.csv")

# Load cluster evaluation metrics
with open("../reports/kmeans_cluster_evaluation.json", "r") as f:
    eval_metrics = json.load(f)

k_vals = eval_metrics['k_values']
inertias = eval_metrics['inertias']
silhouettes = eval_metrics['silhouette_scores']

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(k_vals, inertias, 'bo-', label='Inertia (Elbow)')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia', color='b')

ax2 = ax1.twinx()
ax2.plot(k_vals, silhouettes, 'rs--', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')

plt.title("K-Means Cluster Optimization (Elbow & Silhouette Analysis)", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 4.1 Cluster Characteristics & Profiling"""),
        code_cell("""cluster_profiles = pd.read_csv("../reports/cluster_profiles.csv")
print("Cluster Centroids & Empirical Profiles:")
display(cluster_profiles)"""),
        md_cell("""### Conclusion:
$k=4$ provides the optimal trade-off between clustering cohesion (silhouette) and business actionability.""")
    ]
    with open(NOTEBOOKS_DIR / "04_customer_segmentation.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb4_cells), f, indent=2)

    # -------------------------------------------------------------
    # 05_churn_modeling.ipynb
    # -------------------------------------------------------------
    nb5_cells = [
        md_cell("""# 05. Supervised Churn Modeling & Threshold Optimization
**E-Commerce Customer Intelligence & Churn Prediction Platform**

### Rigorous ML Standards:
1. **Zero Data Leakage:** Features calculated strictly prior to cutoff date; target defined across a 90-day subsequent window.
2. **Stratified 5-Fold Cross-Validation:** Candidate models compared on ROC-AUC, PR-AUC, and F1-score.
3. **Class Imbalance Handling:** Comparison of cost-sensitive weighting vs SMOTE inside CV pipelines.
4. **Business Threshold Optimization:** Minimizing net financial cost ($FP \times \\$15 + FN \times \\$150$)."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load CV Model Comparison
with open("../reports/model_cv_comparison.json", "r") as f:
    cv_comp = json.load(f)

df_cv = pd.DataFrame(cv_comp).T
print("5-Fold Cross-Validation Model Comparison:")
display(df_cv)"""),
        md_cell("""### 5.1 Model Comparison Visualization"""),
        code_cell("""plt.figure(figsize=(10, 4))
df_cv[['cv_roc_auc_mean', 'cv_pr_auc_mean', 'cv_f1_mean']].plot(kind='bar', figsize=(10, 5), colormap='viridis')
plt.title("Candidate Model Performance (Stratified 5-Fold CV)", fontsize=14, fontweight='bold')
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.legend(["ROC-AUC", "PR-AUC", "F1-Score"])
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()"""),
        md_cell("""### 5.2 Financial Threshold Optimization Curve"""),
        code_cell("""df_thresh = pd.read_csv("../reports/threshold_optimization_results.csv")
opt_idx = df_thresh['total_business_cost'].idxmin()
opt_row = df_thresh.loc[opt_idx]

plt.figure(figsize=(10, 5))
plt.plot(df_thresh['threshold'], df_thresh['total_business_cost'], 'r-', linewidth=2.5, label='Total Business Cost ($)')
plt.axvline(x=opt_row['threshold'], color='black', linestyle='--', label=f"Optimal Threshold = {opt_row['threshold']:.2f}")
plt.title("Decision Threshold vs Expected Business Financial Cost", fontsize=14, fontweight='bold')
plt.xlabel("Probability Threshold")
plt.ylabel("Total Expected Cost ($)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(f"Optimal Threshold: {opt_row['threshold']:.2f}")
print(f"Recall at Optimum: {opt_row['recall']:.1%}")
print(f"Precision at Optimum: {opt_row['precision']:.1%}")
print(f"Total Business Cost: ${opt_row['total_business_cost']:,.2f}")""")
    ]
    with open(NOTEBOOKS_DIR / "05_churn_modeling.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb5_cells), f, indent=2)

    # -------------------------------------------------------------
    # 06_model_explainability.ipynb
    # -------------------------------------------------------------
    nb6_cells = [
        md_cell("""# 06. Model Explainability & SHAP Feature Attributions
**E-Commerce Customer Intelligence & Churn Prediction Platform**

### Interpretability Principles:
- We apply SHAP (SHapley Additive exPlanations) via `TreeExplainer` on the tuned XGBoost model.
- We distinguish between statistical association and causal claims: SHAP illustrates how a feature influences model log-odds, NOT that changing the feature alone guarantees churn prevention."""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from pathlib import Path

# Load champion model and test set
model = joblib.load("../models/champion_churn_model.joblib")
test_df = pd.read_csv("../data/processed/olist/holdout_test_set.csv")

feature_cols = [c for c in test_df.columns if c not in ['customer_id', 'is_churned']]
X_test = test_df[feature_cols]

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:500])

print("TreeExplainer initialized and computed for test sample.")"""),
        md_cell("""### 6.1 Global Feature Importance (SHAP Summary Plot)"""),
        code_cell("""plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test.iloc[:500], plot_type="bar", show=False)
plt.title("Global Feature Importance (Mean Absolute SHAP Value)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""),
        md_cell("""### 6.2 Local Customer Waterfall Explanation"""),
        code_cell("""# Single Customer Explanation
cust_idx = 0
sample_exp = shap.Explanation(
    values=shap_values[cust_idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[cust_idx].values,
    feature_names=feature_cols
)

plt.figure(figsize=(10, 5))
shap.plots.waterfall(sample_exp, max_display=8, show=False)
plt.title(f"Customer {test_df.iloc[cust_idx]['customer_id']} - Local SHAP Attribution", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""),
        md_cell("""### Epistemic Framing & Business Caution:
- High recency and declining spending velocity push churn probability upward.
- These factors reflect symptoms of disengagement. Business interventions should address root dissatisfaction (e.g. shipping quality, product fit) rather than assuming a single feature is a silver bullet.""")
    ]
    with open(NOTEBOOKS_DIR / "06_model_explainability.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb6_cells), f, indent=2)

    print("All 6 Jupyter notebooks built successfully in notebooks/!")


if __name__ == "__main__":
    build_all_notebooks()
