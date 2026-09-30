"""
Master Orchestration Script for E-Commerce Customer Intelligence & Churn Platform.
Executes the production pipeline for Brazilian E-Commerce (Olist) data warehouse:
1. Raw Data Ingestion & Standardization (9 Olist Tables)
2. Referential Integrity & Relational Validation (0 Foreign Key Orphans)
3. Database Loading (PostgreSQL Target / SQLite Fallback)
4. Advanced SQL Business Analytics & Cohort Retention Reporting
5. RFM Calculation on customer_unique_id
6. Unsupervised Customer Segmentation (K-Means, k=4)
7. Point-in-Time Temporal Feature Engineering (Delivery Clamping)
8. Multi-Snapshot Supervised Modeling & Holdout Evaluation
9. SHAP Explainability & Feature Attribution
10. Batch Risk Scoring & CRM Table Generation
11. Publication Figure Generation
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
