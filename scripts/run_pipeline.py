"""
Master End-to-End Orchestrator for Real Olist E-Commerce Customer Intelligence & Churn Platform.
Executes the full pipeline sequentially and reproducibly:
1. Raw Data Cleaning & Standardization (Olist Brazilian E-Commerce)
2. Referential Integrity & Relational Validation
3. Database Ingestion (PostgreSQL with SQLite Fallback)
4. Advanced SQL Analytics & Cohort Reporting
5. RFM Calculation on customer_unique_id
6. Unsupervised K-Means Customer Segmentation (k=2..8 evaluation)
7. Leakage-Safe Feature Engineering & Churn Target Extraction
8. Supervised Model Training (5-Fold Stratified CV, SMOTE, XGBoost Tuning, Holdout Evaluation)
9. SHAP Explainability & Feature Importance Attribution
10. Batch Customer Risk Intelligence & CRM Export Generation
11. Publication-Quality Static Figure Generation
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from src.utils.logger import get_logger
from src.data.ingestion.clean_olist import OlistDataCleaner
from src.data.ingestion.validate_olist import OlistDataValidator
from scripts.load_postgres import OlistPostgresLoader
from src.analytics.sql_analytics import SQLAnalyticsEngine
from src.features.rfm import RFMCalculator
from src.models.segmentation import OlistSegmentationEngine
from src.features.feature_engineering import OlistFeatureEngineer
from src.models.train import OlistChurnTrainer
from src.models.explain import OlistModelExplainer
from src.models.predict import OlistPredictor
from src.generate_figures import generate_all_figures
from src.utils.config import PROCESSED_DATA_DIR

logger = get_logger("MasterPipeline")


def run_pipeline(skip_ingestion_if_exists: bool = False) -> None:
    """Executes the entire end-to-end data science and machine learning pipeline."""
    t_start = time.time()
    logger.info("================================================================================")
    logger.info("  STARTING END-TO-END E-COMMERCE INTELLIGENCE & CHURN ML PIPELINE (OLIST DATA)  ")
    logger.info("================================================================================")

    # -------------------------------------------------------------
    # Step 1: Clean & Standardize Raw Olist CSVs
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 1: CLEANING AND STANDARDIZING RAW OLIST DATASETS...")
    cleaner = OlistDataCleaner()
    cleaned_tables = cleaner.clean_all()
    logger.info(f"Step 1 Complete: {len(cleaned_tables)} tables cleaned and written to {PROCESSED_DATA_DIR.name}.")

    # -------------------------------------------------------------
    # Step 2: Referential Integrity Validation
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 2: VALIDATING REFERENTIAL INTEGRITY AND PRIMARY KEYS...")
    validator = OlistDataValidator()
    val_report = validator.validate_all()
    if val_report["validation_status"] != "PASSED":
        logger.error(f"Validation failed: {val_report['critical_errors']}")
        raise ValueError("Data validation checks failed.")
    logger.info(f"Step 2 Complete: All referential integrity checks PASSED (0 orphan records).")

    # -------------------------------------------------------------
    # Step 3: Relational Warehouse Ingestion
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 3: LOADING CLEANED DATA INTO RELATIONAL WAREHOUSE...")
    loader = OlistPostgresLoader()
    db_results = loader.run_load()
    logger.info(f"Step 3 Complete: Relational warehouse populated ({db_results.get('total_rows_loaded', 0):,} total rows).")

    # -------------------------------------------------------------
    # Step 4: Advanced SQL Analytics & Cohort Reporting
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 4: EXECUTING ADVANCED SQL ANALYTICS & COHORT QUERIES...")
    sql_engine = SQLAnalyticsEngine()
    sql_reports = sql_engine.run_all_and_save_reports()
    logger.info(f"Step 4 Complete: {len(sql_reports)} analytical reports generated and saved to reports/.")

    # -------------------------------------------------------------
    # Step 5: RFM Calculation on customer_unique_id
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 5: COMPUTING RFM METRICS (PRE-CUTOFF SNAPSHOT)...")
    df_orders = pd.read_csv(PROCESSED_DATA_DIR / "fact_orders.csv")
    df_items = pd.read_csv(PROCESSED_DATA_DIR / "fact_order_items.csv")
    df_cust = pd.read_csv(PROCESSED_DATA_DIR / "dim_customers.csv")
    rfm_calc = RFMCalculator()
    df_rfm = rfm_calc.compute_rfm(df_orders, df_items, df_cust)
    logger.info(f"Step 5 Complete: RFM scores computed for {len(df_rfm):,} unique customers.")

    # -------------------------------------------------------------
    # Step 6: Customer Segmentation (K-Means)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 6: EVALUATING K-MEANS & SEGMENTING CUSTOMERS...")
    seg_engine = OlistSegmentationEngine()
    df_seg, cluster_summary = seg_engine.fit_segmentation(df_rfm, n_clusters=4)
    logger.info(f"Step 6 Complete: Customer segmentation finalized into 4 clusters.")

    # -------------------------------------------------------------
    # Step 7: Leakage-Safe Feature Engineering & Churn Target
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 7: EXTRACTING LEAKAGE-SAFE PREDICTIVE FEATURES & TARGET...")
    fe_engine = OlistFeatureEngineer()
    df_features = fe_engine.build_churn_feature_matrix()
    logger.info(f"Step 7 Complete: Feature matrix built for {len(df_features):,} customers.")

    # -------------------------------------------------------------
    # Step 8: Supervised Model Training & Cross-Validation
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 8: SUPERVISED TRAINING, 5-FOLD CV & HYPERPARAMETER TUNING...")
    trainer = OlistChurnTrainer()
    train_results = trainer.execute_training_pipeline()
    logger.info("Step 8 Complete: Champion model tuned, validated on holdout set, and threshold optimized.")

    # -------------------------------------------------------------
    # Step 9: Model Explainability (SHAP)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 9: COMPUTING GLOBAL SHAP FEATURE IMPORTANCE...")
    explainer = OlistModelExplainer()
    importance = explainer.compute_global_importance(df_features)
    logger.info("Step 9 Complete: SHAP feature importances computed and persisted.")

    # -------------------------------------------------------------
    # Step 10: Batch Risk Scoring & CRM Table
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 10: GENERATING BATCH CUSTOMER RISK SCORING TABLE...")
    predictor = OlistPredictor()
    risk_table = predictor.generate_batch_risk_table()
    logger.info(f"Step 10 Complete: Batch risk table generated for {len(risk_table):,} customers.")

    # -------------------------------------------------------------
    # Step 11: Static Figure Generation
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 11: GENERATING PUBLICATION FIGURES FOR REPORTS...")
    generate_all_figures()
    logger.info("Step 11 Complete: All figures rendered in reports/figures/.")

    t_elapsed = time.time() - t_start
    logger.info("================================================================================")
    logger.info(f"  OLIST ML PIPELINE COMPLETED SUCCESSFULLY IN {t_elapsed:.1f} SECONDS!          ")
    logger.info("================================================================================")


if __name__ == "__main__":
    run_pipeline()
