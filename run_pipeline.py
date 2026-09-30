"""
Master Orchestration Script for E-Commerce Customer Intelligence & Churn Platform.
Executes the end-to-end reproducible pipeline:
1. Raw Data Generation (Realistic multi-table schema with controlled anomalies)
2. Data Validation, Cleaning & SQLite Database Loading
3. Advanced SQL Analytics & Cohort Retention Reporting
4. Zero-Leakage Feature Engineering & Target Extraction
5. RFM Scoring & Unsupervised K-Means Customer Segmentation
6. Supervised ML Training, 5-Fold Stratified CV Comparison & Tuning
7. Holdout Test Set Evaluation & Cost-Benefit Threshold Optimization
8. Batch Inference, SHAP Local Explanations & Prescriptive Recommendation Generation
"""

import sys
import time
from pathlib import Path

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import get_logger
from src.data_generator import generate_raw_ecommerce_data
from src.data_processing import DataPipeline
from src.sql_analytics import SQLAnalyticsEngine
from src.feature_engineering import FeatureEngineer
from src.segmentation import CustomerSegmentation
from src.train import ChurnModelTrainer
from src.predict import PredictionPipeline

logger = get_logger("MasterPipeline")


def run_all() -> None:
    start_time = time.time()
    logger.info("=====================================================================")
    logger.info("  STARTING END-TO-END E-COMMERCE INTELLIGENCE & CHURN ML PIPELINE   ")
    logger.info("=====================================================================")

    # Step 1: Raw Data Generation
    logger.info("\n--- STEP 1: GENERATING RAW E-COMMERCE RELATIONAL DATASET ---")
    generate_raw_ecommerce_data(num_customers=10000, num_products=500)

    # Step 2: Data Cleaning & SQL Loading
    logger.info("\n--- STEP 2: DATA VALIDATION, CLEANING & DATABASE LOADING ---")
    data_proc = DataPipeline()
    quality_report = data_proc.run_pipeline()

    # Step 3: SQL Business Analytics
    logger.info("\n--- STEP 3: ADVANCED SQL BUSINESS & COHORT ANALYTICS ---")
    sql_engine = SQLAnalyticsEngine()
    sql_reports = sql_engine.run_all_and_save_reports()

    # Step 4: Zero-Leakage Feature Engineering
    logger.info("\n--- STEP 4: FEATURE ENGINEERING & CHURN WINDOW EXTRACTION ---")
    fe = FeatureEngineer()
    df_features = fe.build_churn_dataset()

    # Step 5: RFM & K-Means Segmentation
    logger.info("\n--- STEP 5: RFM SCORING & UNSUPERVISED CUSTOMER SEGMENTATION ---")
    seg = CustomerSegmentation()
    df_rfm = seg.calculate_rfm(df_features)
    df_segmented, cluster_profiles = seg.fit_kmeans(df_rfm, n_clusters=4)

    # Step 6 & 7: Model Comparison, Tuning & Evaluation
    logger.info("\n--- STEP 6 & 7: SUPERVISED ML COMPARISON, TUNING & THRESHOLD OPTIMIZATION ---")
    trainer = ChurnModelTrainer()
    train_results = trainer.execute_training_pipeline()

    # Step 8: Batch Predictions, SHAP Explainability & Recommendations
    logger.info("\n--- STEP 8: INFERENCE, SHAP EXPLAINABILITY & RECOMMENDATIONS ---")
    predictor = PredictionPipeline()
    master_intel = predictor.generate_batch_intelligence_table()

    elapsed = time.time() - start_time
    logger.info("=====================================================================")
    logger.info(f"  PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS!         ")
    logger.info("=====================================================================")


if __name__ == "__main__":
    run_all()
