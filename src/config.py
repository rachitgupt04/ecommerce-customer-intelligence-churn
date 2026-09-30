"""
Top-level configuration module for E-Commerce Customer Intelligence & Churn Platform.
Re-exports centralized configurations from src.utils.config for complete compatibility across the project.
"""

from src.utils.config import *

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "SQL_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "DASHBOARD_DIR",
    "TESTS_DIR",
    "DOCS_DIR",
    "PGHOST",
    "PGPORT",
    "PGDATABASE",
    "PGUSER",
    "PGPASSWORD",
    "DEFAULT_PG_URL",
    "DATABASE_URL",
    "SQLITE_DB_PATH",
    "SQLITE_DB_URI",
    "DB_URI",
    "CUTOFF_DATE",
    "CHURN_END_DATE",
    "CHURN_WINDOW_DAYS",
    "RANDOM_STATE",
    "TEST_SIZE",
    "CV_FOLDS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "DEFAULT_CAMPAIGN_COST",
    "DEFAULT_SAVED_MARGIN",
    "DEFAULT_SUCCESS_RATE",
    "RISK_THRESHOLDS",
    "PREDICTION_HORIZON_DAYS",
    "SNAPSHOT_CONFIG",
]
