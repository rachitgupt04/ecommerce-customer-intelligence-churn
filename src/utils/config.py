"""
Centralized Configuration for Real Olist E-Commerce Analytics & ML Platform.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "olist"
SQL_DIR = PROJECT_ROOT / "sql"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Database Configuration
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE", "ecommerce_olist")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")

# Default PostgreSQL URI with psycopg2 driver
DEFAULT_PG_URL = f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_PG_URL)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# Local Relational Warehouse (SQLite fallback / local standalone)
SQLITE_DB_PATH = PROCESSED_DATA_DIR / "olist_warehouse.db"
SQLITE_DB_URI = f"sqlite:///{SQLITE_DB_PATH.as_posix()}"

# Temporal Churn Framework for Olist
# Active operational span: 2017-01-01 to 2018-08-31
# Observation Cutoff: 2018-05-01 (All features computed strictly BEFORE this date)
CUTOFF_DATE = "2018-05-01"

# Performance Outcome Window: 2018-05-01 to 2018-08-31 (122 days / ~4 months)
# A customer active prior to cutoff is Churned (1) if 0 orders in window, Retained (0) if >= 1 order
CHURN_END_DATE = "2018-08-31"
CHURN_WINDOW_DAYS = 122

# ML Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Predictive Feature Columns (Leakage-Safe)
FEATURE_COLUMNS = [
    "recency_days",
    "order_frequency",
    "monetary_total",
    "avg_order_value",
    "total_quantity",
    "customer_tenure_days",
    "avg_freight_value",
    "freight_ratio",
    "category_diversity",
    "avg_review_score",
    "cancelled_orders_cnt",
    "credit_card_share",
    "avg_installments",
    "last_30d_orders",
    "last_60d_orders",
    "spending_trend_velocity",
    "avg_delivery_delay_days",
]

TARGET_COLUMN = "is_churned"

# Business Scenario / Threshold Optimization Defaults
DEFAULT_CAMPAIGN_COST = 10.0   # $10 retention incentive
DEFAULT_SAVED_MARGIN = 120.0   # $120 gross margin contribution saved per retained customer
DEFAULT_SUCCESS_RATE = 0.20    # 20% campaign conversion success

# Database URI alias for backward compatibility
DB_URI = SQLITE_DB_URI

# Operational Churn Risk Tiers
RISK_THRESHOLDS = {
    "HIGH": 0.70,
    "MEDIUM": 0.40,
    "LOW": 0.00,
}

# Multi-Snapshot Temporal Panel Settings
PREDICTION_HORIZON_DAYS = 90

SNAPSHOT_CONFIG = [
    {"name": "Snapshot 1", "date": "2017-09-01", "split": "train", "horizon_days": 90},
    {"name": "Snapshot 2", "date": "2017-12-01", "split": "train", "horizon_days": 90},
    {"name": "Snapshot 3", "date": "2018-03-01", "split": "val", "horizon_days": 90},
    {"name": "Snapshot 4", "date": "2018-06-01", "split": "test", "horizon_days": 90},
]

