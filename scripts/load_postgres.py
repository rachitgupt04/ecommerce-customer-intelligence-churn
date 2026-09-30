"""
PostgreSQL Database Schema Initialization & Data Loader.
Connects to PostgreSQL using environment variables, creates the star-schema tables,
creates performance indexes, loads cleaned Olist datasets, and verifies foreign key integrity.
Includes an automated relational warehouse fallback option (SQLite) if PostgreSQL service is offline.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
import logging

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.ingestion.validate_olist import OlistDataValidator

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("LoadPostgres")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(handler)


def get_db_engine():
    """
    Builds SQLAlchemy engine using environment variables.
    Checks DATABASE_URL first, then individual PG* variables.
    Falls back to SQLite relational warehouse if PostgreSQL is unavailable.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        pghost = os.getenv("PGHOST", "localhost")
        pgport = os.getenv("PGPORT", "5432")
        pgdatabase = os.getenv("PGDATABASE", "ecommerce_olist")
        pguser = os.getenv("PGUSER", "postgres")
        pgpassword = os.getenv("PGPASSWORD", "postgres")
        db_url = f"postgresql+psycopg2://{pguser}:{pgpassword}@{pghost}:{pgport}/{pgdatabase}"
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    try:
        engine = create_engine(db_url, echo=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL database: {engine.url.render_as_string(hide_password=True)}")
        return engine, "postgresql"
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL ({e}).")
        fallback_db_path = PROJECT_ROOT / "data" / "processed" / "olist" / "olist_warehouse.db"
        fallback_url = f"sqlite:///{fallback_db_path.as_posix()}"
        logger.info(f"Using local relational warehouse database: {fallback_db_path}")
        engine = create_engine(fallback_url, echo=False)
        return engine, "sqlite"


def create_schema(engine, db_type: str = "postgresql"):
    """Creates the 5 dimension tables and 4 fact tables with constraints and indexes."""
    logger.info(f"Creating relational schema for database type: {db_type}...")

    # Data types adapted for PostgreSQL vs SQLite
    ts_type = "TIMESTAMP"
    num_type = "NUMERIC(10, 2)" if db_type == "postgresql" else "REAL"
    float_type = "DOUBLE PRECISION" if db_type == "postgresql" else "REAL"

    ddl_statements = [
        # 1. Category Translation
        f"""
        CREATE TABLE IF NOT EXISTS dim_category_translation (
            product_category_name VARCHAR(100) PRIMARY KEY,
            product_category_name_english VARCHAR(100) NOT NULL
        );
        """,

        # 2. Products
        f"""
        CREATE TABLE IF NOT EXISTS dim_products (
            product_id VARCHAR(50) PRIMARY KEY,
            product_category_name VARCHAR(100),
            product_category_name_english VARCHAR(100),
            product_name_length INTEGER,
            product_description_length INTEGER,
            product_photos_qty INTEGER,
            product_weight_g {float_type},
            product_length_cm {float_type},
            product_height_cm {float_type},
            product_width_cm {float_type}
        );
        """,

        # 3. Geolocation
        f"""
        CREATE TABLE IF NOT EXISTS dim_geolocation (
            geolocation_zip_code_prefix INTEGER PRIMARY KEY,
            latitude {float_type} NOT NULL,
            longitude {float_type} NOT NULL,
            city VARCHAR(100) NOT NULL,
            state VARCHAR(10) NOT NULL
        );
        """,

        # 4. Customers
        f"""
        CREATE TABLE IF NOT EXISTS dim_customers (
            customer_id VARCHAR(50) PRIMARY KEY,
            customer_unique_id VARCHAR(50) NOT NULL,
            customer_zip_code_prefix INTEGER NOT NULL,
            customer_city VARCHAR(100) NOT NULL,
            customer_state VARCHAR(10) NOT NULL
        );
        """,

        # 5. Sellers
        f"""
        CREATE TABLE IF NOT EXISTS dim_sellers (
            seller_id VARCHAR(50) PRIMARY KEY,
            seller_zip_code_prefix INTEGER NOT NULL,
            seller_city VARCHAR(100) NOT NULL,
            seller_state VARCHAR(10) NOT NULL
        );
        """,

        # 6. Orders
        f"""
        CREATE TABLE IF NOT EXISTS fact_orders (
            order_id VARCHAR(50) PRIMARY KEY,
            customer_id VARCHAR(50) NOT NULL,
            order_status VARCHAR(30) NOT NULL,
            order_purchase_timestamp {ts_type} NOT NULL,
            order_approved_at {ts_type},
            order_delivered_carrier_date {ts_type},
            order_delivered_customer_date {ts_type},
            order_estimated_delivery_date {ts_type} NOT NULL
        );
        """,

        # 7. Order Items
        f"""
        CREATE TABLE IF NOT EXISTS fact_order_items (
            order_id VARCHAR(50) NOT NULL,
            order_item_id INTEGER NOT NULL,
            product_id VARCHAR(50) NOT NULL,
            seller_id VARCHAR(50) NOT NULL,
            shipping_limit_date {ts_type},
            price {num_type} NOT NULL,
            freight_value {num_type} NOT NULL,
            PRIMARY KEY (order_id, order_item_id)
        );
        """,

        # 8. Order Payments
        f"""
        CREATE TABLE IF NOT EXISTS fact_order_payments (
            order_id VARCHAR(50) NOT NULL,
            payment_sequential INTEGER NOT NULL,
            payment_type VARCHAR(30) NOT NULL,
            payment_installments INTEGER NOT NULL,
            payment_value {num_type} NOT NULL,
            PRIMARY KEY (order_id, payment_sequential)
        );
        """,

        # 9. Order Reviews
        f"""
        CREATE TABLE IF NOT EXISTS fact_order_reviews (
            review_id VARCHAR(50) NOT NULL,
            order_id VARCHAR(50) NOT NULL,
            review_score INTEGER NOT NULL,
            review_comment_title TEXT,
            review_comment_message TEXT,
            review_creation_date {ts_type} NOT NULL,
            review_answer_timestamp {ts_type} NOT NULL,
            PRIMARY KEY (review_id, order_id)
        );
        """
    ]

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_customers_unique_id ON dim_customers(customer_unique_id);",
        "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON fact_orders(customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_orders_purchase_ts ON fact_orders(order_purchase_timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON fact_orders(order_status);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON fact_order_items(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON fact_order_items(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_seller_id ON fact_order_items(seller_id);",
        "CREATE INDEX IF NOT EXISTS idx_payments_order_id ON fact_order_payments(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_reviews_order_id ON fact_order_reviews(order_id);",
    ]

    with engine.begin() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt))
        for idx in indexes:
            conn.execute(text(idx))

    logger.info("Schema tables and analytical indexes initialized successfully.")


def load_tables(engine, processed_dir: Path):
    """Loads dimension tables first, followed by fact tables in dependency order."""
    tables_in_order = [
        # Dimensions
        "dim_category_translation",
        "dim_products",
        "dim_geolocation",
        "dim_customers",
        "dim_sellers",
        # Facts
        "fact_orders",
        "fact_order_items",
        "fact_order_payments",
        "fact_order_reviews",
    ]

    logger.info("Loading cleaned datasets into database tables...")
    for tbl in tables_in_order:
        csv_file = processed_dir / f"{tbl}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Missing processed file: {csv_file}")
        
        logger.info(f"Loading {tbl} from {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # Batch insert using chunksize
        df.to_sql(tbl, engine, if_exists="replace", index=False, chunksize=10000)
        logger.info(f"Loaded {tbl}: {len(df):,} rows.")


def verify_loaded_database(engine) -> Dict[str, int]:
    """Queries all tables to verify row counts and check for orphaned records."""
    logger.info("Running post-load database verification and row count audit...")
    tables = [
        "dim_category_translation",
        "dim_products",
        "dim_geolocation",
        "dim_customers",
        "dim_sellers",
        "fact_orders",
        "fact_order_items",
        "fact_order_payments",
        "fact_order_reviews",
    ]
    counts = {}
    with engine.connect() as conn:
        for tbl in tables:
            res = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}"))
            count = res.scalar()
            counts[tbl] = count
            logger.info(f"Table '{tbl}': {count:,} records.")

        # SQL Foreign Key Orphan Check
        orphan_check_queries = [
            ("fact_orders", "customer_id", "dim_customers", "customer_id"),
            ("fact_order_items", "order_id", "fact_orders", "order_id"),
            ("fact_order_items", "product_id", "dim_products", "product_id"),
            ("fact_order_items", "seller_id", "dim_sellers", "seller_id"),
            ("fact_order_payments", "order_id", "fact_orders", "order_id"),
            ("fact_order_reviews", "order_id", "fact_orders", "order_id"),
        ]
        for child_tbl, child_fk, parent_tbl, parent_pk in orphan_check_queries:
            query = f"""
            SELECT COUNT(*) 
            FROM {child_tbl} c
            LEFT JOIN {parent_tbl} p ON c.{child_fk} = p.{parent_pk}
            WHERE p.{parent_pk} IS NULL;
            """
            orphan_count = conn.execute(text(query)).scalar()
            if orphan_count > 0:
                logger.error(f"Integrity check failed: {orphan_count:,} orphaned records in {child_tbl}.{child_fk}")
            else:
                logger.info(f"Integrity check PASSED: 0 orphans in {child_tbl}.{child_fk} -> {parent_tbl}.{parent_pk}")

    return counts


class OlistPostgresLoader:
    """Orchestrates database connection, schema creation, data loading, and verification."""

    def __init__(self, processed_dir: Path = None):
        self.processed_dir = processed_dir or (PROJECT_ROOT / "data" / "processed" / "olist")

    def run_load(self) -> Dict[str, Any]:
        engine, db_type = get_db_engine()
        create_schema(engine, db_type=db_type)
        load_results = load_tables(engine, processed_dir=self.processed_dir)
        counts = verify_loaded_database(engine)
        return {
            "backend": db_type,
            "table_results": load_results,
            "row_counts": counts,
            "total_rows_loaded": sum(counts.values()) if counts else 0,
        }


def main():
    processed_dir = PROJECT_ROOT / "data" / "processed" / "olist"

    # Step 1: Pre-load automated validation
    logger.info("Step 1: Running automated validation on processed CSV files...")
    validator = OlistDataValidator(processed_dir=processed_dir)
    validator.validate_all()

    # Step 2: Connect to database
    logger.info("Step 2: Connecting to database engine...")
    engine, db_type = get_db_engine()

    # Step 3: Create schema and indexes
    logger.info("Step 3: Creating schema tables and indexes...")
    create_schema(engine, db_type=db_type)

    # Step 4: Load data
    logger.info("Step 4: Loading tables...")
    load_tables(engine, processed_dir=processed_dir)

    # Step 5: Verify row counts and integrity
    logger.info("Step 5: Verifying loaded records in database...")
    counts = verify_loaded_database(engine)

    logger.info("=====================================================================")
    logger.info(f"Database loading completed successfully ({db_type.upper()}).")
    for tbl, cnt in counts.items():
        logger.info(f" - {tbl:28s}: {cnt:>10,} rows")
    logger.info("=====================================================================")


if __name__ == "__main__":
    main()
