"""
Automated Data Validation Module for Cleaned Olist Datasets.
Executes rigorous integrity checks: primary key uniqueness, foreign key referential integrity,
non-negative monetary validation, timestamp consistency, and orphan record detection.
FAILS loudly if any critical validation check is breached.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("ValidateOlist")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(handler)


class OlistDataValidator:
    """Performs automated integrity validation on processed Olist dimension and fact tables."""

    def __init__(self, processed_dir: Path = None):
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.processed_dir = processed_dir or (project_root / "data" / "processed" / "olist")
        self.validation_results = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def load_processed_tables(self) -> Dict[str, pd.DataFrame]:
        """Loads all processed CSV tables from the target directory."""
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
        dfs = {}
        for tbl in tables:
            path = self.processed_dir / f"{tbl}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Cleaned table file not found: {path}")
            dfs[tbl] = pd.read_csv(path)
        return dfs

    def validate_all(self, dfs: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Runs the complete suite of validation checks."""
        if dfs is None:
            dfs = self.load_processed_tables()

        logger.info("Starting automated validation suite on processed Olist tables...")
        self.errors.clear()
        self.warnings.clear()

        # 1. Row Count Checks
        self._validate_row_counts(dfs)

        # 2. Primary Key & Composite Key Uniqueness
        self._validate_primary_keys(dfs)

        # 3. Foreign Key Referential Integrity
        self._validate_foreign_keys(dfs)

        # 4. Monetary Value Validity (Non-Negative Checks)
        self._validate_monetary_values(dfs)

        # 5. Timestamp Validity
        self._validate_timestamps(dfs)

        # Compile Report
        is_passed = len(self.errors) == 0
        report = {
            "validation_status": "PASSED" if is_passed else "FAILED",
            "critical_errors": self.errors,
            "warnings": self.warnings,
            "row_counts": {k: len(v) for k, v in dfs.items()},
        }
        self.validation_results = report

        if not is_passed:
            error_msg = f"Data validation FAILED with {len(self.errors)} critical errors:\n" + "\n".join(f" - {e}" for e in self.errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("All data validation checks PASSED successfully.")
        return report

    def _validate_row_counts(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Verifies that none of the tables are empty and match expected orders of magnitude."""
        expected_minimums = {
            "dim_category_translation": 70,
            "dim_products": 30000,
            "dim_geolocation": 15000,
            "dim_customers": 90000,
            "dim_sellers": 3000,
            "fact_orders": 90000,
            "fact_order_items": 100000,
            "fact_order_payments": 100000,
            "fact_order_reviews": 90000,
        }
        for name, min_cnt in expected_minimums.items():
            actual = len(dfs[name])
            if actual < min_cnt:
                self.errors.append(f"Table '{name}' has only {actual:,} rows, expected at least {min_cnt:,}")

    def _validate_primary_keys(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Validates primary and composite key uniqueness."""
        pk_checks = [
            ("dim_category_translation", ["product_category_name"]),
            ("dim_products", ["product_id"]),
            ("dim_geolocation", ["geolocation_zip_code_prefix"]),
            ("dim_customers", ["customer_id"]),
            ("dim_sellers", ["seller_id"]),
            ("fact_orders", ["order_id"]),
            ("fact_order_items", ["order_id", "order_item_id"]),
            ("fact_order_payments", ["order_id", "payment_sequential"]),
            ("fact_order_reviews", ["review_id", "order_id"]),
        ]
        for tbl_name, pk_cols in pk_checks:
            df = dfs[tbl_name]
            dups = df.duplicated(subset=pk_cols).sum()
            nulls = df[pk_cols].isnull().any(axis=1).sum()
            if dups > 0:
                self.errors.append(f"Table '{tbl_name}' has {dups:,} duplicate primary key records on {pk_cols}")
            if nulls > 0:
                self.errors.append(f"Table '{tbl_name}' has {nulls:,} null values in primary key {pk_cols}")

    def _validate_foreign_keys(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Validates referential integrity across parent and child fact tables."""
        fk_relationships = [
            ("fact_orders", "customer_id", "dim_customers", "customer_id"),
            ("fact_order_items", "order_id", "fact_orders", "order_id"),
            ("fact_order_items", "product_id", "dim_products", "product_id"),
            ("fact_order_items", "seller_id", "dim_sellers", "seller_id"),
            ("fact_order_payments", "order_id", "fact_orders", "order_id"),
            ("fact_order_reviews", "order_id", "fact_orders", "order_id"),
        ]
        for child_tbl, child_fk, parent_tbl, parent_pk in fk_relationships:
            child_series = dfs[child_tbl][child_fk].dropna()
            parent_set = set(dfs[parent_tbl][parent_pk])
            orphan_count = (~child_series.isin(parent_set)).sum()
            if orphan_count > 0:
                self.errors.append(
                    f"Referential integrity failure: '{child_tbl}.{child_fk}' has {orphan_count:,} orphan records "
                    f"not found in '{parent_tbl}.{parent_pk}'"
                )

    def _validate_monetary_values(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Ensures price, freight, and payments are strictly non-negative."""
        # Order items
        df_items = dfs["fact_order_items"]
        neg_price = (df_items["price"] < 0).sum()
        neg_freight = (df_items["freight_value"] < 0).sum()
        if neg_price > 0:
            self.errors.append(f"fact_order_items has {neg_price:,} records with negative price")
        if neg_freight > 0:
            self.errors.append(f"fact_order_items has {neg_freight:,} records with negative freight value")

        # Order payments
        df_pay = dfs["fact_order_payments"]
        neg_pay = (df_pay["payment_value"] < 0).sum()
        zero_pay = (df_pay["payment_value"] == 0).sum()
        if neg_pay > 0:
            self.errors.append(f"fact_order_payments has {neg_pay:,} records with negative payment values")
        if zero_pay > 0:
            self.warnings.append(f"fact_order_payments has {zero_pay:,} zero-value payments (e.g. voucher subsidies)")

    def _validate_timestamps(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Validates timestamp consistency on orders."""
        df_orders = dfs["fact_orders"]
        null_purchase = df_orders["order_purchase_timestamp"].isnull().sum()
        if null_purchase > 0:
            self.errors.append(f"fact_orders has {null_purchase:,} records with null purchase timestamps")

        # Check delivery timestamp vs purchase timestamp where both exist
        # Temporal inconsistency check (delivered before purchase)
        purch_dt = pd.to_datetime(df_orders["order_purchase_timestamp"])
        deliv_dt = pd.to_datetime(df_orders["order_delivered_customer_date"])
        inconsistent_delivery = ((deliv_dt.notnull()) & (deliv_dt < purch_dt)).sum()
        if inconsistent_delivery > 0:
            self.warnings.append(
                f"fact_orders has {inconsistent_delivery:,} delivered orders where customer delivery date "
                f"is earlier than purchase timestamp (carrier system recording anomaly)"
            )


if __name__ == "__main__":
    validator = OlistDataValidator()
    report = validator.validate_all()
    print("Validation passed! Report summary:", report["validation_status"])
