"""
Olist E-Commerce Data Cleaning & Standardization Module.
Transforms raw Olist CSV tables into normalized, production-grade dimension and fact datasets.
Outputs cleaned tables to data/processed/olist/.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

from src.data.ingestion.load_olist import load_raw_olist_data

logger = logging.getLogger("CleanOlist")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(handler)


class OlistDataCleaner:
    """Cleans and standardizes raw Olist datasets adhering to strict domain rules."""

    def __init__(self, raw_datasets: Dict[str, pd.DataFrame] = None, output_dir: Path = None):
        self.raw_datasets = raw_datasets
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.output_dir = output_dir or (project_root / "data" / "processed" / "olist")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cleaning_stats = {}

    def clean_all(self) -> Dict[str, pd.DataFrame]:
        """Executes full cleaning pipeline across all dimensions and facts."""
        if self.raw_datasets is None:
            self.raw_datasets = load_raw_olist_data()

        logger.info("Starting Olist data cleaning pipeline...")

        # 1. Category Translation Dimension
        dim_category_translation = self._clean_category_translation(
            self.raw_datasets["category_translation"]
        )

        # 2. Products Dimension
        dim_products = self._clean_products(
            self.raw_datasets["products"],
            dim_category_translation
        )

        # 3. Geolocation Dimension (Aggregated)
        dim_geolocation = self._clean_geolocation(
            self.raw_datasets["geolocation"]
        )

        # 4. Customers Dimension
        dim_customers = self._clean_customers(
            self.raw_datasets["customers"]
        )

        # 5. Sellers Dimension
        dim_sellers = self._clean_sellers(
            self.raw_datasets["sellers"]
        )

        # 6. Orders Fact Table
        fact_orders = self._clean_orders(
            self.raw_datasets["orders"]
        )

        # 7. Order Items Fact Table
        fact_order_items = self._clean_order_items(
            self.raw_datasets["order_items"]
        )

        # 8. Order Payments Fact Table
        fact_order_payments = self._clean_order_payments(
            self.raw_datasets["payments"]
        )

        # 9. Order Reviews Fact Table
        fact_order_reviews = self._clean_order_reviews(
            self.raw_datasets["reviews"]
        )

        cleaned_dfs = {
            "dim_category_translation": dim_category_translation,
            "dim_products": dim_products,
            "dim_geolocation": dim_geolocation,
            "dim_customers": dim_customers,
            "dim_sellers": dim_sellers,
            "fact_orders": fact_orders,
            "fact_order_items": fact_order_items,
            "fact_order_payments": fact_order_payments,
            "fact_order_reviews": fact_order_reviews,
        }

        # Save all processed datasets to CSV
        self._export_cleaned_tables(cleaned_dfs)
        logger.info("Olist data cleaning pipeline completed successfully.")
        return cleaned_dfs

    def _clean_category_translation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans category translations and incorporates missing Portuguese categories."""
        df_clean = df.copy()
        
        # Add documented missing categories from products table
        # 1. pc_gamer -> pc_gamer
        # 2. portateis_cozinha_e_preparadores_de_alimentos -> small_appliances_kitchen_and_food_preparers
        missing_rows = [
            {"product_category_name": "pc_gamer", "product_category_name_english": "pc_gamer"},
            {"product_category_name": "portateis_cozinha_e_preparadores_de_alimentos", "product_category_name_english": "small_appliances_kitchen_and_food_preparers"},
            {"product_category_name": "unknown_category", "product_category_name_english": "unknown_category"}
        ]
        missing_df = pd.DataFrame(missing_rows)
        df_clean = pd.concat([df_clean, missing_df], ignore_index=True).drop_duplicates(subset=["product_category_name"])
        df_clean = df_clean.sort_values(by="product_category_name").reset_index(drop=True)
        return df_clean

    def _clean_products(self, df: pd.DataFrame, df_trans: pd.DataFrame) -> pd.DataFrame:
        """Standardizes products, preserves missing categories, and joins English translations."""
        df_clean = df.copy()

        # Handle missing product categories without dropping rows
        null_cat_count = int(df_clean["product_category_name"].isnull().sum())
        logger.info(f"Products with missing category: {null_cat_count} (preserved as 'unknown_category')")
        df_clean["product_category_name"] = df_clean["product_category_name"].fillna("unknown_category")

        # Merge English translation
        df_clean = df_clean.merge(
            df_trans[["product_category_name", "product_category_name_english"]],
            on="product_category_name",
            how="left"
        )
        df_clean["product_category_name_english"] = df_clean["product_category_name_english"].fillna("unknown_category")

        # Standardize physical dimension column names
        df_clean.rename(columns={
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length"
        }, inplace=True)

        return df_clean

    def _clean_geolocation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates raw 1,000,163 geolocation records into a unique postal-code dimension table.
        - Latitude: Mean latitude per prefix
        - Longitude: Mean longitude per prefix
        - City: Deterministic mode (most frequent)
        - State: Deterministic mode (most frequent)
        """
        logger.info("Aggregating raw 1M-row geolocation records by zip code prefix...")
        df_clean = df.copy()

        # Calculate mean coordinates
        coords = df_clean.groupby("geolocation_zip_code_prefix").agg(
            latitude=("geolocation_lat", "mean"),
            longitude=("geolocation_lng", "mean")
        ).reset_index()

        # Determine most frequent city and state deterministically
        def get_deterministic_mode(series):
            return series.value_counts().sort_index().index[0]

        city_state = df_clean.groupby("geolocation_zip_code_prefix").agg(
            city=("geolocation_city", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            state=("geolocation_state", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        ).reset_index()

        dim_geo = coords.merge(city_state, on="geolocation_zip_code_prefix")
        dim_geo["city"] = dim_geo["city"].astype(str).str.strip().str.title()
        dim_geo["state"] = dim_geo["state"].astype(str).str.strip().str.upper()

        logger.info(f"Geolocation dimension aggregated: {len(dim_geo):,} unique postal code prefixes.")
        return dim_geo

    def _clean_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preserves customer_id and customer_unique_id distinction, standardizes location."""
        df_clean = df.copy()
        df_clean["customer_city"] = df_clean["customer_city"].astype(str).str.strip().str.title()
        df_clean["customer_state"] = df_clean["customer_state"].astype(str).str.strip().str.upper()
        return df_clean

    def _clean_sellers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preserves sellers and standardizes city and state."""
        df_clean = df.copy()
        df_clean["seller_city"] = df_clean["seller_city"].astype(str).str.strip().str.title()
        df_clean["seller_state"] = df_clean["seller_state"].astype(str).str.strip().str.upper()
        return df_clean

    def _clean_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses all 5 timestamp columns into datetime types without fabricating dates."""
        df_clean = df.copy()
        timestamp_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
        for col in timestamp_cols:
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

        return df_clean

    def _clean_order_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preserves price, freight, seller, product, and parses shipping limit date."""
        df_clean = df.copy()
        df_clean["shipping_limit_date"] = pd.to_datetime(df_clean["shipping_limit_date"], errors="coerce")
        df_clean["price"] = df_clean["price"].astype(float)
        df_clean["freight_value"] = df_clean["freight_value"].astype(float)
        return df_clean

    def _clean_order_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preserves payment sequential, installments, and payment value."""
        df_clean = df.copy()
        df_clean["payment_value"] = df_clean["payment_value"].astype(float)
        df_clean["payment_installments"] = df_clean["payment_installments"].astype(int)
        df_clean["payment_sequential"] = df_clean["payment_sequential"].astype(int)
        return df_clean

    def _clean_order_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses review timestamps and standardizes review score."""
        df_clean = df.copy()
        df_clean["review_creation_date"] = pd.to_datetime(df_clean["review_creation_date"], errors="coerce")
        df_clean["review_answer_timestamp"] = pd.to_datetime(df_clean["review_answer_timestamp"], errors="coerce")
        df_clean["review_score"] = df_clean["review_score"].astype(int)
        df_clean["review_comment_title"] = df_clean["review_comment_title"].fillna("")
        df_clean["review_comment_message"] = df_clean["review_comment_message"].fillna("")
        return df_clean

    def _export_cleaned_tables(self, cleaned_dfs: Dict[str, pd.DataFrame]) -> None:
        """Saves all cleaned tables to data/processed/olist/."""
        for name, df in cleaned_dfs.items():
            filepath = self.output_dir / f"{name}.csv"
            logger.info(f"Writing {name}.csv ({len(df):,} rows)...")
            df.to_csv(filepath, index=False)


if __name__ == "__main__":
    cleaner = OlistDataCleaner()
    dfs = cleaner.clean_all()
    print("All Olist tables cleaned and saved to data/processed/olist/.")
