"""
Raw Olist E-Commerce Data Loader.
Reads raw CSV files exclusively from data/raw/ and returns structured DataFrames.
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import logging

logger = logging.getLogger("LoadOlist")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(handler)


def get_raw_data_dir() -> Path:
    """Locates the raw data directory."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found at {raw_dir}")
    return raw_dir


def load_raw_olist_data(raw_dir: Path = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all 9 raw Olist CSV files into memory without modifications.
    """
    if raw_dir is None:
        raw_dir = get_raw_data_dir()

    logger.info(f"Loading raw Olist CSV datasets from: {raw_dir}")

    file_mapping = {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "products": "olist_products_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "category_translation": "product_category_name_translation.csv",
    }

    datasets = {}
    for key, filename in file_mapping.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required raw dataset file missing: {filepath}")
        
        logger.info(f"Reading {filename}...")
        df = pd.read_csv(filepath)
        datasets[key] = df
        logger.info(f"Loaded {key}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    return datasets


if __name__ == "__main__":
    dfs = load_raw_olist_data()
    print("All datasets loaded successfully.")
