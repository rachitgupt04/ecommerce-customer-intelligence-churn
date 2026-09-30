"""
RFM Analytics Module for Real Olist E-Commerce Data.
Computes Recency, Frequency, Monetary metrics on customer_unique_id.
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from src.utils.config import CUTOFF_DATE, PROCESSED_DATA_DIR, REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger("RFM")


class RFMCalculator:
    """Calculates customer-level RFM metrics and quintile scores."""

    def __init__(self, cutoff_date: str = CUTOFF_DATE, processed_dir: Path = PROCESSED_DATA_DIR):
        self.cutoff_dt = pd.to_datetime(cutoff_date)
        self.processed_dir = processed_dir
        self.reports_dir = REPORTS_DIR

    def compute_rfm(self, df_orders: pd.DataFrame, df_items: pd.DataFrame, df_cust: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates RFM on customer_unique_id strictly before the cutoff date.
        """
        logger.info(f"Computing RFM metrics as of snapshot date: {self.cutoff_dt.strftime('%Y-%m-%d')}...")

        df_orders["order_purchase_timestamp"] = pd.to_datetime(df_orders["order_purchase_timestamp"])
        
        # Filter strictly prior to cutoff date
        hist_orders = df_orders[
            (df_orders["order_purchase_timestamp"] < self.cutoff_dt) &
            (~df_orders["order_status"].isin(["canceled", "unavailable"]))
        ].copy()

        # Join to customer_unique_id and items
        hist = hist_orders.merge(df_cust[["customer_id", "customer_unique_id"]], on="customer_id")
        hist = hist.merge(df_items[["order_id", "price"]], on="order_id")

        # Group by customer_unique_id
        rfm = hist.groupby("customer_unique_id").agg(
            latest_order=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("price", "sum"),
        ).reset_index()

        # Calculate recency in days relative to snapshot date
        rfm["recency"] = (self.cutoff_dt - rfm["latest_order"]).dt.total_seconds() / (24 * 3600)
        rfm["recency"] = rfm["recency"].round(1)
        rfm["monetary"] = rfm["monetary"].round(2)

        # Quintile scoring (1-5)
        # R_score: lower recency is better (rank ascending=False)
        rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        # F_score: higher frequency is better
        rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        # M_score: higher spend is better
        rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

        rfm["rfm_score"] = rfm["r_score"] * 100 + rfm["f_score"] * 10 + rfm["m_score"]

        # Data-grounded rule segment assignment
        def map_rfm_segment(row):
            r, f, m = row["r_score"], row["f_score"], row["m_score"]
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3:
                return "Loyal Customers"
            elif r >= 4 and f <= 2:
                return "New / Promising"
            elif r <= 2 and f >= 3 and m >= 3:
                return "At-Risk High-Value"
            elif r <= 2 and m >= 4:
                return "At-Risk Spenders"
            elif r >= 3 and f <= 2:
                return "Potential Loyalists"
            elif r <= 2 and f <= 2:
                return "Hibernating"
            else:
                return "Needs Attention"

        rfm["rfm_segment"] = rfm.apply(map_rfm_segment, axis=1)

        # Export RFM summary
        output_file = self.processed_dir / "customer_rfm.csv"
        rfm.to_csv(output_file, index=False)
        logger.info(f"RFM analysis computed for {len(rfm):,} unique customers.")
        return rfm


if __name__ == "__main__":
    df_o = pd.read_csv(PROCESSED_DATA_DIR / "fact_orders.csv")
    df_i = pd.read_csv(PROCESSED_DATA_DIR / "fact_order_items.csv")
    df_c = pd.read_csv(PROCESSED_DATA_DIR / "dim_customers.csv")
    rfm_calc = RFMCalculator()
    df_rfm = rfm_calc.compute_rfm(df_o, df_i, df_c)
    print("RFM preview:\n", df_rfm.head())

