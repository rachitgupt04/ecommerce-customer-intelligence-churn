"""
Temporal Multi-Snapshot Feature Engineering Engine for Real Olist E-Commerce Churn Modeling.
Constructs panel datasets across multiple historical observation-prediction windows with
strict delivery timestamp clamping and zero look-ahead information leakage.
"""

from pathlib import Path
from datetime import timedelta
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np

from src.utils.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SNAPSHOT_CONFIG,
    PREDICTION_HORIZON_DAYS,
)
from src.utils.logger import get_logger
from src.utils import save_json

logger = get_logger("FeatureEngineering")


class OlistFeatureEngineer:
    """Constructs leakage-safe multi-snapshot customer feature matrices and ground-truth churn labels."""

    def __init__(self, data_dir: Path = PROCESSED_DATA_DIR, reports_dir: Path = REPORTS_DIR):
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.df_orders = None
        self.df_items = None
        self.df_cust = None
        self.df_products = None
        self.df_payments = None
        self.df_reviews = None
        self._load_tables()

    def _load_tables(self) -> None:
        """Loads and pre-parses processed relational tables."""
        logger.info("Loading processed relational tables for multi-snapshot feature extraction...")
        self.df_orders = pd.read_csv(self.data_dir / "fact_orders.csv")
        self.df_items = pd.read_csv(self.data_dir / "fact_order_items.csv")
        self.df_cust = pd.read_csv(self.data_dir / "dim_customers.csv")
        self.df_products = pd.read_csv(self.data_dir / "dim_products.csv")
        self.df_payments = pd.read_csv(self.data_dir / "fact_order_payments.csv")
        self.df_reviews = pd.read_csv(self.data_dir / "fact_order_reviews.csv")

        # Parse datetime columns
        self.df_orders["order_purchase_timestamp"] = pd.to_datetime(self.df_orders["order_purchase_timestamp"])
        self.df_orders["order_delivered_customer_date"] = pd.to_datetime(self.df_orders["order_delivered_customer_date"])
        self.df_orders["order_estimated_delivery_date"] = pd.to_datetime(self.df_orders["order_estimated_delivery_date"])
        self.df_reviews["review_creation_date"] = pd.to_datetime(self.df_reviews["review_creation_date"])

    def extract_single_snapshot(
        self,
        snapshot_date: str,
        horizon_days: int = PREDICTION_HORIZON_DAYS,
        split_role: str = "train",
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extracts customer features strictly prior to snapshot_date and defines churn in forward horizon:
        - Observation Window: All orders before snapshot_date (or within lookback_days).
        - Clamping: Any delivery timestamp on or after snapshot_date is masked to NaT.
        - Prediction Window: [snapshot_date, snapshot_date + horizon_days].
        - Target: 1 if customer placed 0 qualifying orders in prediction window, 0 if >= 1 order.
        """
        snap_dt = pd.to_datetime(snapshot_date)
        pred_end_dt = snap_dt + timedelta(days=horizon_days)

        logger.info(
            f"Extracting Snapshot [{snapshot_date}] (Horizon: {horizon_days}d -> {pred_end_dt.strftime('%Y-%m-%d')} | Split: {split_role})..."
        )

        # 1. Historical Observation Orders strictly BEFORE snapshot_date
        obs_orders = self.df_orders[self.df_orders["order_purchase_timestamp"] < snap_dt].copy()
        if lookback_days is not None:
            obs_orders = obs_orders[obs_orders["order_purchase_timestamp"] >= snap_dt - timedelta(days=lookback_days)]

        obs_orders = obs_orders.merge(self.df_cust[["customer_id", "customer_unique_id"]], on="customer_id")
        active_customers = obs_orders["customer_unique_id"].unique()

        # Zero-Leakage Delivery Date Clamping:
        # If an order placed before snapshot was delivered on/after snapshot, delivery date must be masked!
        obs_orders.loc[obs_orders["order_delivered_customer_date"] >= snap_dt, "order_delivered_customer_date"] = pd.NaT

        # Calculate delivery delay only where delivery date was confirmed prior to snapshot
        obs_orders["delivery_delay_days"] = (
            obs_orders["order_delivered_customer_date"] - obs_orders["order_estimated_delivery_date"]
        ).dt.total_seconds() / (24 * 3600)

        # Merge items, products, payments, reviews strictly before snapshot
        obs_items = self.df_items.merge(
            obs_orders[["order_id", "customer_unique_id", "order_purchase_timestamp", "order_status"]],
            on="order_id",
        )
        obs_items = obs_items.merge(self.df_products[["product_id", "product_category_name_english"]], on="product_id", how="left")
        obs_payments = self.df_payments.merge(obs_orders[["order_id", "customer_unique_id"]], on="order_id")

        valid_reviews = self.df_reviews[self.df_reviews["review_creation_date"] < snap_dt]
        obs_reviews = valid_reviews.merge(obs_orders[["order_id", "customer_unique_id"]], on="order_id")

        # 2. Compute Customer-Level Aggregates
        cust_first = obs_orders.groupby("customer_unique_id")["order_purchase_timestamp"].min()
        cust_latest = obs_orders.groupby("customer_unique_id")["order_purchase_timestamp"].max()
        order_frequency = obs_orders.groupby("customer_unique_id")["order_id"].nunique()

        item_grp = obs_items.groupby("customer_unique_id")
        monetary_total = item_grp["price"].sum()
        total_quantity = item_grp["order_item_id"].count()
        avg_freight = item_grp["freight_value"].mean()
        total_freight = item_grp["freight_value"].sum()
        freight_ratio = total_freight / np.maximum(monetary_total + total_freight, 0.01)
        avg_order_value = monetary_total / np.maximum(order_frequency, 1)
        category_diversity = item_grp["product_category_name_english"].nunique()

        avg_review_score = obs_reviews.groupby("customer_unique_id")["review_score"].mean()
        cancelled_orders_cnt = obs_orders[obs_orders["order_status"] == "canceled"].groupby("customer_unique_id")["order_id"].nunique()

        pay_grp = obs_payments.groupby("customer_unique_id")
        cc_val = obs_payments[obs_payments["payment_type"] == "credit_card"].groupby("customer_unique_id")["payment_value"].sum()
        credit_card_share = cc_val / np.maximum(pay_grp["payment_value"].sum(), 0.01)
        avg_installments = pay_grp["payment_installments"].mean()

        t_minus_30 = snap_dt - timedelta(days=30)
        t_minus_60 = snap_dt - timedelta(days=60)
        last_30d_orders = obs_orders[obs_orders["order_purchase_timestamp"] >= t_minus_30].groupby("customer_unique_id")["order_id"].nunique()
        last_60d_orders = obs_orders[obs_orders["order_purchase_timestamp"] >= t_minus_60].groupby("customer_unique_id")["order_id"].nunique()
        last_60d_spend = obs_items[obs_items["order_purchase_timestamp"] >= t_minus_60].groupby("customer_unique_id")["price"].sum()

        customer_tenure_days = (snap_dt - cust_first).dt.total_seconds() / (24 * 3600)
        recency_days = (snap_dt - cust_latest).dt.total_seconds() / (24 * 3600)
        expected_60d_spend = (monetary_total / np.maximum(customer_tenure_days, 60.0)) * 60.0
        spending_trend_velocity = last_60d_spend / np.maximum(expected_60d_spend, 1.0)
        avg_delivery_delay_days = obs_orders.groupby("customer_unique_id")["delivery_delay_days"].mean()

        # Build feature dataframe
        df_feat = pd.DataFrame(index=active_customers)
        df_feat["recency_days"] = recency_days
        df_feat["order_frequency"] = order_frequency
        df_feat["monetary_total"] = monetary_total
        df_feat["avg_order_value"] = avg_order_value
        df_feat["total_quantity"] = total_quantity
        df_feat["customer_tenure_days"] = customer_tenure_days
        df_feat["avg_freight_value"] = avg_freight
        df_feat["freight_ratio"] = freight_ratio
        df_feat["category_diversity"] = category_diversity
        df_feat["avg_review_score"] = avg_review_score
        df_feat["cancelled_orders_cnt"] = cancelled_orders_cnt
        df_feat["credit_card_share"] = credit_card_share
        df_feat["avg_installments"] = avg_installments
        df_feat["last_30d_orders"] = last_30d_orders
        df_feat["last_60d_orders"] = last_60d_orders
        df_feat["spending_trend_velocity"] = spending_trend_velocity
        df_feat["avg_delivery_delay_days"] = avg_delivery_delay_days

        df_feat.fillna({
            "recency_days": 999.0,
            "order_frequency": 1,
            "monetary_total": 0.0,
            "avg_order_value": 0.0,
            "total_quantity": 1,
            "customer_tenure_days": 0.0,
            "avg_freight_value": 0.0,
            "freight_ratio": 0.0,
            "category_diversity": 1,
            "avg_review_score": 4.0,
            "cancelled_orders_cnt": 0,
            "credit_card_share": 0.0,
            "avg_installments": 1.0,
            "last_30d_orders": 0,
            "last_60d_orders": 0,
            "spending_trend_velocity": 0.0,
            "avg_delivery_delay_days": 0.0,
        }, inplace=True)

        # 3. Ground-truth Churn Target in Forward Prediction Window
        future_orders = self.df_orders[
            (self.df_orders["order_purchase_timestamp"] >= snap_dt) &
            (self.df_orders["order_purchase_timestamp"] <= pred_end_dt) &
            (~self.df_orders["order_status"].isin(["canceled", "unavailable"]))
        ]
        future_custs = future_orders.merge(self.df_cust[["customer_id", "customer_unique_id"]], on="customer_id")
        retained_customers = set(future_custs["customer_unique_id"].unique())

        df_feat[TARGET_COLUMN] = df_feat.index.map(lambda cid: 0 if cid in retained_customers else 1)
        df_feat["snapshot_date"] = snapshot_date
        df_feat["split_role"] = split_role

        df_feat.reset_index(inplace=True)
        df_feat.rename(columns={"index": "customer_unique_id"}, inplace=True)
        return df_feat

    def build_multi_snapshot_dataset(self, snapshots: Optional[List[Dict]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Builds the complete multi-snapshot panel dataset across the configured chronological snapshots.
        Performs target quality audit and exports reports.
        """
        snaps = snapshots or SNAPSHOT_CONFIG
        logger.info(f"Generating temporal multi-snapshot churn dataset across {len(snaps)} snapshots...")

        dfs = []
        snapshot_summaries = []

        for cfg in snaps:
            df_snap = self.extract_single_snapshot(
                snapshot_date=cfg["date"],
                horizon_days=cfg.get("horizon_days", PREDICTION_HORIZON_DAYS),
                split_role=cfg.get("split", "train"),
            )
            dfs.append(df_snap)

            n_obs = len(df_snap)
            n_churn = int(df_snap[TARGET_COLUMN].sum())
            n_ret = n_obs - n_churn
            churn_pct = (n_churn / n_obs) * 100.0 if n_obs > 0 else 0.0

            snapshot_summaries.append({
                "snapshot_name": cfg["name"],
                "snapshot_date": cfg["date"],
                "split_role": cfg["split"],
                "horizon_days": cfg.get("horizon_days", PREDICTION_HORIZON_DAYS),
                "total_observations": n_obs,
                "retained_count": n_ret,
                "churned_count": n_churn,
                "churn_rate_pct": round(churn_pct, 2),
                "retention_rate_pct": round(100.0 - churn_pct, 2),
            })

        df_all = pd.concat(dfs, ignore_index=True)
        total_obs = len(df_all)
        total_churn = int(df_all[TARGET_COLUMN].sum())
        total_ret = total_obs - total_churn
        overall_churn_rate = (total_churn / total_obs) * 100.0

        # Unique customers
        unique_customers = df_all["customer_unique_id"].nunique()
        cust_obs_dist = df_all.groupby("customer_unique_id")["snapshot_date"].count().value_counts().to_dict()

        target_report = {
            "num_snapshots": len(snaps),
            "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
            "total_observations": total_obs,
            "unique_customers": unique_customers,
            "overall_churned_count": total_churn,
            "overall_retained_count": total_ret,
            "positive_class_pct": round(overall_churn_rate, 2),
            "negative_class_pct": round(100.0 - overall_churn_rate, 2),
            "customer_snapshot_frequency": {str(k): int(v) for k, v in cust_obs_dist.items()},
            "snapshots": snapshot_summaries,
            "splits": {
                role: {
                    "count": int((df_all["split_role"] == role).sum()),
                    "churn_count": int(df_all[df_all["split_role"] == role][TARGET_COLUMN].sum()),
                    "retention_count": int((df_all[df_all["split_role"] == role][TARGET_COLUMN] == 0).sum()),
                    "retention_rate_pct": round((df_all[df_all["split_role"] == role][TARGET_COLUMN] == 0).mean() * 100.0, 2),
                }
                for role in ["train", "val", "test"]
            },
        }

        # Export outputs
        multi_snap_path = self.data_dir / "churn_multi_snapshot.csv"
        df_all.to_csv(multi_snap_path, index=False)

        # Export test snapshot as standard churn_features.csv for backward compatibility
        test_snap = df_all[df_all["split_role"] == "test"].copy()
        features_path = self.data_dir / "churn_features.csv"
        test_snap.to_csv(features_path, index=False)

        save_json(target_report, self.reports_dir / "target_quality_report.json")
        logger.info(
            f"Multi-snapshot dataset complete: {total_obs:,} observations across {len(snaps)} snapshots. "
            f"Overall Churn: {overall_churn_rate:.2f}% | Retained: {100.0 - overall_churn_rate:.2f}%"
        )
        return df_all, target_report

    def build_churn_feature_matrix(self) -> pd.DataFrame:
        """Standard pipeline compatibility method."""
        df_all, _ = self.build_multi_snapshot_dataset()
        return df_all


if __name__ == "__main__":
    fe = OlistFeatureEngineer()
    df_all, report = fe.build_multi_snapshot_dataset()
    print("Multi-Snapshot Summary:\n", report)
