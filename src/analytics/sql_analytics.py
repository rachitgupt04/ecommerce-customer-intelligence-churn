"""
SQL Analytics Engine for Real Olist E-Commerce Data.
Executes advanced SQL queries using CTEs, Window Functions (LAG, DENSE_RANK, NTILE),
and cohort aggregations against the active relational warehouse.
"""

from pathlib import Path
from typing import Dict
import pandas as pd
from sqlalchemy import text
from src.data.database import get_db_engine
from src.utils.config import REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger("SQLAnalytics")


class SQLAnalyticsEngine:
    """Executes production SQL analytical workflows against the Olist database."""

    def __init__(self):
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.engine, self.backend = get_db_engine()

    def get_monthly_revenue_trends(self) -> pd.DataFrame:
        """Calculates monthly gross revenue, order volume, and Month-over-Month (MoM) growth using LAG()."""
        query = """
        WITH monthly_orders AS (
            SELECT 
                substr(o.order_purchase_timestamp, 1, 7) AS order_month,
                COUNT(DISTINCT o.order_id) AS total_orders,
                COUNT(DISTINCT c.customer_unique_id) AS unique_customers,
                ROUND(SUM(oi.price), 2) AS item_revenue,
                ROUND(SUM(oi.freight_value), 2) AS total_freight,
                ROUND(SUM(oi.price + oi.freight_value), 2) AS gross_revenue
            FROM fact_orders o
            JOIN dim_customers c ON o.customer_id = c.customer_id
            JOIN fact_order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY substr(o.order_purchase_timestamp, 1, 7)
        ),
        sales_lagged AS (
            SELECT 
                order_month,
                total_orders,
                unique_customers,
                item_revenue,
                gross_revenue,
                LAG(gross_revenue, 1) OVER (ORDER BY order_month) AS prev_month_gross
            FROM monthly_orders
        )
        SELECT 
            order_month,
            total_orders,
            unique_customers,
            item_revenue,
            gross_revenue AS net_revenue,
            prev_month_gross AS prev_month_revenue,
            ROUND(((gross_revenue - prev_month_gross) / prev_month_gross) * 100.0, 2) AS mom_growth_pct
        FROM sales_lagged
        ORDER BY order_month;
        """
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        return df

    def get_category_performance(self) -> pd.DataFrame:
        """Evaluates revenue share, total units sold, and average price per English category."""
        query = """
        WITH category_totals AS (
            SELECT 
                COALESCE(p.product_category_name_english, 'unknown_category') AS category_name,
                COUNT(DISTINCT oi.order_id) AS orders_count,
                COUNT(oi.order_item_id) AS units_sold,
                ROUND(SUM(oi.price), 2) AS category_revenue,
                ROUND(AVG(oi.price), 2) AS avg_item_price,
                ROUND(AVG(oi.freight_value), 2) AS avg_freight
            FROM fact_order_items oi
            JOIN dim_products p ON oi.product_id = p.product_id
            JOIN fact_orders o ON oi.order_id = o.order_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY COALESCE(p.product_category_name_english, 'unknown_category')
        ),
        overall_market AS (
            SELECT SUM(category_revenue) AS total_market_revenue FROM category_totals
        )
        SELECT 
            c.category_name,
            c.orders_count,
            c.units_sold,
            c.category_revenue,
            c.avg_item_price,
            c.avg_freight,
            ROUND((c.category_revenue / m.total_market_revenue) * 100.0, 2) AS revenue_share_pct,
            DENSE_RANK() OVER (ORDER BY c.category_revenue DESC) AS revenue_rank
        FROM category_totals c, overall_market m
        ORDER BY c.category_revenue DESC;
        """
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        return df

    def get_top_customers(self, limit: int = 25) -> pd.DataFrame:
        """Finds top spenders based on customer_unique_id using DENSE_RANK()."""
        query = f"""
        WITH customer_aggregates AS (
            SELECT 
                c.customer_unique_id,
                c.customer_state,
                c.customer_city,
                COUNT(DISTINCT o.order_id) AS total_orders,
                COUNT(oi.order_item_id) AS total_items,
                ROUND(SUM(oi.price), 2) AS total_spend,
                ROUND(AVG(oi.price), 2) AS avg_item_spend
            FROM dim_customers c
            JOIN fact_orders o ON c.customer_id = o.customer_id
            JOIN fact_order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY c.customer_unique_id, c.customer_state, c.customer_city
        )
        SELECT 
            customer_unique_id,
            customer_state,
            customer_city,
            total_orders,
            total_items,
            total_spend,
            avg_item_spend,
            DENSE_RANK() OVER (ORDER BY total_spend DESC) AS customer_rank
        FROM customer_aggregates
        ORDER BY total_spend DESC
        LIMIT {limit};
        """
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        return df

    def get_spend_quartiles(self) -> pd.DataFrame:
        """Analyzes customer spend tiers using NTILE(4) on customer_unique_id."""
        query = """
        WITH customer_spending AS (
            SELECT 
                c.customer_unique_id,
                ROUND(SUM(oi.price), 2) AS total_spend,
                COUNT(DISTINCT o.order_id) AS order_count
            FROM dim_customers c
            JOIN fact_orders o ON c.customer_id = o.customer_id
            JOIN fact_order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY c.customer_unique_id
        ),
        quartiles AS (
            SELECT 
                customer_unique_id,
                total_spend,
                order_count,
                NTILE(4) OVER (ORDER BY total_spend DESC) AS spend_quartile
            FROM customer_spending
        )
        SELECT 
            spend_quartile,
            COUNT(customer_unique_id) AS customer_count,
            ROUND(SUM(total_spend), 2) AS group_total_spend,
            ROUND(AVG(total_spend), 2) AS group_avg_spend,
            ROUND(AVG(order_count), 2) AS group_avg_orders
        FROM quartiles
        GROUP BY spend_quartile
        ORDER BY spend_quartile ASC;
        """
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        return df

    def get_cohort_retention_matrix(self) -> pd.DataFrame:
        """Executes full monthly cohort retention analysis on customer_unique_id."""
        query = """
        WITH customer_first_purchase AS (
            SELECT 
                c.customer_unique_id,
                MIN(o.order_purchase_timestamp) AS first_order_ts,
                substr(MIN(o.order_purchase_timestamp), 1, 7) AS cohort_month
            FROM fact_orders o
            JOIN dim_customers c ON o.customer_id = c.customer_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY c.customer_unique_id
        ),
        customer_order_activity AS (
            SELECT DISTINCT
                c.customer_unique_id,
                fp.cohort_month,
                substr(o.order_purchase_timestamp, 1, 7) AS activity_month,
                (CAST(substr(o.order_purchase_timestamp, 1, 4) AS INTEGER) - CAST(substr(fp.first_order_ts, 1, 4) AS INTEGER)) * 12 +
                (CAST(substr(o.order_purchase_timestamp, 6, 2) AS INTEGER) - CAST(substr(fp.first_order_ts, 6, 2) AS INTEGER)) AS cohort_index
            FROM fact_orders o
            JOIN dim_customers c ON o.customer_id = c.customer_id
            JOIN customer_first_purchase fp ON c.customer_unique_id = fp.customer_unique_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
        ),
        cohort_base_sizes AS (
            SELECT 
                cohort_month,
                COUNT(DISTINCT customer_unique_id) AS cohort_size
            FROM customer_first_purchase
            GROUP BY cohort_month
        ),
        monthly_active_retention AS (
            SELECT 
                ca.cohort_month,
                ca.cohort_index,
                COUNT(DISTINCT ca.customer_unique_id) AS active_customers
            FROM customer_order_activity ca
            GROUP BY ca.cohort_month, ca.cohort_index
        )
        SELECT 
            mar.cohort_month,
            cbs.cohort_size,
            mar.cohort_index,
            mar.active_customers,
            ROUND((CAST(mar.active_customers AS REAL) / cbs.cohort_size) * 100.0, 2) AS retention_rate_pct
        FROM monthly_active_retention mar
        JOIN cohort_base_sizes cbs ON mar.cohort_month = cbs.cohort_month
        WHERE mar.cohort_index >= 0 AND mar.cohort_index <= 12
        ORDER BY mar.cohort_month, mar.cohort_index;
        """
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        return df

    def run_all_and_save_reports(self) -> Dict[str, pd.DataFrame]:
        """Runs all core SQL analytical workflows and exports CSV reports."""
        logger.info(f"Executing SQL analytics suite against active database ({self.backend})...")
        monthly_rev = self.get_monthly_revenue_trends()
        categories = self.get_category_performance()
        top_cust = self.get_top_customers(limit=25)
        quartiles = self.get_spend_quartiles()
        cohorts = self.get_cohort_retention_matrix()

        monthly_rev.to_csv(self.reports_dir / "monthly_revenue_trends.csv", index=False)
        categories.to_csv(self.reports_dir / "category_performance.csv", index=False)
        top_cust.to_csv(self.reports_dir / "top_vip_customers.csv", index=False)
        quartiles.to_csv(self.reports_dir / "customer_spend_quartiles.csv", index=False)
        cohorts.to_csv(self.reports_dir / "cohort_retention.csv", index=False)

        logger.info(f"SQL analytics reports successfully exported to {self.reports_dir}")
        return {
            "monthly_revenue": monthly_rev,
            "category_performance": categories,
            "top_customers": top_cust,
            "spend_quartiles": quartiles,
            "cohort_retention": cohorts,
        }


if __name__ == "__main__":
    engine = SQLAnalyticsEngine()
    reports = engine.run_all_and_save_reports()
    print("SQL analytics executed successfully.")

