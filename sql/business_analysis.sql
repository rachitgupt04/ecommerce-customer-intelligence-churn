-- =====================================================================
-- E-Commerce Business Analytics SQL Queries
-- Demonstrating Advanced SQL: CTEs, Window Functions (LAG, LEAD, RANK, NTILE), Aggregations
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Monthly Revenue & Month-over-Month (MoM) Growth Rate
-- ---------------------------------------------------------------------
WITH monthly_sales AS (
    SELECT 
        strftime('%Y-%m', o.order_purchase_timestamp) AS order_month,
        COUNT(DISTINCT o.order_id) AS total_orders,
        COUNT(DISTINCT o.customer_id) AS unique_customers,
        ROUND(SUM(oi.price * oi.quantity - oi.discount_amount), 2) AS net_revenue,
        ROUND(AVG(oi.price * oi.quantity - oi.discount_amount), 2) AS avg_item_revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status NOT IN ('cancelled')
    GROUP BY strftime('%Y-%m', o.order_purchase_timestamp)
),
sales_with_lag AS (
    SELECT 
        order_month,
        total_orders,
        unique_customers,
        net_revenue,
        LAG(net_revenue, 1) OVER (ORDER BY order_month) AS prev_month_revenue
    FROM monthly_sales
)
SELECT 
    order_month,
    total_orders,
    unique_customers,
    net_revenue,
    prev_month_revenue,
    ROUND(((net_revenue - prev_month_revenue) / prev_month_revenue) * 100.0, 2) AS mom_growth_pct
FROM sales_with_lag
ORDER BY order_month;


-- ---------------------------------------------------------------------
-- 2. Category Performance & Revenue Share
-- ---------------------------------------------------------------------
WITH category_metrics AS (
    SELECT 
        p.category_name,
        COUNT(DISTINCT o.order_id) AS orders_count,
        SUM(oi.quantity) AS units_sold,
        ROUND(SUM(oi.price * oi.quantity - oi.discount_amount), 2) AS category_revenue,
        ROUND(AVG(oi.discount_amount / (oi.price * oi.quantity + 0.0001)) * 100.0, 2) AS avg_discount_pct
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_status NOT IN ('cancelled')
    GROUP BY p.category_name
),
total_summary AS (
    SELECT SUM(category_revenue) AS total_market_revenue FROM category_metrics
)
SELECT 
    cm.category_name,
    cm.orders_count,
    cm.units_sold,
    cm.category_revenue,
    cm.avg_discount_pct,
    ROUND((cm.category_revenue / ts.total_market_revenue) * 100.0, 2) AS revenue_share_pct,
    DENSE_RANK() OVER (ORDER BY cm.category_revenue DESC) AS revenue_rank
FROM category_metrics cm, total_summary ts
ORDER BY cm.category_revenue DESC;


-- ---------------------------------------------------------------------
-- 3. Top 10 Most Valuable Customers (Ranked with DENSE_RANK)
-- ---------------------------------------------------------------------
WITH customer_spending AS (
    SELECT 
        c.customer_id,
        c.customer_city,
        c.customer_state,
        COUNT(DISTINCT o.order_id) AS lifetime_orders,
        ROUND(SUM(oi.price * oi.quantity - oi.discount_amount), 2) AS total_spent,
        ROUND(AVG(oi.price * oi.quantity - oi.discount_amount), 2) AS avg_spend_per_order
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status NOT IN ('cancelled')
    GROUP BY c.customer_id, c.customer_city, c.customer_state
)
SELECT 
    customer_id,
    customer_city,
    customer_state,
    lifetime_orders,
    total_spent,
    avg_spend_per_order,
    DENSE_RANK() OVER (ORDER BY total_spent DESC) AS customer_rank
FROM customer_spending
LIMIT 10;


-- ---------------------------------------------------------------------
-- 4. Customer Spend Quartiles using NTILE(4)
-- ---------------------------------------------------------------------
WITH customer_ltv AS (
    SELECT 
        c.customer_id,
        ROUND(SUM(oi.price * oi.quantity - oi.discount_amount), 2) AS total_spend,
        COUNT(DISTINCT o.order_id) AS total_orders
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status NOT IN ('cancelled')
    GROUP BY c.customer_id
),
customer_quartiles AS (
    SELECT 
        customer_id,
        total_spend,
        total_orders,
        NTILE(4) OVER (ORDER BY total_spend DESC) AS spend_quartile
    FROM customer_ltv
)
SELECT 
    spend_quartile,
    COUNT(customer_id) AS customer_count,
    ROUND(SUM(total_spend), 2) AS group_total_spend,
    ROUND(AVG(total_spend), 2) AS group_avg_spend,
    ROUND(AVG(total_orders), 2) AS group_avg_orders
FROM customer_quartiles
GROUP BY spend_quartile
ORDER BY spend_quartile ASC;


-- ---------------------------------------------------------------------
-- 5. Inter-Purchase Time Gap Analysis Using LAG() Window Function
-- ---------------------------------------------------------------------
WITH order_sequence AS (
    SELECT 
        customer_id,
        order_id,
        order_purchase_timestamp,
        LAG(order_purchase_timestamp, 1) OVER (
            PARTITION BY customer_id 
            ORDER BY order_purchase_timestamp
        ) AS previous_order_timestamp
    FROM orders
    WHERE order_status NOT IN ('cancelled')
),
days_between_orders AS (
    SELECT 
        customer_id,
        order_id,
        order_purchase_timestamp,
        previous_order_timestamp,
        ROUND(JULIANDAY(order_purchase_timestamp) - JULIANDAY(previous_order_timestamp), 1) AS days_since_prev_order
    FROM order_sequence
    WHERE previous_order_timestamp IS NOT NULL
)
SELECT 
    customer_id,
    COUNT(order_id) AS repeat_orders_count,
    ROUND(AVG(days_since_prev_order), 1) AS avg_days_between_orders,
    ROUND(MIN(days_since_prev_order), 1) AS min_days_between_orders,
    ROUND(MAX(days_since_prev_order), 1) AS max_days_between_orders
FROM days_between_orders
GROUP BY customer_id
ORDER BY repeat_orders_count DESC
LIMIT 15;
