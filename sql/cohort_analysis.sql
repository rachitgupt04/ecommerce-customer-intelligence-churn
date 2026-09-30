-- =====================================================================
-- E-Commerce Monthly Cohort Retention Analysis
-- Tracks customer acquisition cohorts and retention rates over 12 months
-- =====================================================================

WITH first_purchase AS (
    -- Step 1: Determine the acquisition cohort (month of first purchase) for each customer
    SELECT 
        customer_id,
        MIN(order_purchase_timestamp) AS first_order_time,
        strftime('%Y-%m', MIN(order_purchase_timestamp)) AS cohort_month
    FROM orders
    WHERE order_status NOT IN ('cancelled')
    GROUP BY customer_id
),
customer_activities AS (
    -- Step 2: Track all distinct active activity months for each customer
    SELECT DISTINCT
        o.customer_id,
        fp.cohort_month,
        strftime('%Y-%m', o.order_purchase_timestamp) AS activity_month,
        -- Calculate month offset (period index)
        (CAST(strftime('%Y', o.order_purchase_timestamp) AS INTEGER) - CAST(strftime('%Y', fp.first_order_time) AS INTEGER)) * 12 +
        (CAST(strftime('%m', o.order_purchase_timestamp) AS INTEGER) - CAST(strftime('%m', fp.first_order_time) AS INTEGER)) AS cohort_index
    FROM orders o
    JOIN first_purchase fp ON o.customer_id = fp.customer_id
    WHERE o.order_status NOT IN ('cancelled')
),
cohort_sizes AS (
    -- Step 3: Count total new customers acquired in each cohort (Month 0 size)
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_id) AS cohort_size
    FROM first_purchase
    GROUP BY cohort_month
),
retention_counts AS (
    -- Step 4: Count surviving customers per cohort per month index
    SELECT 
        ca.cohort_month,
        ca.cohort_index,
        COUNT(DISTINCT ca.customer_id) AS active_customers
    FROM customer_activities ca
    GROUP BY ca.cohort_month, ca.cohort_index
)
-- Step 5: Final output with active counts and retention percentage
SELECT 
    rc.cohort_month,
    cs.cohort_size,
    rc.cohort_index,
    rc.active_customers,
    ROUND((CAST(rc.active_customers AS REAL) / cs.cohort_size) * 100.0, 2) AS retention_rate_pct
FROM retention_counts rc
JOIN cohort_sizes cs ON rc.cohort_month = cs.cohort_month
WHERE rc.cohort_index >= 0 AND rc.cohort_index <= 12
ORDER BY rc.cohort_month, rc.cohort_index;
