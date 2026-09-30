-- =====================================================================
-- E-Commerce RFM (Recency, Frequency, Monetary) Segmentation Query
-- Calculates RFM metrics and assigns quintile scores (1-5) using NTILE()
-- =====================================================================

WITH observation_snapshot AS (
    -- Reference date fixed to analysis cutoff point
    SELECT '2024-09-01' AS snapshot_date
),
customer_rfm_raw AS (
    SELECT 
        c.customer_id,
        -- Recency: Days between snapshot date and customer's latest purchase
        CAST(ROUND(JULIANDAY((SELECT snapshot_date FROM observation_snapshot)) - JULIANDAY(MAX(o.order_purchase_timestamp))) AS INTEGER) AS recency,
        -- Frequency: Total distinct delivered orders placed before snapshot
        COUNT(DISTINCT o.order_id) AS frequency,
        -- Monetary: Total net spending before snapshot
        ROUND(SUM(oi.price * oi.quantity - oi.discount_amount), 2) AS monetary
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_purchase_timestamp < (SELECT snapshot_date FROM observation_snapshot)
      AND o.order_status NOT IN ('cancelled')
    GROUP BY c.customer_id
),
rfm_scored AS (
    SELECT 
        customer_id,
        recency,
        frequency,
        monetary,
        -- R Score: Lower recency (more recent) = higher score (5 is best)
        NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
        -- F Score: Higher frequency = higher score (5 is best)
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        -- M Score: Higher monetary spend = higher score (5 is best)
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM customer_rfm_raw
)
SELECT 
    customer_id,
    recency,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    (r_score * 100 + f_score * 10 + m_score) AS rfm_score,
    CASE 
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2 THEN 'Promising / New Customers'
        WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'At Risk High-Value'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score >= 3 THEN 'At Risk Spenders'
        WHEN r_score >= 3 AND f_score <= 2 AND m_score <= 2 THEN 'Potential Loyalists'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN 'Hibernating / Lost'
        ELSE 'Needs Attention'
    END AS rfm_segment
FROM rfm_scored
ORDER BY monetary DESC;
