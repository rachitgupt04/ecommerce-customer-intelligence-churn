# Data Dictionary: Cleaned Olist Relational Data Warehouse

Comprehensive documentation of all 9 relational tables, data types, constraints, keys, and analytical purposes.

---

### 1. `dim_customers`
Dimension table storing customer account tokens, master consumer identities, and geographic locations.

| Column Name | Data Type | Nullable | Primary Key | Foreign Key | Description | Analytical Purpose |
|---|---|:---:|:---:|:---:|---|---|
| `customer_id` | `VARCHAR(50)` | No | **PK** | No | Unique order-scoped customer transaction token | Joining orders to customer records |
| `customer_unique_id` | `VARCHAR(50)` | No | No | No | Unique real-world human customer identifier | **Core analytical key for RFM, LTV, and churn modeling** |
| `customer_zip_code_prefix`| `INTEGER` | No | No | `dim_geolocation.geolocation_zip_code_prefix` | First 5 digits of Brazilian postal code | Regional segmentation & spatial logistics |
| `customer_city` | `VARCHAR(100)` | No | No | No | Customer registered city (Title Case) | City-level market penetration analysis |
| `customer_state` | `VARCHAR(10)` | No | No | No | Two-letter Brazilian federative state code | State-level revenue and retention analysis |

---

### 2. `dim_products`
Dimension table storing catalog products, translated category taxonomies, and physical dimensions.

| Column Name | Data Type | Nullable | Primary Key | Foreign Key | Description | Analytical Purpose |
|---|---|:---:|:---:|:---:|---|---|
| `product_id` | `VARCHAR(50)` | No | **PK** | No | Unique product hash identifier | Product-level tracking & basket diversity |
| `product_category_name` | `VARCHAR(100)` | No | No | `dim_category_translation.product_category_name` | Original Portuguese category name | Raw category lookup |
| `product_category_name_english` | `VARCHAR(100)` | No | No | No | English translated category name | Reporting, category revenue, diversity metrics |
| `product_name_length` | `INTEGER` | Yes | No | No | Character count of product title | Listing optimization & metadata completeness |
| `product_description_length` | `INTEGER` | Yes | No | No | Character count of product description | Catalog depth analysis |
| `product_photos_qty` | `INTEGER` | Yes | No | No | Number of published product photos | Listing quality vs conversion analysis |
| `product_weight_g` | `NUMERIC(10,2)` | Yes | No | No | Product weight in grams | Freight cost modeling |
| `product_length_cm` | `NUMERIC(10,2)` | Yes | No | No | Package length in centimeters | Volumetric freight calculations |
| `product_height_cm` | `NUMERIC(10,2)` | Yes | No | No | Package height in centimeters | Volumetric freight calculations |
| `product_width_cm` | `NUMERIC(10,2)` | Yes | No | No | Package width in centimeters | Volumetric freight calculations |

---

### 3. `dim_sellers`
Dimension table storing marketplace third-party merchants and fulfillment hubs.

| Column Name | Data Type | Nullable | Primary Key | Foreign Key | Description | Analytical Purpose |
|---|---|:---:|:---:|:---:|---|---|
| `seller_id` | `VARCHAR(50)` | No | **PK** | No | Unique merchant identifier | Seller performance & fulfillment latency |
| `seller_zip_code_prefix`| `INTEGER` | No | No | `dim_geolocation.geolocation_zip_code_prefix` | First 5 digits of seller postal code | Distance calculation between seller & buyer |
| `seller_city` | `VARCHAR(100)` | No | No | No | Seller registered city | Merchant concentration analysis |
| `seller_state` | `VARCHAR(10)` | No | No | No | Two-letter Brazilian state code | Regional supply-chain analytics |

---

### 4. `dim_geolocation`
Deduplicated spatial dimension mapping postal code prefixes to coordinates and regions.

| Column Name | Data Type | Nullable | Primary Key | Foreign Key | Description | Analytical Purpose |
|---|---|:---:|:---:|:---:|---|---|
| `geolocation_zip_code_prefix` | `INTEGER` | No | **PK** | No | First 5 digits of Brazilian postal code | Spatial lookup key |
| `latitude` | `DOUBLE PRECISION` | No | No | No | Mean geographic latitude for prefix | Geospatial distance and mapping |
| `longitude` | `DOUBLE PRECISION` | No | No | No | Mean geographic longitude for prefix | Geospatial distance and mapping |
| `city` | `VARCHAR(100)` | No | No | No | Most frequent city for prefix | Standardized city naming |
| `state` | `VARCHAR(10)` | No | No | No | Most frequent state for prefix | Standardized state naming |

---

### 5. `dim_category_translation`
Lookup dimension providing English translations for Brazilian Portuguese product categories.

| Column Name | Data Type | Nullable | Primary Key | Foreign Key | Description | Analytical Purpose |
|---|---|:---:|:---:|:---:|---|---|
| `product_category_name` | `VARCHAR(100)` | No | **PK** | No | Portuguese category name | Dimension join key |
| `product_category_name_english` | `VARCHAR(100)` | No | No | No | English translated category name | User-facing dashboard reporting |

---

### 6. `fact_orders`
Central fact table tracking order lifecycle timestamps and status transitions.

| Column Name | Data Type | Nullable | Primary Key | Foreign Key | Description | Analytical Purpose |
|---|---|:---:|:---:|:---:|---|---|
| `order_id` | `VARCHAR(50)` | No | **PK** | No | Unique order identifier | Primary transaction entity key |
| `customer_id` | `VARCHAR(50)` | No | No | `dim_customers.customer_id` | Foreign key referencing customer token | Linking orders to customer dimensions |
| `order_status` | `VARCHAR(30)` | No | No | No | Current lifecycle status | Filtering completed vs cancelled orders |
| `order_purchase_timestamp` | `TIMESTAMP` | No | No | No | Exact purchase timestamp | **Primary date key for cohort, RFM, and churn** |
| `order_approved_at` | `TIMESTAMP` | Yes | No | No | Payment authorization timestamp | Payment latency & processing friction |
| `order_delivered_carrier_date` | `TIMESTAMP` | Yes | No | No | Carrier pickup timestamp | Seller fulfillment turnaround time |
| `order_delivered_customer_date` | `TIMESTAMP` | Yes | No | No | Final customer delivery timestamp | End-to-end delivery speed & delay analysis |
| `order_estimated_delivery_date` | `TIMESTAMP` | No | No | No | Estimated promised delivery date | Delivery SLA compliance & delay calculations |

---

### 7. `fact_order_items`
Granular line-item fact table recording individual units, prices, and freight fees.

| Column Name | Data Type | Nullable | Primary Key | Foreign Key | Description | Analytical Purpose |
|---|---|:---:|:---:|:---:|---|---|
| `order_id` | `VARCHAR(50)` | No | **PK (1/2)** | `fact_orders.order_id` | Parent order identifier | Fact join key |
| `order_item_id` | `INTEGER` | No | **PK (2/2)** | No | Sequential line item index in order | Quantity aggregation per order |
| `product_id` | `VARCHAR(50)` | No | No | `dim_products.product_id` | Product catalog identifier | Product & category revenue aggregation |
| `seller_id` | `VARCHAR(50)` | No | No | `dim_sellers.seller_id` | Fulfilling seller identifier | Merchant volume & GMV contribution |
| `shipping_limit_date` | `TIMESTAMP` | No | No | No | Seller shipping SLA deadline | Seller fulfillment SLA compliance |
| `price` | `NUMERIC(10,2)` | No | No | No | Unit item price in R$ | Net revenue, GMV, and AOV calculations |
| `freight_value` | `NUMERIC(10,2)` | No | No | No | Item freight shipping fee in R$ | Logistics cost & freight ratio calculations |

---

### 8. `fact_order_payments`
Payment fact table detailing transaction methods, installments, and amounts.

| Column Name | Data Type | Nullable | Primary Key | Foreign Key | Description | Analytical Purpose |
|---|---|:---:|:---:|:---:|---|---|
| `order_id` | `VARCHAR(50)` | No | **PK (1/2)** | `fact_orders.order_id` | Parent order identifier | Fact join key |
| `payment_sequential` | `INTEGER` | No | **PK (2/2)** | No | Payment sequence index (1, 2, 3...) | Multi-payment transaction sequencing |
| `payment_type` | `VARCHAR(30)` | No | No | No | Method (`credit_card`, `boleto`, etc.) | Payment channel preference & risk scoring |
| `payment_installments` | `INTEGER` | No | No | No | Number of installment payments | Credit financing behavior |
| `payment_value` | `NUMERIC(10,2)` | No | No | No | Gross payment amount in R$ | Total revenue reconciled per order |

---

### 9. `fact_order_reviews`
Customer review and satisfaction fact table.

| Column Name | Data Type | Nullable | Primary Key | Foreign Key | Description | Analytical Purpose |
|---|---|:---:|:---:|:---:|---|---|
| `review_id` | `VARCHAR(50)` | No | **PK (1/2)** | No | Unique review identifier | Review record tracking |
| `order_id` | `VARCHAR(50)` | No | **PK (2/2)** | `fact_orders.order_id` | Order being reviewed | Linking CSAT feedback to transactions |
| `review_score` | `INTEGER` | No | No | No | Rating from 1 (poor) to 5 (excellent) | Customer satisfaction & churn risk feature |
| `review_comment_title` | `TEXT` | Yes | No | No | Review headline text | Sentiment & feedback analysis |
| `review_comment_message` | `TEXT` | Yes | No | No | Review detailed feedback message | Text mining & defect discovery |
| `review_creation_date` | `TIMESTAMP` | No | No | No | Date review survey was generated | Review response latency |
| `review_answer_timestamp` | `TIMESTAMP` | No | No | No | Timestamp review was completed | Review turnaround tracking |

---

### 10. `customer_features.parquet` (Engineered Feature Store)
Snapshot-indexed behavioral feature store constructed across quarterly observation cutoffs with point-in-time safety.

| Feature Name | Data Type | Nullable | Source Tables | Description | Modeling Role |
|---|---|:---:|---|---|---|
| `customer_unique_id` | `VARCHAR(50)` | No | `dim_customers` | Master physical customer identifier | Entity Join Key |
| `snapshot_id` | `VARCHAR(10)` | No | Generated | Snapshot identifier (`S1`, `S2`, `S3`, `S4`) | Temporal Split Key |
| `recency_days` | `DOUBLE` | No | `fact_orders` | Days elapsed between last purchase and snapshot cutoff | Top predictive driver of churn |
| `order_frequency` | `INTEGER` | No | `fact_orders` | Count of orders placed prior to snapshot cutoff | Repeat loyalty measure |
| `monetary_total` | `DOUBLE` | No | `fact_order_items` | Cumulative gross merchandise spend prior to cutoff | Customer lifetime value tier |
| `avg_order_value` | `DOUBLE` | No | `fact_order_items` | Mean basket size (`monetary_total / order_frequency`) | Spend capacity indicator |
| `max_order_value` | `DOUBLE` | No | `fact_order_items` | Maximum single-order spend amount | Outlier high-ticket willingness |
| `items_per_order` | `DOUBLE` | No | `fact_order_items` | Mean number of line items purchased per order | Basket complexity |
| `avg_item_price` | `DOUBLE` | No | `fact_order_items` | Mean unit price of purchased items | Price sensitivity indicator |
| `avg_freight_value` | `DOUBLE` | No | `fact_order_items` | Mean shipping fee paid per order | Shipping cost burden |
| `freight_ratio` | `DOUBLE` | No | Computed | `avg_freight_value / avg_order_value` | Shipping friction metric |
| `unique_categories` | `INTEGER` | No | `dim_products` | Count of distinct English categories purchased | Cross-category engagement |
| `avg_review_score` | `DOUBLE` | No | `fact_order_reviews` | Mean customer review rating prior to cutoff (1 to 5) | Customer satisfaction / CSAT |
| `avg_installments` | `DOUBLE` | No | `fact_order_payments`| Mean financing installments per order | Credit engagement behavior |
| `pct_credit_card` | `DOUBLE` | No | `fact_order_payments`| Percentage of spend paid via credit card | Payment channel preference |
| `pct_boleto` | `DOUBLE` | No | `fact_order_payments`| Percentage of spend paid via Brazilian boleto | Payment channel preference |
| `pct_voucher` | `DOUBLE` | No | `fact_order_payments`| Percentage of spend paid via discount vouchers | Promotion reliance |
| `avg_delivery_delay_days` | `DOUBLE` | No | `fact_orders` | Days actual delivery exceeded estimated SLA (0 if early) | Fulfillment friction signal |
| `late_delivery_count` | `INTEGER` | No | `fact_orders` | Number of orders delivered after promised SLA date | Logistics defect counter |
| `spending_trend_velocity`| `DOUBLE` | No | Computed | Ratio of recent 60-day spend to lifetime mean spend | Spend acceleration / deceleration |
| `customer_lifetime_days` | `DOUBLE` | No | `fact_orders` | Days elapsed between first and last purchase prior to cutoff | Customer maturity |
| `is_churned` | `INTEGER` | No | Forward window | Binary target (1 if 0 orders in next 90 days, 0 if repurchased) | **Supervised ML Ground Truth** |

