# Data Pipeline Documentation: Real Olist E-Commerce Ingestion

This document details the production data ingestion, cleaning, and relational database warehouse pipeline implemented for the Brazilian E-Commerce Public Dataset by Olist.

---

## 1. Raw Dataset Source

The pipeline ingests raw transactional CSVs located at `data/raw/`:
- `olist_customers_dataset.csv` (99,441 records)
- `olist_orders_dataset.csv` (99,441 records)
- `olist_order_items_dataset.csv` (112,650 records)
- `olist_order_payments_dataset.csv` (103,886 records)
- `olist_products_dataset.csv` (32,951 records)
- `olist_order_reviews_dataset.csv` (99,224 records)
- `olist_sellers_dataset.csv` (3,095 records)
- `olist_geolocation_dataset.csv` (1,000,163 records)
- `product_category_name_translation.csv` (71 records)

---

## 2. Cleaning & Transformation Rules

### A. Customers Dimension (`dim_customers`)
- **Customer Identity Model:** Preserves both `customer_id` and `customer_unique_id`.
  - `customer_id` represents an order-scoped identifier (unique per order transaction).
  - `customer_unique_id` represents the unique human consumer. All analytical metrics (RFM, LTV, repeat purchase, churn) must aggregate on `customer_unique_id`.
- **Location Normalization:** Standardizes `customer_city` to Title Case and `customer_state` to 2-letter uppercase.

### B. Products Dimension (`dim_products`)
- **Category Preservation:** 610 products with missing categories are preserved and assigned to `'unknown_category'` rather than deleted.
- **Category Translations:** Merges Portuguese categories with English translations from `product_category_name_translation.csv`.
- **Missing Translations Handled:** Adds explicit mappings for Portuguese categories missing from the translation file:
  - `pc_gamer` $\rightarrow$ `pc_gamer`
  - `portateis_cozinha_e_preparadores_de_alimentos` $\rightarrow$ `small_appliances_kitchen_and_food_preparers`
- **Physical Attributes:** Renames `product_name_lenght` and `product_description_lenght` to standard English spelling (`length`).

### C. Geolocation Dimension (`dim_geolocation`)
- **Aggregation Logic:** The raw table contains 1,000,163 rows with 261,831 exact duplicate rows and multiple coordinate pairs per postal code prefix.
- The pipeline aggregates the table by `geolocation_zip_code_prefix` into exactly **19,015 clean unique rows**:
  - `latitude`: Arithmetic mean of all latitude readings for the prefix.
  - `longitude`: Arithmetic mean of all longitude readings for the prefix.
  - `city`: Deterministic mode (most frequently occurring city name for the prefix).
  - `state`: Deterministic mode (most frequently occurring state abbreviation for the prefix).

### D. Orders Fact Table (`fact_orders`)
- **Timestamp Parsing:** Strictly parses all 5 timestamp columns into standard ISO datetimes:
  - `order_purchase_timestamp`
  - `order_approved_at`
  - `order_delivered_carrier_date`
  - `order_delivered_customer_date`
  - `order_estimated_delivery_date`
- **Missing Timestamp Integrity:** Missing delivery timestamps (2,965 orders) and carrier dates (1,783 orders) are preserved as `NULL`/`NaT` without fabricating synthetic dates.
- **Status Preservation:** Keeps all 8 raw statuses (`delivered`, `shipped`, `canceled`, `unavailable`, `invoiced`, `processing`, `created`, `approved`).

### E. Order Items Fact Table (`fact_order_items`)
- **Composite Primary Key:** `(order_id, order_item_id)` uniquely identifies each line item.
- **Monetary Integrity:** Validates that `price` is strictly positive and `freight_value` is non-negative.
- **Timestamps:** Parses `shipping_limit_date` to ISO datetime.

### F. Order Payments Fact Table (`fact_order_payments`)
- **Composite Primary Key:** `(order_id, payment_sequential)`.
- **Payment Handling:** Accommodates split payments (multiple payment methods per order).
- **Payment Types:** `credit_card`, `boleto`, `voucher`, `debit_card`, and `not_defined`.

### G. Order Reviews Fact Table (`fact_order_reviews`)
- **Composite Primary Key:** `(review_id, order_id)`.
- **Scores & Timestamps:** Preserves integer rating `review_score` (1 to 5) and parses `review_creation_date` and `review_answer_timestamp`.

---

## 3. Relational / Star Schema Architecture

```text
Dimension Tables:
  - dim_category_translation  (PK: product_category_name)
  - dim_products              (PK: product_id, FK: product_category_name)
  - dim_geolocation           (PK: geolocation_zip_code_prefix)
  - dim_customers             (PK: customer_id, Indexed: customer_unique_id)
  - dim_sellers               (PK: seller_id)

Fact Tables:
  - fact_orders               (PK: order_id, FK: customer_id)
  - fact_order_items          (PK: [order_id, order_item_id], FKs: order_id, product_id, seller_id)
  - fact_order_payments       (PK: [order_id, payment_sequential], FK: order_id)
  - fact_order_reviews        (PK: [review_id, order_id], FK: order_id)
```

---

## 4. Automated Validation Rules

The validation engine (`src/data/ingestion/validate_olist.py`) enforces:
1. **Row Count Thresholds:** Verifies minimum row counts on all tables.
2. **Primary Key Uniqueness:** Confirms zero duplicate or null primary keys across all 9 tables.
3. **Foreign Key Referential Integrity:**
   - Every `fact_orders.customer_id` exists in `dim_customers.customer_id`.
   - Every `fact_order_items.order_id` exists in `fact_orders.order_id`.
   - Every `fact_order_items.product_id` exists in `dim_products.product_id`.
   - Every `fact_order_items.seller_id` exists in `dim_sellers.seller_id`.
   - Every `fact_order_payments.order_id` exists in `fact_orders.order_id`.
   - Every `fact_order_reviews.order_id` exists in `fact_orders.order_id`.
4. **Monetary Sanity:** Zero negative prices, freight fees, or payment values.
5. **Timestamp Consistency:** Verifies non-null purchase timestamps.

---

## 5. Known Data Limitations

1. **Low Historical Repurchase Rate:** In Olist, repeat buyers ($\ge 2$ orders) account for 3.12% of consumers (2,997 customers out of 96,096). This produces extreme class imbalance in churn modeling that requires cost-sensitive learning or SMOTE.
2. **Missing Delivery Dates for Open Orders:** Orders with statuses other than `delivered` naturally lack customer delivery timestamps.
3. **Postal Code Precision:** Brazilian CEP prefixes represent neighborhoods rather than exact street addresses, producing multiple coordinate readings in the raw data.

---

## 6. Point-in-Time Temporal Feature Engineering Pipeline

The feature engineering pipeline (`src/features/build_features.py`) constructs ML-ready feature matrices across **4 quarterly temporal snapshots** to prevent look-ahead bias:

| Snapshot | Cutoff Date ($T_{\text{snap}}$) | Role | Observation Window | Performance Target Window |
|---|---|---|---|---|
| **$S_1$** | `2017-12-01` | Train | All orders $< 2017\text{-}12\text{-}01$ | $[2017\text{-}12\text{-}01, 2018\text{-}03\text{-}01)$ |
| **$S_2$** | `2018-03-01` | Train | All orders $< 2018\text{-}03\text{-}01$ | $[2018\text{-}03\text{-}01, 2018\text{-}05\text{-}30)$ |
| **$S_3$** | `2018-06-01` | Validation (Threshold Freeze) | All orders $< 2018\text{-}06\text{-}01$ | $[2018\text{-}06\text{-}01, 2018\text{-}08\text{-}30)$ |
| **$S_4$** | `2018-08-31` | Holdout Test | All orders $< 2018\text{-}08\text{-}31$ | $[2018\text{-}08\text{-}31, 2018\text{-}11\text{-}29)$ |

### In-Flight Delivery Clamping
To prevent target leakage through fulfillment tracking, any order with `order_delivered_customer_date >= T_snap` has its delivery date clamped to `NaT` during snapshot feature extraction.

---

## 7. Lookback Window Architecture

The pipeline supports three lookback window modes evaluated in `scripts/compare_lookback_windows.py`:

1. **180-Day Lookback:** Restricts historical aggregations to $[T_{\text{snap}} - 180\text{d}, T_{\text{snap}})$. Discards 38.5% of panel observations and 3,069 customers.
2. **365-Day Lookback:** Restricts historical aggregations to $[T_{\text{snap}} - 365\text{d}, T_{\text{snap}})$. Discards 7.2% of panel observations.
3. **All-Time Lookback (Champion):** Ingests all historical orders prior to $T_{\text{snap}}$. Retains 100% of panel observations (196,508) and all 77,808 holdout customers, delivering superior discrimination (Holdout ROC-AUC: 0.6112 vs 0.5873).

---

## 8. Relational Database Backend Architecture

The application implements a dual-backend engine abstraction (`src/data/db.py`):
- **PostgreSQL (Primary Enterprise Target):** `postgresql://postgres:***@localhost:5432/ecommerce_olist`.
- **SQLite (Automated Offline Fallback):** `sqlite:///data/processed/olist/olist_warehouse.db` containing all 9 normalized tables (569,774 records, 0 orphans).
- **Backend Transparency:** The UI and audit logs explicitly surface the active database backend to prevent deceptive claims.
