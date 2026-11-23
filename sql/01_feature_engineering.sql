-- ============================================================================
-- 01_feature_engineering.sql
-- TheLook Churn and CLV, BigQuery ML Customer Retention Analytics
--
-- Purpose
--   Build the labeled feature table used to train the churn models.
--
-- Churn definition (used consistently across this whole project)
--   * Anchor: the maximum order date present in the orders table.
--   * Cutoff: 90 days before that anchor date.
--   * Observation window: the 6 months before the cutoff. All features are
--     computed from this window only, so no future information leaks in.
--   * Label window: the 90 days starting at the cutoff.
--   * Cohort: customers with at least one valid order in the observation
--     window.
--   * Label: churn = 1 when the customer placed NO valid order in the label
--     window, churn = 0 otherwise.
--   * A "valid order" is an order whose status is not Cancelled and not
--     Returned. Cancelled orders were never fulfilled and Returned orders
--     were sent back, so neither counts as a purchase.
--
-- Output
--   thelook_ml.churn_features: one row per cohort customer with the label
--   and every feature column. Run in your own project; the sandbox is free.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS thelook_ml;

CREATE OR REPLACE VIEW thelook_ml.churn_features AS
WITH params AS (
  SELECT
    DATE(MAX(created_at)) AS anchor_date,
    DATE_SUB(DATE(MAX(created_at)), INTERVAL 90 DAY) AS cutoff_date,
    DATE_SUB(DATE_SUB(DATE(MAX(created_at)), INTERVAL 90 DAY), INTERVAL 6 MONTH) AS obs_start
  FROM `bigquery-public-data.thelook_ecommerce.orders`
),
-- Valid orders inside the observation window, one row per order.
obs_orders AS (
  SELECT o.order_id, o.user_id, o.created_at
  FROM `bigquery-public-data.thelook_ecommerce.orders` o
  CROSS JOIN params p
  WHERE o.status NOT IN ('Cancelled', 'Returned')
    AND DATE(o.created_at) >= p.obs_start
    AND DATE(o.created_at) < p.cutoff_date
),
-- Monetary value and item counts per order. Cancelled items are ignored
-- entirely. Returned items count toward returned_items (for the return
-- rate) but contribute no revenue and no kept item count.
order_values AS (
  SELECT
    oo.order_id,
    COALESCE(SUM(IF(oi.status != 'Returned', oi.sale_price, 0)), 0) AS order_value,
    COUNTIF(oi.status != 'Returned') AS item_count,
    COUNTIF(oi.status = 'Returned') AS returned_items
  FROM obs_orders oo
  LEFT JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi
    ON oi.order_id = oo.order_id
    AND oi.status != 'Cancelled'
  GROUP BY oo.order_id
),
-- Behavioural aggregates per customer from the orders side.
order_features AS (
  SELECT
    oo.user_id,
    DATE_DIFF(p.cutoff_date, DATE(MAX(oo.created_at)), DAY) AS recency_days,
    COUNT(*) AS frequency,
    SUM(ov.order_value) AS monetary,
    SUM(ov.item_count) AS total_items,
    SUM(ov.order_value) / COUNT(*) AS avg_order_value,
    SUM(ov.item_count) / COUNT(*) AS avg_basket_items,
    SAFE_DIVIDE(SUM(ov.returned_items), SUM(ov.item_count)) AS return_rate
  FROM obs_orders oo
  JOIN order_values ov USING (order_id)
  CROSS JOIN params p
  GROUP BY oo.user_id, p.cutoff_date
),
-- Behavioural aggregates per customer from the web events side.
-- Events with a NULL user_id are anonymous ghost sessions and are ignored.
event_features AS (
  SELECT
    e.user_id,
    COUNT(DISTINCT e.session_id) AS sessions,
    COUNT(*) AS total_events,
    COUNTIF(e.event_type = 'product') AS product_views,
    COUNTIF(e.event_type = 'cart') AS cart_events,
    COUNTIF(e.event_type = 'purchase') AS purchase_events,
    COUNTIF(e.event_type = 'cart')
      / (COUNTIF(e.event_type = 'purchase') + 1) AS cart_to_purchase_ratio
  FROM `bigquery-public-data.thelook_ecommerce.events` e
  CROSS JOIN params p
  WHERE e.user_id IS NOT NULL
    AND DATE(e.created_at) >= p.obs_start
    AND DATE(e.created_at) < p.cutoff_date
  GROUP BY e.user_id
),
-- Most purchased product category per customer in the observation window.
favorite_category AS (
  SELECT user_id, category AS favorite_category
  FROM (
    SELECT
      oi.user_id,
      pr.category,
      COUNT(*) AS n,
      ROW_NUMBER() OVER (PARTITION BY oi.user_id ORDER BY COUNT(*) DESC) AS rn
    FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
    JOIN obs_orders oo ON oi.order_id = oo.order_id
    JOIN `bigquery-public-data.thelook_ecommerce.products` pr
      ON oi.product_id = pr.id
    WHERE oi.status NOT IN ('Cancelled', 'Returned')
    GROUP BY oi.user_id, pr.category
  )
  WHERE rn = 1
),
-- The label: 1 when no valid order appears in the 90 day label window.
labels AS (
  SELECT
    of.user_id,
    IF(COUNT(o.order_id) = 0, 1, 0) AS churn
  FROM order_features of
  CROSS JOIN params p
  LEFT JOIN `bigquery-public-data.thelook_ecommerce.orders` o
    ON o.user_id = of.user_id
    AND o.status NOT IN ('Cancelled', 'Returned')
    AND DATE(o.created_at) >= p.cutoff_date
    AND DATE(o.created_at) < DATE_ADD(p.cutoff_date, INTERVAL 90 DAY)
  GROUP BY of.user_id
)
SELECT
  u.id AS user_id,
  l.churn,
  of.recency_days,
  of.frequency,
  of.monetary,
  of.total_items,
  of.avg_order_value,
  of.avg_basket_items,
  of.return_rate,
  COALESCE(ef.sessions, 0) AS sessions,
  COALESCE(ef.total_events, 0) AS total_events,
  COALESCE(ef.product_views, 0) AS product_views,
  COALESCE(ef.cart_events, 0) AS cart_events,
  COALESCE(ef.purchase_events, 0) AS purchase_events,
  COALESCE(ef.cart_to_purchase_ratio, 0) AS cart_to_purchase_ratio,
  DATE_DIFF(p.cutoff_date, DATE(u.created_at), DAY) AS tenure_days,
  u.age,
  CASE
    WHEN u.age < 25 THEN 'under_25'
    WHEN u.age < 35 THEN '25_34'
    WHEN u.age < 45 THEN '35_44'
    WHEN u.age < 55 THEN '45_54'
    ELSE '55_plus'
  END AS age_band,
  u.gender,
  u.traffic_source,
  u.country,
  fc.favorite_category
FROM order_features of
JOIN labels l USING (user_id)
JOIN `bigquery-public-data.thelook_ecommerce.users` u ON u.id = of.user_id
LEFT JOIN event_features ef ON ef.user_id = of.user_id
LEFT JOIN favorite_category fc ON fc.user_id = of.user_id
CROSS JOIN params p;

-- Cohort sanity check: cohort size and observed churn rate.
SELECT
  COUNT(*) AS cohort_customers,
  SUM(churn) AS churned_customers,
  ROUND(AVG(churn), 4) AS churn_rate
FROM thelook_ml.churn_features;
