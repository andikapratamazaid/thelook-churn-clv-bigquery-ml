-- ============================================================================
-- 06_prediction_and_deciles.sql
-- Two jobs:
--   A. Decile analysis on the labeled cohort. Customers are ranked by
--      predicted churn probability and split into ten equal groups. If the
--      model is useful, the actual churn rate must fall steadily from the
--      highest risk decile to the lowest. This is the business facing proof
--      that the score means something.
--   B. Score currently active customers. The same feature pipeline is
--      rebuilt with the cutoff moved to the anchor date itself (the most
--      recent order date), so the observation window is the last 6 months
--      and there is no label. These scores feed the retention priority
--      list in 07_clv_and_rfm_segmentation.sql.
-- ============================================================================

-- --------------------------------------------------------------------------
-- A. Decile analysis on the labeled cohort (boosted tree scores).
-- --------------------------------------------------------------------------
CREATE OR REPLACE TABLE thelook_ml.cohort_predictions AS
SELECT
  user_id,
  churn,
  (SELECT prob FROM UNNEST(predicted_churn_probs) WHERE label = 1) AS churn_prob
FROM ML.PREDICT(
  MODEL thelook_ml.churn_boosted_tree,
  (SELECT * FROM thelook_ml.churn_features)
);

SELECT
  NTILE(10) OVER (ORDER BY churn_prob DESC) AS risk_decile,
  COUNT(*) AS customers,
  ROUND(AVG(churn_prob), 4) AS avg_predicted_prob,
  ROUND(AVG(churn), 4) AS actual_churn_rate,
  MIN(churn_prob) AS min_prob,
  MAX(churn_prob) AS max_prob
FROM thelook_ml.cohort_predictions
GROUP BY risk_decile
ORDER BY risk_decile;

-- --------------------------------------------------------------------------
-- B. Features for currently active customers, anchored at the latest
--    order date. Identical feature logic to 01_feature_engineering.sql,
--    only the window moves and the label step is dropped.
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW thelook_ml.current_features AS
WITH params AS (
  SELECT
    DATE(MAX(created_at)) AS cutoff_date,
    DATE_SUB(DATE(MAX(created_at)), INTERVAL 6 MONTH) AS obs_start
  FROM `bigquery-public-data.thelook_ecommerce.orders`
),
obs_orders AS (
  SELECT o.order_id, o.user_id, o.created_at
  FROM `bigquery-public-data.thelook_ecommerce.orders` o
  CROSS JOIN params p
  WHERE o.status NOT IN ('Cancelled', 'Returned')
    AND DATE(o.created_at) >= p.obs_start
    AND DATE(o.created_at) <= p.cutoff_date
),
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
    AND DATE(e.created_at) <= p.cutoff_date
  GROUP BY e.user_id
),
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
)
SELECT
  u.id AS user_id,
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
JOIN `bigquery-public-data.thelook_ecommerce.users` u ON u.id = of.user_id
LEFT JOIN event_features ef ON ef.user_id = of.user_id
LEFT JOIN favorite_category fc ON fc.user_id = of.user_id
CROSS JOIN params p;

-- Score the current customers with the boosted tree model.
CREATE OR REPLACE TABLE thelook_ml.current_churn_scores AS
SELECT
  user_id,
  (SELECT prob FROM UNNEST(predicted_churn_probs) WHERE label = 1) AS churn_prob
FROM ML.PREDICT(
  MODEL thelook_ml.churn_boosted_tree,
  (SELECT * FROM thelook_ml.current_features)
);
