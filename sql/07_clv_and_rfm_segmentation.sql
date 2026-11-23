-- ============================================================================
-- 07_clv_and_rfm_segmentation.sql
-- Customer lifetime value, RFM segmentation, and the retention priority
-- list.
--
-- Definitions
--   * Historical CLV: the total monetary value of all valid orders a
--     customer has ever placed up to the anchor date (the most recent
--     order date in the data). Valid means status is not Cancelled and
--     not Returned, the same rule as the churn features.
--   * RFM scores: recency, frequency and monetary are each split into
--     quintiles with NTILE(5). Recency is scored so that 5 means most
--     recent. Frequency and monetary score 5 as highest.
--   * Segments are assigned by an ordered CASE over the R and F scores:
--       Champions          R >= 4 and F >= 4
--       Loyal              R >= 3 and F >= 3
--       Potential Loyalist R >= 4 and F in (2, 3)
--       New                R = 5 and F = 1
--       At Risk            R <= 2 and F >= 3
--       Hibernating        R <= 2 and F = 2
--       Lost               R <= 2 and F = 1
--       Others             everything else
--   * Retention priority: current churn probability (from
--     06_prediction_and_deciles.sql) multiplied by historical CLV. A
--     customer who is both valuable and likely to leave is where a
--     retention offer earns the most.
-- ============================================================================

CREATE OR REPLACE TABLE thelook_ml.customer_clv_rfm AS
WITH params AS (
  SELECT DATE(MAX(created_at)) AS anchor_date
  FROM `bigquery-public-data.thelook_ecommerce.orders`
),
valid_orders AS (
  SELECT o.order_id, o.user_id, o.created_at
  FROM `bigquery-public-data.thelook_ecommerce.orders` o
  WHERE o.status NOT IN ('Cancelled', 'Returned')
),
order_values AS (
  SELECT oi.order_id, SUM(oi.sale_price) AS order_value
  FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
  JOIN valid_orders vo USING (order_id)
  WHERE oi.status NOT IN ('Cancelled', 'Returned')
  GROUP BY oi.order_id
),
rfm_raw AS (
  SELECT
    vo.user_id,
    DATE_DIFF(p.anchor_date, DATE(MAX(vo.created_at)), DAY) AS recency_days,
    COUNT(*) AS frequency,
    SUM(ov.order_value) AS monetary
  FROM valid_orders vo
  JOIN order_values ov USING (order_id)
  CROSS JOIN params p
  GROUP BY vo.user_id, p.anchor_date
),
rfm_scored AS (
  SELECT
    user_id,
    recency_days,
    frequency,
    monetary,
    -- NTILE over recency ascending gives 1 to the most recent customers,
    -- so flip it: 6 - tile makes 5 the most recent score.
    6 - NTILE(5) OVER (ORDER BY recency_days ASC) AS r_score,
    NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
    NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
  FROM rfm_raw
)
SELECT
  user_id,
  recency_days,
  frequency,
  monetary AS historical_clv,
  r_score,
  f_score,
  m_score,
  CASE
    WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
    WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
    WHEN r_score >= 4 AND f_score IN (2, 3) THEN 'Potential Loyalist'
    WHEN r_score = 5 AND f_score = 1 THEN 'New'
    WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
    WHEN r_score <= 2 AND f_score = 2 THEN 'Hibernating'
    WHEN r_score <= 2 AND f_score = 1 THEN 'Lost'
    ELSE 'Others'
  END AS segment
FROM rfm_scored;

-- Segment summary: size, share and value per segment.
SELECT
  segment,
  COUNT(*) AS customers,
  ROUND(COUNT(*) / SUM(COUNT(*)) OVER () * 100, 2) AS customer_share_pct,
  ROUND(AVG(historical_clv), 2) AS avg_clv,
  ROUND(SUM(historical_clv), 2) AS total_clv,
  ROUND(AVG(frequency), 2) AS avg_frequency,
  ROUND(AVG(recency_days), 1) AS avg_recency_days
FROM thelook_ml.customer_clv_rfm
GROUP BY segment
ORDER BY total_clv DESC;

-- Retention priority list: valuable customers most likely to churn.
-- expected_value_at_risk is the simple product of churn probability and
-- historical CLV, a ranking heuristic, not a forecast.
SELECT
  c.user_id,
  s.churn_prob,
  c.historical_clv,
  ROUND(s.churn_prob * c.historical_clv, 2) AS expected_value_at_risk,
  c.segment,
  c.frequency,
  c.recency_days
FROM thelook_ml.customer_clv_rfm c
JOIN thelook_ml.current_churn_scores s USING (user_id)
ORDER BY expected_value_at_risk DESC
LIMIT 100;
