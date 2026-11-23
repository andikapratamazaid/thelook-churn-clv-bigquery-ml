-- ============================================================================
-- 02_train_logistic_regression.sql
-- Train the baseline churn model: BigQuery ML logistic regression.
--
-- The training query lists the feature columns explicitly so that the
-- customer key (user_id) is never used as a feature. BigQuery ML treats
-- STRING columns as categorical and numeric columns as numeric.
-- A random 80/20 split is used; the held out 20 percent is what
-- ML.EVALUATE scores in 04_evaluation.sql.
-- ============================================================================

CREATE OR REPLACE MODEL thelook_ml.churn_logreg
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['churn'],
  data_split_method = 'RANDOM',
  data_split_eval_fraction = 0.2,
  l2_reg = 1.0,
  max_iterations = 50
) AS
SELECT
  churn,
  recency_days,
  frequency,
  monetary,
  total_items,
  avg_order_value,
  avg_basket_items,
  return_rate,
  sessions,
  total_events,
  product_views,
  cart_events,
  purchase_events,
  cart_to_purchase_ratio,
  tenure_days,
  age,
  age_band,
  gender,
  traffic_source,
  country,
  favorite_category
FROM thelook_ml.churn_features;
