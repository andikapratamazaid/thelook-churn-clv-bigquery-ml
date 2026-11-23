-- ============================================================================
-- 03_train_boosted_tree.sql
-- Train the main churn model: BigQuery ML boosted tree classifier.
--
-- Same features, same split settings as the logistic regression baseline in
-- 02_train_logistic_regression.sql, so the comparison in 04_evaluation.sql
-- is apples to apples. Boosted trees usually win on behavioural data like
-- this because they capture interactions (for example recency combined
-- with frequency) without manual feature crosses.
-- ============================================================================

CREATE OR REPLACE MODEL thelook_ml.churn_boosted_tree
OPTIONS (
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['churn'],
  data_split_method = 'RANDOM',
  data_split_eval_fraction = 0.2,
  max_iterations = 100,
  learn_rate = 0.1,
  max_tree_depth = 6,
  subsample = 0.85,
  l2_reg = 1.0
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
