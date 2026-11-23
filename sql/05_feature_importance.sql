-- ============================================================================
-- 05_feature_importance.sql
-- Inspect what each model relies on.
--
-- ML.WEIGHTS for the boosted tree reports importance scores based on the
-- gain each feature contributes across all trees. For logistic regression
-- it reports the learned coefficient per feature value, which shows both
-- direction and strength: a positive weight pushes the prediction toward
-- churn, a negative weight pushes it away from churn.
-- ============================================================================

-- Boosted tree feature importance, strongest first.
SELECT feature, importance_weight, importance_gain, importance_cover
FROM ML.WEIGHTS(MODEL thelook_ml.churn_boosted_tree)
ORDER BY importance_gain DESC;

-- Logistic regression weights, strongest absolute effect first.
SELECT feature, weight, category_weights
FROM ML.WEIGHTS(MODEL thelook_ml.churn_logreg)
ORDER BY ABS(weight) DESC;
