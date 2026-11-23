-- ============================================================================
-- 04_evaluation.sql
-- Evaluate both churn models on their held out evaluation split.
--
-- ML.EVALUATE with no input table scores the evaluation split created at
-- training time. For binary classifiers it returns precision, recall,
-- accuracy, F1 score, log loss and ROC AUC at the default 0.5 threshold.
-- ML.CONFUSION_MATRIX shows the raw counts behind those numbers.
-- ============================================================================

-- Headline metrics, one row per model.
SELECT 'logistic_regression' AS model, *
FROM ML.EVALUATE(MODEL thelook_ml.churn_logreg);

SELECT 'boosted_tree' AS model, *
FROM ML.EVALUATE(MODEL thelook_ml.churn_boosted_tree);

-- Confusion matrix at the default 0.5 threshold, boosted tree.
SELECT *
FROM ML.CONFUSION_MATRIX(MODEL thelook_ml.churn_boosted_tree);

-- Confusion matrix at the default 0.5 threshold, logistic regression.
SELECT *
FROM ML.CONFUSION_MATRIX(MODEL thelook_ml.churn_logreg);

-- ROC curve points for the boosted tree, useful for plotting the curve
-- outside BigQuery and for inspecting precision and recall trade offs at
-- thresholds other than 0.5.
SELECT *
FROM ML.ROC_CURVE(MODEL thelook_ml.churn_boosted_tree)
ORDER BY threshold DESC;
