"""Validation tests for the analysis result tables.

Run from the project root after analysis/run_analysis.py:
    python3 -m pytest tests/ -q

The tests check internal consistency of the published results: cohort
arithmetic, metric ranges, decile sanity, and segment shares.
"""

import json
import os

import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "data", "results")
PROCESSED = os.path.join(BASE, "data", "processed")


@pytest.fixture(scope="module")
def summary():
    with open(os.path.join(RESULTS, "cohort_summary.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(os.path.join(RESULTS, "model_metrics.csv"))


@pytest.fixture(scope="module")
def deciles():
    return pd.read_csv(os.path.join(RESULTS, "decile_table.csv"))


@pytest.fixture(scope="module")
def segments():
    return pd.read_csv(os.path.join(RESULTS, "rfm_segment_summary.csv"))


def test_cohort_counts_consistent(summary):
    assert summary["cohort_customers"] > 0
    assert (summary["churned_customers"] + summary["retained_customers"]
            == summary["cohort_customers"])
    assert (summary["train_customers"] + summary["test_customers"]
            == summary["cohort_customers"])


def test_churn_rate_valid(summary):
    assert 0.0 < summary["churn_rate"] < 1.0
    expected = summary["churned_customers"] / summary["cohort_customers"]
    assert summary["churn_rate"] == pytest.approx(expected, abs=1e-9)


def test_table_counts_match_raw(summary):
    raw = os.path.join(BASE, "data", "raw_mirror")
    for table, key in (("users", "table_users"), ("orders", "table_orders"),
                       ("order_items", "table_order_items"),
                       ("events", "table_events"),
                       ("products", "table_products")):
        with open(os.path.join(raw, f"{table}.csv")) as fh:
            rows = sum(1 for _ in fh) - 1
        assert rows == summary[key], table


def test_both_models_present(metrics):
    assert set(metrics["model"]) == {"logistic_regression", "boosted_tree"}


def test_metric_ranges(metrics):
    for _, row in metrics.iterrows():
        assert 0.5 <= row["roc_auc"] <= 1.0
        assert 0.0 < row["pr_auc"] <= 1.0
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0
        assert (row["tn"] + row["fp"] + row["fn"] + row["tp"]
                == row["test_customers"])


def test_decile_table_shape(deciles, summary):
    assert len(deciles) == 10
    assert deciles["customers"].sum() == summary["test_customers"]
    assert deciles["decile"].tolist() == list(range(1, 11))


def test_decile_monotonic_sanity(deciles):
    rates = deciles.sort_values("decile")["actual_churn_rate"].values
    # The highest risk decile must churn more than the lowest risk one,
    # and the broad trend must be downward (allow small local noise).
    assert rates[0] > rates[-1]
    assert rates[:3].mean() > rates[-3:].mean()
    weighted = (deciles["actual_churn_rate"]
                * deciles["customers"]).sum() / deciles["customers"].sum()
    assert 0.0 < weighted < 1.0


def test_decile_probabilities_ordered(deciles):
    avg = deciles.sort_values("decile")["avg_predicted_prob"].values
    assert all(avg[i] >= avg[i + 1] for i in range(len(avg) - 1))


def test_segment_shares_sum_to_100(segments, summary):
    assert segments["customer_share_pct"].sum() == pytest.approx(
        100.0, abs=0.5)
    assert segments["customers"].sum() == \
        summary["customers_with_orders_all_time"]


def test_feature_importance_valid():
    imp = pd.read_csv(os.path.join(RESULTS, "feature_importance.csv"))
    assert len(imp) >= 10
    assert imp["importance"].notna().all()
    assert imp["importance"].max() > 0


def test_retention_priority_sample():
    prio = pd.read_csv(os.path.join(RESULTS,
                                    "retention_priority_sample.csv"))
    assert 0 < len(prio) <= 100
    expected = prio["churn_prob"] * prio["historical_clv"]
    assert prio["expected_value_at_risk"].values == pytest.approx(
        expected.values, rel=1e-6)
    vals = prio["expected_value_at_risk"].values
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    assert prio["churn_prob"].between(0, 1).all()


def test_cohort_features_sane():
    cohort = pd.read_csv(os.path.join(PROCESSED, "cohort_features.csv"))
    assert (cohort["frequency"] >= 1).all()
    assert (cohort["monetary"] >= 0).all()
    assert (cohort["recency_days"] >= 0).all()
    assert cohort["churn"].isin([0, 1]).all()
    assert cohort["return_rate"].between(0, 1).all()


def test_clv_distribution_stats():
    clv = pd.read_csv(os.path.join(RESULTS, "clv_distribution.csv"))
    vals = dict(zip(clv["statistic"], clv["historical_clv"]))
    assert vals["count"] > 0
    assert vals["p0"] >= 0
    assert vals["p50"] <= vals["p90"] <= vals["p100"]
    assert vals["mean"] >= vals["p50"] or vals["mean"] > 0
