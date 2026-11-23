"""Local validation pipeline for TheLook Churn and CLV.

This script mirrors, definition for definition, the BigQuery ML pipeline in
the sql/ folder, and trains the local equivalents of the two BigQuery ML
model types:

  * BigQuery ML LOGISTIC_REG            -> sklearn LogisticRegression
  * BigQuery ML BOOSTED_TREE_CLASSIFIER -> sklearn HistGradientBoostingClassifier

It runs on the local copy of the TheLook tables in data/raw_mirror/ (see
generate_thelook_copy.py for how that copy was produced and why it is a
faithful copy of the same dataset).

Churn definition (identical to sql/01_feature_engineering.sql):
  anchor date   = max order date in the data
  cutoff        = anchor date minus 90 days
  observation   = the 6 months before the cutoff (features come from here)
  label window  = the 90 days from the cutoff
  cohort        = customers with at least one valid order in the observation
                  window (valid = order status is not Cancelled or Returned)
  churn = 1     = no valid order in the label window

Outputs go to data/results/ (deliverable CSVs) and data/processed/
(intermediate feature and prediction tables used by the charts, the
dashboard and the tests).

Usage:
    python3 analysis/run_analysis.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw_mirror")
RESULTS = os.path.join(BASE, "data", "results")
PROCESSED = os.path.join(BASE, "data", "processed")
SEED = 42

INVALID = ("Cancelled", "Returned")
NUMERIC = [
    "recency_days", "frequency", "monetary", "total_items",
    "avg_order_value", "avg_basket_items", "return_rate", "sessions",
    "total_events", "product_views", "cart_events", "purchase_events",
    "cart_to_purchase_ratio", "tenure_days", "age",
]
CATEGORICAL = ["age_band", "gender", "traffic_source", "country",
               "favorite_category"]
FEATURES = NUMERIC + CATEGORICAL


def load_tables():
    users = pd.read_csv(os.path.join(RAW, "users.csv"))
    orders = pd.read_csv(
        os.path.join(RAW, "orders.csv"),
        usecols=["order_id", "user_id", "status", "created_at",
                 "num_of_item"],
    )
    items = pd.read_csv(
        os.path.join(RAW, "order_items.csv"),
        usecols=["id", "order_id", "user_id", "product_id", "status",
                 "created_at", "sale_price"],
    )
    events = pd.read_csv(
        os.path.join(RAW, "events.csv"),
        usecols=["id", "user_id", "session_id", "created_at", "event_type"],
    )
    products = pd.read_csv(
        os.path.join(RAW, "products.csv"), usecols=["id", "category"]
    )
    for df, col in ((users, "created_at"), (orders, "created_at"),
                    (items, "created_at"), (events, "created_at")):
        df[col] = pd.to_datetime(df[col], format="mixed", utc=True)
        df[col] = df[col].dt.tz_localize(None)
    return users, orders, items, events, products


def order_value_table(items, order_ids):
    """Per order aggregates, mirroring the order_values CTE in the SQL.

    Cancelled items are ignored. Returned items add to returned_items but
    contribute no revenue and no kept item count.
    """
    it = items[items["order_id"].isin(order_ids)]
    it = it[it["status"] != "Cancelled"].copy()
    it["kept_value"] = np.where(it["status"] != "Returned",
                                it["sale_price"], 0.0)
    it["kept"] = (it["status"] != "Returned").astype(int)
    it["returned"] = (it["status"] == "Returned").astype(int)
    ov = it.groupby("order_id").agg(
        order_value=("kept_value", "sum"),
        item_count=("kept", "sum"),
        returned_items=("returned", "sum"),
    )
    return ov


def build_features(users, orders, items, events, products, obs_start,
                   cutoff, include_label, label_end=None):
    """Feature table for one observation window. Mirrors the SQL views in
    01_feature_engineering.sql (include_label=True) and
    06_prediction_and_deciles.sql (include_label=False, window inclusive of
    the cutoff day)."""
    if include_label:
        in_window = (orders["created_at"].dt.date >= obs_start) & \
                    (orders["created_at"].dt.date < cutoff)
    else:
        in_window = (orders["created_at"].dt.date >= obs_start) & \
                    (orders["created_at"].dt.date <= cutoff)
    obs_orders = orders[in_window & ~orders["status"].isin(INVALID)].copy()
    if obs_orders.empty:
        return pd.DataFrame()

    ov = order_value_table(items, set(obs_orders["order_id"]))
    oo = obs_orders.set_index("order_id").join(ov).fillna(
        {"order_value": 0.0, "item_count": 0, "returned_items": 0})
    grp = oo.groupby("user_id")
    feats = pd.DataFrame({
        "recency_days": (pd.Timestamp(cutoff) - grp["created_at"].max()
                         ).dt.days,
        "frequency": grp.size(),
        "monetary": grp["order_value"].sum(),
        "total_items": grp["item_count"].sum(),
        "returned_items": grp["returned_items"].sum(),
    })
    feats["avg_order_value"] = feats["monetary"] / feats["frequency"]
    feats["avg_basket_items"] = feats["total_items"] / feats["frequency"]
    feats["return_rate"] = np.where(
        feats["total_items"] > 0,
        feats["returned_items"] / feats["total_items"], 0.0)
    feats = feats.drop(columns=["returned_items"])

    # Event features. Anonymous ghost events (user_id null) are ignored.
    ev = events[events["user_id"].notna()].copy()
    ev["user_id"] = ev["user_id"].astype(int)
    if include_label:
        ev_in = (ev["created_at"].dt.date >= obs_start) & \
                (ev["created_at"].dt.date < cutoff)
    else:
        ev_in = (ev["created_at"].dt.date >= obs_start) & \
                (ev["created_at"].dt.date <= cutoff)
    ev = ev[ev_in]
    if not ev.empty:
        ev["is_product"] = (ev["event_type"] == "product").astype(int)
        ev["is_cart"] = (ev["event_type"] == "cart").astype(int)
        ev["is_purchase"] = (ev["event_type"] == "purchase").astype(int)
        ef = ev.groupby("user_id").agg(
            sessions=("session_id", "nunique"),
            total_events=("id", "count"),
            product_views=("is_product", "sum"),
            cart_events=("is_cart", "sum"),
            purchase_events=("is_purchase", "sum"),
        )
        ef["cart_to_purchase_ratio"] = ef["cart_events"] / \
            (ef["purchase_events"] + 1)
        feats = feats.join(ef)
    for col in ("sessions", "total_events", "product_views", "cart_events",
                "purchase_events", "cart_to_purchase_ratio"):
        if col not in feats.columns:
            feats[col] = 0
        feats[col] = feats[col].fillna(0)

    # Favorite category by kept item count in the window.
    kept = items[items["order_id"].isin(set(obs_orders["order_id"])) &
                 ~items["status"].isin(INVALID)]
    if not kept.empty:
        kc = kept.merge(products, left_on="product_id", right_on="id",
                        how="left")
        fav = (kc.groupby(["user_id", "category"]).size()
               .reset_index(name="n")
               .sort_values(["user_id", "n"], ascending=[True, False])
               .drop_duplicates("user_id")
               .set_index("user_id")["category"])
        feats["favorite_category"] = fav

    u = users.set_index("id")
    feats["tenure_days"] = [
        (pd.Timestamp(cutoff) - pd.Timestamp(d)).days
        for d in u.loc[feats.index, "created_at"].dt.date
    ]
    feats["age"] = u.loc[feats.index, "age"].values
    feats["gender"] = u.loc[feats.index, "gender"].values
    feats["traffic_source"] = u.loc[feats.index, "traffic_source"].values
    feats["country"] = u.loc[feats.index, "country"].values
    feats["age_band"] = pd.cut(
        feats["age"], bins=[-np.inf, 24, 34, 44, 54, np.inf],
        labels=["under_25", "25_34", "35_44", "45_54", "55_plus"]).astype(str)
    feats["favorite_category"] = feats["favorite_category"].fillna("None")
    feats = feats.reset_index().rename(columns={"user_id": "user_id"})

    if include_label:
        lab_orders = orders[
            ~orders["status"].isin(INVALID)
            & (orders["created_at"].dt.date >= cutoff)
            & (orders["created_at"].dt.date < label_end)]
        active = set(lab_orders["user_id"])
        feats["churn"] = np.where(feats["user_id"].isin(active), 0, 1)
    return feats


def ntile(values, n, ascending=True):
    """Positional n-tile assignment mirroring BigQuery NTILE semantics:
    rows are ordered by value and split into n groups as evenly as
    possible, with earlier groups receiving the remainder rows. Ties are
    split positionally, which matches NTILE behaviour on tied values."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    if not ascending:
        order = order[::-1]
    base, rem = divmod(len(values), n)
    tile_by_pos = np.empty(len(values), dtype=int)
    start = 0
    for i in range(n):
        size = base + 1 if i < rem else base
        tile_by_pos[start:start + size] = i + 1
        start += size
    out = np.empty(len(values), dtype=int)
    out[order] = tile_by_pos
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PROCESSED, exist_ok=True)
    users, orders, items, events, products = load_tables()

    anchor = orders["created_at"].max().date()
    cutoff = anchor - pd.Timedelta(days=90)
    obs_start = (pd.Timestamp(cutoff) - pd.DateOffset(months=6)).date()
    label_end = cutoff + pd.Timedelta(days=90)

    table_counts = {
        "users": int(len(users)), "orders": int(len(orders)),
        "order_items": int(len(items)), "events": int(len(events)),
        "products": int(len(products)),
    }
    max_order_date = str(anchor)

    # ------------------------------------------------------------------
    # Cohort features and labels
    # ------------------------------------------------------------------
    cohort = build_features(users, orders, items, events, products,
                            obs_start, cutoff, include_label=True,
                            label_end=label_end)
    cohort.to_csv(os.path.join(PROCESSED, "cohort_features.csv"),
                  index=False)

    # ------------------------------------------------------------------
    # Train and evaluate both models
    # ------------------------------------------------------------------
    train, test = train_test_split(
        cohort, test_size=0.2, random_state=SEED, stratify=cohort["churn"])
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    pre_tree = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    models = {
        "logistic_regression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, C=1.0))]),
        "boosted_tree": Pipeline([
            ("pre", pre_tree),
            ("clf", HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.1, max_depth=6,
                min_samples_leaf=20, l2_regularization=1.0,
                random_state=SEED))]),
    }
    metrics_rows = []
    test_pred = test[["user_id", "churn"]].copy()
    for name, model in models.items():
        model.fit(train[FEATURES], train["churn"])
        prob = model.predict_proba(test[FEATURES])[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(test["churn"], pred).ravel()
        metrics_rows.append({
            "model": name,
            "roc_auc": roc_auc_score(test["churn"], prob),
            "pr_auc": average_precision_score(test["churn"], prob),
            "precision": precision_score(test["churn"], pred,
                                         zero_division=0),
            "recall": recall_score(test["churn"], pred, zero_division=0),
            "f1": f1_score(test["churn"], pred, zero_division=0),
            "threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "test_customers": int(len(test)),
        })
        test_pred[f"prob_{name}"] = prob
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(os.path.join(RESULTS, "model_metrics.csv"), index=False)
    test_pred.to_csv(os.path.join(PROCESSED, "test_predictions.csv"),
                     index=False)

    # ------------------------------------------------------------------
    # Decile analysis on the boosted tree test predictions
    # ------------------------------------------------------------------
    tp_df = test_pred.copy()
    tp_df["decile"] = ntile(tp_df["prob_boosted_tree"].values, 10,
                             ascending=False)
    deciles = tp_df.groupby("decile").agg(
        customers=("user_id", "count"),
        avg_predicted_prob=("prob_boosted_tree", "mean"),
        actual_churn_rate=("churn", "mean"),
        min_prob=("prob_boosted_tree", "min"),
        max_prob=("prob_boosted_tree", "max"),
    ).reset_index()
    deciles.to_csv(os.path.join(RESULTS, "decile_table.csv"), index=False)

    # ------------------------------------------------------------------
    # Permutation importance for the boosted tree (grouped to base
    # features). BigQuery ML.WEIGHTS reports gain based importance; the
    # local equivalent reported here is permutation importance on the
    # held out test set, measured as the drop in ROC AUC.
    # ------------------------------------------------------------------
    tree = models["boosted_tree"]
    perm = permutation_importance(
        tree, test[FEATURES], test["churn"], n_repeats=5,
        random_state=SEED, scoring="roc_auc")
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(RESULTS, "feature_importance.csv"),
               index=False)

    # ------------------------------------------------------------------
    # Score currently active customers (cutoff at the anchor date)
    # ------------------------------------------------------------------
    cur_start = (pd.Timestamp(anchor) - pd.DateOffset(months=6)).date()
    current = build_features(users, orders, items, events, products,
                             cur_start, anchor, include_label=False)
    current["churn_prob"] = models["boosted_tree"].predict_proba(
        current[FEATURES])[:, 1]
    current.to_csv(os.path.join(PROCESSED, "current_features.csv"),
                   index=False)

    # ------------------------------------------------------------------
    # CLV and RFM segmentation over the full history (mirrors SQL 07)
    # ------------------------------------------------------------------
    valid_orders = orders[~orders["status"].isin(INVALID)]
    ov_all = order_value_table(items, set(valid_orders["order_id"]))
    vo = valid_orders.set_index("order_id").join(ov_all).fillna(
        {"order_value": 0.0, "item_count": 0, "returned_items": 0})
    rfm = vo.groupby("user_id").agg(
        last_order=("created_at", "max"),
        frequency=("order_value", "size"),
        historical_clv=("order_value", "sum"),
    ).reset_index()
    rfm["recency_days"] = (pd.Timestamp(anchor)
                            - rfm["last_order"].dt.date.apply(
                                lambda d: pd.Timestamp(d))).dt.days
    rfm["r_score"] = 6 - ntile(rfm["recency_days"].values, 5,
                                ascending=True)
    rfm["f_score"] = ntile(rfm["frequency"].values, 5, ascending=True)
    rfm["m_score"] = ntile(rfm["historical_clv"].values, 5,
                           ascending=True)

    def segment(row):
        r, f = row["r_score"], row["f_score"]
        if r >= 4 and f >= 4:
            return "Champions"
        if r >= 3 and f >= 3:
            return "Loyal"
        if r >= 4 and f in (2, 3):
            return "Potential Loyalist"
        if r == 5 and f == 1:
            return "New"
        if r <= 2 and f >= 3:
            return "At Risk"
        if r <= 2 and f == 2:
            return "Hibernating"
        if r <= 2 and f == 1:
            return "Lost"
        return "Others"

    rfm["segment"] = rfm.apply(segment, axis=1)
    rfm.to_csv(os.path.join(PROCESSED, "customer_clv_rfm.csv"),
               index=False)

    total_customers = len(rfm)
    seg = rfm.groupby("segment").agg(
        customers=("user_id", "count"),
        avg_clv=("historical_clv", "mean"),
        total_clv=("historical_clv", "sum"),
        avg_frequency=("frequency", "mean"),
        avg_recency_days=("recency_days", "mean"),
    ).reset_index()
    seg["customer_share_pct"] = seg["customers"] / total_customers * 100
    seg = seg.sort_values("total_clv", ascending=False)
    seg.to_csv(os.path.join(RESULTS, "rfm_segment_summary.csv"),
               index=False)

    qs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    clv_stats = pd.DataFrame({
        "statistic": ["count", "mean", "std"] +
                     [f"p{int(q * 100)}" for q in qs],
        "historical_clv": [len(rfm), rfm["historical_clv"].mean(),
                           rfm["historical_clv"].std()] +
                          list(rfm["historical_clv"].quantile(qs)),
    })
    clv_stats.to_csv(os.path.join(RESULTS, "clv_distribution.csv"),
                     index=False)

    # ------------------------------------------------------------------
    # Retention priority list: current churn probability times CLV
    # ------------------------------------------------------------------
    prio = current[["user_id", "churn_prob"]].merge(
        rfm[["user_id", "historical_clv", "segment", "frequency",
             "recency_days"]], on="user_id", how="inner")
    prio["expected_value_at_risk"] = prio["churn_prob"] * \
        prio["historical_clv"]
    prio = prio.sort_values("expected_value_at_risk",
                            ascending=False).head(100)
    prio.to_csv(os.path.join(RESULTS, "retention_priority_sample.csv"),
                index=False)

    # Churn rate by traffic source (extra context table for charts)
    by_src = cohort.groupby("traffic_source").agg(
        customers=("user_id", "count"), churn_rate=("churn", "mean")
    ).reset_index().sort_values("churn_rate", ascending=False)
    by_src.to_csv(os.path.join(RESULTS, "churn_by_traffic_source.csv"),
                  index=False)

    # ------------------------------------------------------------------
    # Cohort summary
    # ------------------------------------------------------------------
    summary = {
        "anchor_date": max_order_date,
        "cutoff_date": str(cutoff),
        "observation_start": str(obs_start),
        "label_window_end": str(label_end),
        "cohort_customers": int(len(cohort)),
        "churned_customers": int(cohort["churn"].sum()),
        "retained_customers": int((cohort["churn"] == 0).sum()),
        "churn_rate": float(cohort["churn"].mean()),
        "train_customers": int(len(train)),
        "test_customers": int(len(test)),
        "current_active_customers": int(len(current)),
        "customers_with_orders_all_time": int(total_customers),
        "table_users": table_counts["users"],
        "table_orders": table_counts["orders"],
        "table_order_items": table_counts["order_items"],
        "table_events": table_counts["events"],
        "table_products": table_counts["products"],
        "min_order_date": str(orders["created_at"].min().date()),
        "max_order_date": max_order_date,
    }
    with open(os.path.join(RESULTS, "cohort_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULTS, "cohort_summary.csv"), index=False)
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
