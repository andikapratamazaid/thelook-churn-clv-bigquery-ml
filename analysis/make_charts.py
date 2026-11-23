"""Generate all README and dashboard charts from the result tables.

Reads data/results/*.csv and data/processed/*.csv produced by
run_analysis.py and writes PNG charts to charts/. Every number plotted
comes from the actual model run.

Usage:
    python3 analysis/make_charts.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "data", "results")
PROCESSED = os.path.join(BASE, "data", "processed")
CHARTS = os.path.join(BASE, "charts")

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
ACCENT = "#2456a6"
ACCENT2 = "#e07b39"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def roc_curves():
    pred = pd.read_csv(os.path.join(PROCESSED, "test_predictions.csv"))
    metrics = pd.read_csv(os.path.join(RESULTS, "model_metrics.csv"))
    auc = dict(zip(metrics["model"], metrics["roc_auc"]))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for col, label, color in (
            ("prob_logistic_regression", "Logistic regression", ACCENT2),
            ("prob_boosted_tree", "Boosted tree", ACCENT)):
        fpr, tpr, _ = roc_curve(pred["churn"], pred[col])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC = {auc[label_key(col)]:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--",
            label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, held out test set")
    ax.legend(loc="lower right")
    save(fig, "roc_curves.png")


def label_key(col):
    return col.replace("prob_", "")


def feature_importance():
    imp = pd.read_csv(os.path.join(RESULTS, "feature_importance.csv"))
    imp = imp.sort_values("importance").tail(12)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.barh(imp["feature"], imp["importance"], color=ACCENT,
            xerr=imp["importance_std"], capsize=3)
    ax.set_xlabel("Permutation importance (drop in ROC AUC)")
    ax.set_title("Boosted tree feature importance, test set")
    save(fig, "feature_importance.png")


def decile_chart():
    dec = pd.read_csv(os.path.join(RESULTS, "decile_table.csv"))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(dec["decile"], dec["actual_churn_rate"] * 100, color=ACCENT,
           label="Actual churn rate")
    ax.plot(dec["decile"], dec["avg_predicted_prob"] * 100, color=ACCENT2,
            marker="o", lw=2, label="Average predicted probability")
    ax.set_xticks(dec["decile"])
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Decile analysis: predicted risk vs actual churn")
    ax.legend()
    save(fig, "decile_churn.png")


def clv_distribution():
    rfm = pd.read_csv(os.path.join(PROCESSED, "customer_clv_rfm.csv"))
    vals = rfm["historical_clv"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bins = np.logspace(np.log10(max(vals.min(), 1)), np.log10(vals.max()),
                       40)
    ax.hist(vals, bins=bins, color=ACCENT)
    ax.set_xscale("log")
    ax.set_xlabel("Historical CLV in USD (log scale)")
    ax.set_ylabel("Customers")
    ax.set_title("Distribution of historical customer lifetime value")
    save(fig, "clv_distribution.png")


def rfm_segments():
    seg = pd.read_csv(os.path.join(RESULTS, "rfm_segment_summary.csv"))
    seg = seg.sort_values("customers")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
    axes[0].barh(seg["segment"], seg["customers"], color=ACCENT)
    axes[0].set_xlabel("Customers")
    axes[0].set_title("Customers per RFM segment")
    axes[1].barh(seg["segment"], seg["avg_clv"], color=ACCENT2)
    axes[1].set_xlabel("Average historical CLV (USD)")
    axes[1].set_title("Average CLV per RFM segment")
    save(fig, "rfm_segments.png")


def churn_by_source():
    src = pd.read_csv(os.path.join(RESULTS, "churn_by_traffic_source.csv"))
    src = src.sort_values("churn_rate")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.barh(src["traffic_source"], src["churn_rate"] * 100, color=ACCENT)
    ax.set_xlabel("Churn rate in cohort (%)")
    ax.set_title("Observed churn rate by user traffic source")
    save(fig, "churn_by_traffic_source.png")


def main():
    os.makedirs(CHARTS, exist_ok=True)
    roc_curves()
    feature_importance()
    decile_chart()
    clv_distribution()
    rfm_segments()
    churn_by_source()


if __name__ == "__main__":
    main()
