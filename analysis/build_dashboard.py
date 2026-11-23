"""Build the self-contained interactive HTML dashboard.

Reads the result tables from data/results/ and the charts from charts/
and writes dashboard/index.html. The dashboard is a single file: charts
are embedded as base64 images and the tables are rendered from the real
result data, with tab navigation and a filterable retention table.

Usage:
    python3 analysis/build_dashboard.py
"""

import base64
import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "data", "results")
CHARTS = os.path.join(BASE, "charts")
OUT = os.path.join(BASE, "dashboard", "index.html")


def img64(name):
    path = os.path.join(CHARTS, name)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(
            fh.read()).decode()


def table_html(df, cols, rename=None, fmt=None):
    rename = rename or {}
    fmt = fmt or {}
    head = "".join(f"<th>{rename.get(c, c)}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    with open(os.path.join(RESULTS, "cohort_summary.json")) as fh:
        s = json.load(fh)
    metrics = pd.read_csv(os.path.join(RESULTS, "model_metrics.csv"))
    deciles = pd.read_csv(os.path.join(RESULTS, "decile_table.csv"))
    imp = pd.read_csv(os.path.join(RESULTS, "feature_importance.csv"))
    seg = pd.read_csv(os.path.join(RESULTS, "rfm_segment_summary.csv"))
    prio = pd.read_csv(os.path.join(RESULTS,
                                    "retention_priority_sample.csv"))
    bt = metrics[metrics["model"] == "boosted_tree"].iloc[0]
    lr = metrics[metrics["model"] == "logistic_regression"].iloc[0]

    metrics_tbl = table_html(
        metrics,
        ["model", "roc_auc", "pr_auc", "precision", "recall", "f1",
         "tp", "fp", "fn", "tn"],
        rename={"model": "Model", "roc_auc": "ROC AUC",
                "pr_auc": "PR AUC", "precision": "Precision",
                "recall": "Recall", "f1": "F1", "tp": "TP", "fp": "FP",
                "fn": "FN", "tn": "TN"},
        fmt={"roc_auc": lambda v: f"{v:.3f}",
             "pr_auc": lambda v: f"{v:.3f}",
             "precision": lambda v: f"{v:.3f}",
             "recall": lambda v: f"{v:.3f}",
             "f1": lambda v: f"{v:.3f}"})
    decile_tbl = table_html(
        deciles,
        ["decile", "customers", "avg_predicted_prob",
         "actual_churn_rate"],
        rename={"decile": "Risk decile", "customers": "Customers",
                "avg_predicted_prob": "Avg predicted prob",
                "actual_churn_rate": "Actual churn rate"},
        fmt={"avg_predicted_prob": pct, "actual_churn_rate": pct})
    imp_tbl = table_html(
        imp.head(10), ["feature", "importance"],
        rename={"feature": "Feature", "importance": "Importance"},
        fmt={"importance": lambda v: f"{v:.4f}"})
    seg_tbl = table_html(
        seg,
        ["segment", "customers", "customer_share_pct", "avg_clv",
         "total_clv", "avg_frequency", "avg_recency_days"],
        rename={"segment": "Segment", "customers": "Customers",
                "customer_share_pct": "Share (%)", "avg_clv": "Avg CLV",
                "total_clv": "Total CLV", "avg_frequency": "Avg frequency",
                "avg_recency_days": "Avg recency (days)"},
        fmt={"customer_share_pct": lambda v: f"{v:.1f}",
             "avg_clv": lambda v: f"${v:,.0f}",
             "total_clv": lambda v: f"${v:,.0f}",
             "avg_frequency": lambda v: f"{v:.2f}",
             "avg_recency_days": lambda v: f"{v:.0f}"})
    prio_tbl = table_html(
        prio.head(50),
        ["user_id", "churn_prob", "historical_clv",
         "expected_value_at_risk", "segment", "frequency",
         "recency_days"],
        rename={"user_id": "User", "churn_prob": "Churn prob",
                "historical_clv": "Historical CLV",
                "expected_value_at_risk": "Value at risk",
                "segment": "Segment", "frequency": "Orders",
                "recency_days": "Recency (days)"},
        fmt={"churn_prob": pct,
             "historical_clv": lambda v: f"${v:,.0f}",
             "expected_value_at_risk": lambda v: f"${v:,.0f}"})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TheLook Churn and CLV Dashboard</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
    sans-serif; margin: 0; background: #f4f6fa; color: #1c2733; }}
  header {{ background: #16324f; color: #fff; padding: 22px 28px; }}
  header h1 {{ margin: 0 0 4px; font-size: 22px; }}
  header p {{ margin: 0; opacity: .85; font-size: 13px; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 20px; }}
  .kpis {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }}
  .kpi {{ background: #fff; border-radius: 10px; padding: 14px 18px;
    flex: 1 1 150px; box-shadow: 0 1px 3px rgba(20,40,60,.12); }}
  .kpi .v {{ font-size: 24px; font-weight: 700; color: #2456a6; }}
  .kpi .l {{ font-size: 12px; color: #5a6b7c; margin-top: 2px; }}
  nav.tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }}
  nav.tabs button {{ border: 1px solid #c4d2e0; background: #fff;
    padding: 8px 14px; border-radius: 999px; cursor: pointer;
    font-size: 13px; }}
  nav.tabs button.active {{ background: #2456a6; color: #fff;
    border-color: #2456a6; }}
  section.panel {{ display: none; background: #fff; border-radius: 10px;
    padding: 18px 20px; box-shadow: 0 1px 3px rgba(20,40,60,.12); }}
  section.panel.active {{ display: block; }}
  section.panel h2 {{ margin-top: 0; font-size: 17px; }}
  section.panel p.note {{ color: #5a6b7c; font-size: 13px; }}
  img.chart {{ max-width: 100%; height: auto; }}
  .two {{ display: flex; gap: 18px; flex-wrap: wrap; }}
  .two > div {{ flex: 1 1 320px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #e2e8f0; padding: 6px 8px;
    text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: #eef3f9; }}
  input#filter {{ padding: 8px 10px; border: 1px solid #c4d2e0;
    border-radius: 8px; width: 260px; margin-bottom: 10px; }}
  footer {{ color: #5a6b7c; font-size: 12px; text-align: center;
    padding: 18px; }}
</style>
</head>
<body>
<header>
  <h1>TheLook Churn and CLV Dashboard</h1>
  <p>BigQuery ML customer retention analytics, local validation results.
     TheLook is a synthetic dataset generated by Looker.</p>
</header>
<main>
  <div class="kpis">
    <div class="kpi"><div class="v">{s['cohort_customers']:,}</div>
      <div class="l">Cohort customers</div></div>
    <div class="kpi"><div class="v">{s['churn_rate'] * 100:.1f}%</div>
      <div class="l">Observed churn rate</div></div>
    <div class="kpi"><div class="v">{bt['roc_auc']:.3f}</div>
      <div class="l">Boosted tree ROC AUC</div></div>
    <div class="kpi"><div class="v">{lr['roc_auc']:.3f}</div>
      <div class="l">Logistic regression ROC AUC</div></div>
    <div class="kpi"><div class="v">{s['current_active_customers']:,}</div>
      <div class="l">Currently active customers scored</div></div>
    <div class="kpi"><div class="v">{s['table_orders']:,}</div>
      <div class="l">Orders in dataset copy</div></div>
  </div>

  <nav class="tabs">
    <button data-tab="models" class="active">Models</button>
    <button data-tab="deciles">Decile analysis</button>
    <button data-tab="segments">CLV and RFM segments</button>
    <button data-tab="retention">Retention priorities</button>
  </nav>

  <section class="panel active" id="models">
    <h2>Model performance, held out test set</h2>
    <p class="note">Local equivalents of the BigQuery ML models:
      logistic regression and a boosted tree (HistGradientBoosting),
      evaluated on a random 20 percent holdout. Threshold 0.5.</p>
    {metrics_tbl}
    <div class="two">
      <div><img class="chart" src="{img64('roc_curves.png')}"
        alt="ROC curves"></div>
      <div><img class="chart" src="{img64('feature_importance.png')}"
        alt="Feature importance"></div>
    </div>
    <h3>Top features, boosted tree (permutation importance)</h3>
    {imp_tbl}
  </section>

  <section class="panel" id="deciles">
    <h2>Decile analysis</h2>
    <p class="note">Test customers ranked by predicted churn probability.
      Decile 1 is the highest risk group. A useful model shows the actual
      churn rate falling steadily toward decile 10.</p>
    <img class="chart" src="{img64('decile_churn.png')}"
      alt="Decile churn chart">
    {decile_tbl}
  </section>

  <section class="panel" id="segments">
    <h2>Historical CLV and RFM segments</h2>
    <p class="note">Segments use quintile scores for recency, frequency
      and monetary value over each customer full history.</p>
    <img class="chart" src="{img64('rfm_segments.png')}"
      alt="RFM segments">
    {seg_tbl}
    <img class="chart" src="{img64('clv_distribution.png')}"
      alt="CLV distribution">
  </section>

  <section class="panel" id="retention">
    <h2>Retention priority list, top 50</h2>
    <p class="note">Currently active customers ranked by expected value
      at risk: churn probability multiplied by historical CLV. Filter by
      segment name to narrow the list.</p>
    <input id="filter" type="text" placeholder="Filter, try Champions">
    <div id="prio">{prio_tbl}</div>
    <img class="chart" src="{img64('churn_by_traffic_source.png')}"
      alt="Churn by traffic source">
  </section>
</main>
<footer>
  Churn definition: no valid order in the 90 days after the cutoff
  ({s['cutoff_date']}), with features from the 6 months before it.
  Anchor date {s['anchor_date']}. Results computed locally with
  scikit-learn equivalents on a generated copy of the same dataset.
</footer>
<script>
  document.querySelectorAll("nav.tabs button").forEach(function (b) {{
    b.addEventListener("click", function () {{
      document.querySelectorAll("nav.tabs button").forEach(
        function (x) {{ x.classList.remove("active"); }});
      document.querySelectorAll("section.panel").forEach(
        function (x) {{ x.classList.remove("active"); }});
      b.classList.add("active");
      document.getElementById(b.dataset.tab).classList.add("active");
    }});
  }});
  var filter = document.getElementById("filter");
  filter.addEventListener("input", function () {{
    var q = filter.value.toLowerCase();
    document.querySelectorAll("#prio tbody tr").forEach(function (row) {{
      row.style.display = row.textContent.toLowerCase().includes(q)
        ? "" : "none";
    }});
  }});
</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", OUT, f"({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
