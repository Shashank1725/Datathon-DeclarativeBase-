"""Trains both models and precomputes dashboard aggregates. Run once: python train.py"""
import json, joblib, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score

df = pd.read_csv("data/merged_transactions_master.csv")

# ---- Model 1: Isolation Forest (unsupervised leak detector)
IF_FEATS = ["quantity", "unit_price", "discount_pct", "underpricing_leak", "net_profit"]
iso = IsolationForest(contamination=0.05, n_estimators=200, random_state=42).fit(df[IF_FEATS])
df["anomaly_score"] = iso.decision_function(df[IF_FEATS])   # lower = more anomalous
df["is_anomaly"] = (iso.predict(df[IF_FEATS]) == -1).astype(int)

# ---- Model 2: LightGBM order-friction predictor (supervised)
LGB_FEATS = ["supplier_lead_days", "supplier_delay_days", "unit_cost", "discount_pct", "quantity"]
d2 = df[df.status.isin(["Completed", "Returned", "Cancelled"])].copy()
d2["is_friction_order"] = d2.status.isin(["Returned", "Cancelled"]).astype(int)
Xtr, Xte, ytr, yte = train_test_split(d2[LGB_FEATS], d2.is_friction_order, test_size=0.2,
                                      stratify=d2.is_friction_order, random_state=42)
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                         is_unbalance=True, random_state=42, verbose=-1).fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]
metrics = {"precision": round(precision_score(yte, p > .5), 3), "recall": round(recall_score(yte, p > .5), 3),
           "roc_auc": round(roc_auc_score(yte, p), 3), "train_rows": len(Xtr), "test_rows": len(Xte),
           "friction_rate": round(float(d2.is_friction_order.mean()), 3)}
df["friction_risk"] = clf.predict_proba(df[LGB_FEATS])[:, 1]
print("LightGBM:", metrics)

# ---- Catalog (one row per product)
cat = df.drop_duplicates("product_id")[["product_id", "category", "list_price", "unit_cost", "supplier_lead_days"]].copy()
cat["negative_margin"] = cat.list_price < cat.unit_cost
cat["suggested_price"] = (cat.unit_cost / 0.75).round(2)          # 25% target margin
cat["margin_pct"] = ((cat.list_price - cat.unit_cost) / cat.list_price * 100).round(1)
cat.to_csv("data/catalog.csv", index=False)

# ---- Dashboard aggregates
done = df[df.status == "Completed"]
fric = df[df.status.isin(["Returned", "Cancelled"])]
g = pd.DataFrame({
    "underpricing": df.groupby("category").underpricing_leak.sum(),
    "unprofitable": done[done.net_profit < 0].groupby("category").net_profit.sum().abs(),
    "friction": fric.groupby("category").net_revenue.sum()}).fillna(0).round(0)
neg = cat.groupby("category").negative_margin.sum()
g["neg_products"] = neg
kpis = {"underpricing": round(df.underpricing_leak.sum(), 2), "underpriced_orders": int((df.underpricing_leak > 0).sum()),
        "unprofitable": round(-done[done.net_profit < 0].net_profit.sum(), 2), "unprofitable_orders": int((done.net_profit < 0).sum()),
        "friction": round(fric.net_revenue.sum(), 2), "friction_orders": len(fric),
        "neg_products": int(cat.negative_margin.sum()), "products": len(cat), "rows": len(df),
        "ad_waste": round(df[(df.spend > 0) & (df.conversions == 0)].spend.sum(), 2)}

# Hybrid rule from the slides: anomalous AND net-negative
hy = df[(df.is_anomaly == 1) & (df.net_profit < 0)].sort_values("anomaly_score")
cols = ["transaction_id", "product_id", "category", "quantity", "unit_price", "list_price", "discount_pct",
        "net_profit", "underpricing_leak", "anomaly_score", "friction_risk", "status"]
def rootcause(r):
    if r.unit_cost > r.list_price: return "Negative-margin listing"
    if r.underpricing_leak > 0 and r.discount_pct >= .15: return "Underpricing + discount stack"
    if r.underpricing_leak > 0: return "Checkout underpricing"
    return "Discount stacking"
hy["root_cause"] = hy.apply(rootcause, axis=1)
anoms = hy[cols + ["root_cause"]].head(60).round(3)

# SLA claims: supplier delay beyond promised lead time on a failed order
sla = fric[fric.supplier_delay_days > fric.supplier_lead_days]
sla = sla.assign(days_late=sla.supplier_delay_days - sla.supplier_lead_days)
claims = sla[["transaction_id", "product_id", "category", "status", "supplier_lead_days", "supplier_delay_days",
              "days_late", "net_revenue"]].sort_values("net_revenue", ascending=False).round(2)
print("SLA claims:", len(claims), round(claims.net_revenue.sum(), 2))

json.dump({"kpis": kpis, "categories": g.reset_index().rename(columns={"index": "category"}).to_dict("records"),
           "metrics": metrics, "anomalies": anoms.to_dict("records"),
           "hybrid_flagged": int(len(hy)), "sla": claims.head(40).to_dict("records"),
           "sla_total": round(float(claims.net_revenue.sum()), 2), "sla_count": len(claims)},
          open("data/summary.json", "w"), indent=1, default=float)
joblib.dump({"iso": iso, "clf": clf, "IF_FEATS": IF_FEATS, "LGB_FEATS": LGB_FEATS,
             "score_p5": float(np.percentile(df.anomaly_score, 5))}, "data/models.joblib")
print(json.dumps(kpis, indent=1))
