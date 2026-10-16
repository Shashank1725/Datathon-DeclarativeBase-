import json, joblib, pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Revenue Sentinel")
S = json.load(open("data/summary.json"))
M = joblib.load("data/models.joblib")
CAT = pd.read_csv("data/catalog.csv").set_index("product_id")
MIN_MARGIN = 0.10          # floor on net margin after discount

class Order(BaseModel):
    product_id: str
    quantity: int = 1
    unit_price: float
    discount_pct: float = 0.0

@app.get("/api/summary")
def summary():
    return {k: S[k] for k in ("kpis", "categories", "metrics", "hybrid_flagged", "sla_total", "sla_count")}

@app.get("/api/anomalies")
def anomalies(): return S["anomalies"]

@app.get("/api/sla")
def sla(): return S["sla"]

@app.get("/api/catalog")
def catalog(only_bad: bool = True):
    c = CAT[CAT.negative_margin] if only_bad else CAT
    return c.reset_index().to_dict("records")

@app.get("/api/products")
def products(): return CAT.reset_index()[["product_id", "category", "list_price", "unit_cost"]].to_dict("records")

@app.post("/api/checkout")
def checkout(o: Order):
    """Margin-Guard: deterministic rules + Isolation Forest, applied before the order is placed."""
    if o.product_id not in CAT.index: return {"error": "Unknown product_id"}
    p = CAT.loc[o.product_id]
    lp, uc = float(p.list_price), float(p.unit_cost)
    leak = max(lp - o.unit_price, 0) * o.quantity
    net = o.quantity * (o.unit_price * (1 - o.discount_pct) - uc)
    X = pd.DataFrame([[o.quantity, o.unit_price, o.discount_pct, leak, net]], columns=M["IF_FEATS"])
    score = float(M["iso"].decision_function(X)[0]); anomalous = int(M["iso"].predict(X)[0]) == -1
    risk = float(M["clf"].predict_proba(pd.DataFrame([[p.supplier_lead_days, None, uc, o.discount_pct, o.quantity]],
                 columns=M["LGB_FEATS"]).astype(float))[0, 1])
    actions, price, disc = [], o.unit_price, o.discount_pct
    if lp < uc: actions.append(f"Catalog listing is below cost (list ${lp:.2f} < cost ${uc:.2f}); sale held until the price is corrected.")
    if price < lp:
        actions.append(f"Underpricing glitch: charged ${price:.2f} vs list ${lp:.2f}. Reset to list price."); price = lp
    floor = uc * (1 + MIN_MARGIN) / max(price, 1e-9)   # price*(1-d) >= cost*(1+margin)
    max_disc = max(0.0, round(1 - floor, 2))
    if disc > max_disc:
        actions.append(f"Discount {disc:.0%} would breach the {MIN_MARGIN:.0%} margin floor. Capped to {max_disc:.0%}."); disc = max_disc
    new_net = o.quantity * (price * (1 - disc) - uc)
    if lp < uc and new_net < 0: decision = "BLOCK"
    elif actions: decision = "ADJUST"
    else: decision = "ALLOW"
    if decision == "ADJUST" and new_net < 0: decision = "BLOCK"; actions.append("Even at list price with no discount the order loses money.")
    return {"decision": decision, "actions": actions, "list_price": lp, "unit_cost": uc,
            "original": {"price": o.unit_price, "discount": o.discount_pct, "net_profit": round(net, 2), "leak": round(leak, 2)},
            "enforced": {"price": round(price, 2), "discount": disc, "net_profit": round(new_net, 2)},
            "anomaly_score": round(score, 3), "is_anomaly": anomalous, "friction_risk": round(risk, 3),
            "recovered": round(new_net - net, 2)}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
