"""Stress tests on the two held-out recordings (2 and 3): do the conclusions survive latency, a pessimistic
queue model, fees and retraining with other random seeds?

Scenarios:  base       the published setting (no latency, queue-aware fills)
            lat250     quotes, re-prices and cancels land 250 ms after the decision
            lat1000    the same with 1 second
            through    pessimistic queue: we fill only when a trade prints through our price
Fees are applied afterwards (maker fee x traded notional), so every scenario also gets 1 and 2 bps columns.

Bots: simple, textbook (tuned on recording 1, as in compare.py), the published learned policies, and every
seed policy found in seeds/ (each run with the guard it was trained with; unguarded ones also with the
barrier guard bolted on at test time).
Usage: stress.py <scenario> [recording ...]       writes stress_<scenario>.csv and stress_<scenario>_hourly.npz
"""
import glob, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sim import Market, Trader, touch, make_as, hard_cap, make_cbf
from learn import Population, REC1

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = {"base": {}, "lat250": dict(latency_ms=250), "lat1000": dict(latency_ms=1000), "through": dict(through_only=True)}
scen = sys.argv[1]
recs = sys.argv[2:] or [f"{HERE}/data/btcusd_l3_part2.csv", f"{HERE}/data/btcusd_l3_part3.csv"]


class Batches:
    def __init__(self, pops): self.pops = pops
    def assign(self, base):
        for p in self.pops: p.assign(base)


def textbook_params():
    cache = f"{HERE}/textbook_params.json"
    if os.path.exists(cache): return json.load(open(cache))
    grid = [Trader(f"{g},{k},{a}", make_as(g, k, a), guard=hard_cap) for g, k, a in itertools.product((0.02, 0.1, 0.5, 2.0), (0.5, 1.5, 5.0), (0.0, 1.0, 3.0))]
    tune = pd.DataFrame(Market(grid).run(REC1)).sort_values("total", ascending=False)
    g, k, a = map(float, tune.name.iloc[0].split(",")); json.dump([g, k, a], open(cache, "w")); return [g, k, a]


def bots():
    g, k, a = textbook_params()
    out = [("simple", Trader("", touch, guard=hard_cap), None), ("textbook", Trader("", make_as(g, k, a), guard=hard_cap), None)]
    pops = []
    def add(label, weights, guard):
        p = Population([weights], guard); pops.append(p); out.append((label, p.traders[0], None))
    pub = {n: json.load(open(f"{HERE}/policy_{n}.json"))["weights"] for n in ("unguarded", "cbf")}
    add("published: no guard", pub["unguarded"], None)
    add("published: no-guard policy + guard", pub["unguarded"], make_cbf(0.2))
    add("published: guard in training", pub["cbf"], make_cbf(0.2))
    for f in sorted(glob.glob(f"{HERE}/seeds/policy_cbf_s*.json") + glob.glob(f"{HERE}/seeds/policy_none_s*.json")):
        tag = os.path.basename(f)[7:-5]; w = json.load(open(f))["weights"]
        if tag.startswith("cbf"): add(f"seed {tag[-1]}: guard in training", w, make_cbf(0.2))
        else:
            add(f"seed {tag[-1]}: no guard", w, None)
            add(f"seed {tag[-1]}: no-guard policy + guard", w, make_cbf(0.2))
    return out, pops


rows, hourly = [], []
for rec in recs:
    entries, pops = bots()
    m = Market([e[1] for e in entries], batch=Batches(pops), **SCEN[scen])
    rep = pd.DataFrame(m.run(rec)); rep["name"] = [e[0] for e in entries]; rep["recording"] = os.path.basename(rec)
    eq = np.array([e[1].equity for e in entries]); hourly.append(np.diff(eq[:, ::60], axis=1))
    rows.append(rep)
    print(f"{scen} {os.path.basename(rec)}: {rep.hours.iloc[0]} h", flush=True)

rep = pd.concat(rows)
H = np.concatenate(hourly, axis=1)                       # bots x hours, both recordings
names = rows[0].name.tolist()
summ = rep.groupby("name", sort=False).agg(fills=("fills", "sum"), volume=("volume", "sum"), total=("total", "sum"),
                                           max_inv=("max_inv", "max"), seconds_over=("breaches", "sum"),
                                           mean_inv=("mean_abs_inv_units", "mean")).reindex(names)
for bps in (1, 2): summ[f"total_{bps}bps"] = (summ.total - summ.volume * bps / 1e4).round(2)
summ["per_hour"] = H.mean(1).round(3); summ["se"] = (H.std(1, ddof=1) / np.sqrt(H.shape[1])).round(3)
for ref in ("simple", "textbook"):
    d = H - H[names.index(ref)]
    summ[f"vs_{ref}"] = d.mean(1).round(3); summ[f"t_vs_{ref}"] = (d.mean(1) / (d.std(1, ddof=1) / np.sqrt(d.shape[1]) + 1e-12)).round(2)
summ.insert(0, "hours", H.shape[1])
summ.to_csv(f"{HERE}/stress_{scen}.csv")
np.savez(f"{HERE}/stress_{scen}_hourly.npz", H=H, names=np.array(names))
pd.set_option("display.width", 250)
print(summ.to_string())
