"""Does a volatility-aware barrier (limit shrinks when the market is jumpy) help on the held-out recordings?
Compares, on recordings 2 and 3 together: the fixed barrier (alpha 0.2), the volatility-aware barrier with two
reference levels (median and 90th percentile of training volatility), bolted on at test time to the policies
trained with the fixed barrier, and policies trained with the volatility-aware barrier itself.
Usage: volguard.py [latency_ms] [recording ...]. Writes volguard.csv (volguard_lat<ms>.csv with latency) and the hourly profits."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sim import Market, Trader, touch, hard_cap, make_cbf, make_cbf_vol
from learn import Population

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RECS = sys.argv[2:] or [f"{HERE}/data/btcusd_l3_part2.csv", f"{HERE}/data/btcusd_l3_part3.csv"]
TAG = "" if len(sys.argv) <= 2 else "_" + "+".join(os.path.basename(r).replace(".csv", "") for r in RECS)
MED, P90 = 1.18, 2.36


class Batches:
    def __init__(self, pops): self.pops = pops
    def assign(self, base):
        for p in self.pops: p.assign(base)


def entries():
    out, pops = [("simple", Trader("", touch, guard=hard_cap), "simple", "-")], []
    def add(policy, weights, gname, guard):
        p = Population([weights], guard); pops.append(p); out.append((f"{policy} | {gname}", p.traders[0], policy, gname))
    trained = {"published": json.load(open(f"{HERE}/policy_cbf.json"))["weights"]}
    for f in sorted(glob.glob(f"{HERE}/seeds/policy_cbf_s*.json")): trained["fixed-trained s" + f[-6]] = json.load(open(f))["weights"]
    for name, w in trained.items():
        add(name, w, "fixed barrier", make_cbf(0.2))
        add(name, w, "vol barrier (median ref)", make_cbf_vol(0.2, ref_sd=MED))
        add(name, w, "vol barrier (p90 ref)", make_cbf_vol(0.2, ref_sd=P90))
    for f in sorted(glob.glob(f"{HERE}/seeds/policy_cbfvol_s*.json")):
        w = json.load(open(f))["weights"]
        add("vol-trained s" + f[-6], w, "vol barrier (p90 ref)", make_cbf_vol(0.2, ref_sd=P90))
        add("vol-trained s" + f[-6], w, "fixed barrier", make_cbf(0.2))
    return out, pops


reps, hours = [], []
for rec in RECS:
    ent, pops = entries()
    m = Market([e[1] for e in ent], batch=Batches(pops), latency_ms=LAT)
    r = pd.DataFrame(m.run(rec)); r["name"] = [e[0] for e in ent]; r["policy"] = [e[2] for e in ent]; r["guard"] = [e[3] for e in ent]
    reps.append(r); hours.append(np.diff(np.array([e[1].equity for e in ent])[:, ::60], axis=1))
    print(os.path.basename(rec), "done", flush=True)
H = np.concatenate(hours, axis=1); names = reps[0].name.tolist()
rep = pd.concat(reps)
s = rep.groupby("name", sort=False).agg(policy=("policy", "first"), guard=("guard", "first"), total=("total", "sum"), volume=("volume", "sum"),
                                       max_inv=("max_inv", "max"), mean_inv=("mean_abs_inv_units", "mean"), overrides=("overrides", "sum")).reindex(names)
s["per_hour"] = H.mean(1).round(3); s["hourly_sd"] = H.std(1, ddof=1).round(3)
s["worst_hour"] = H.min(1).round(2)
d = H - H[0]; s["vs_simple"] = d.mean(1).round(3); s["t_vs_simple"] = (d.mean(1) / (d.std(1, ddof=1) / np.sqrt(H.shape[1]) + 1e-12)).round(2)
sfx = (f"_lat{LAT}" if LAT else "") + TAG
s.to_csv(f"{HERE}/volguard{sfx}.csv"); np.savez(f"{HERE}/volguard{sfx}_hourly.npz", H=H, names=np.array(names))
pd.set_option("display.width", 250); print(s.drop(columns=["policy", "guard"]).to_string())
