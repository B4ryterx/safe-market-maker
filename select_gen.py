"""Which training generation should we keep? Four rules fixed before looking at the test data, applied to the five
guard-trained seeds (seeds_hist/history_cbf_s*.json), each run with the fixed barrier on recordings 2 and 3:
  A  best on the 4-hour validation window (the published rule)
  B  the last generation
  C  the average of the weights of the last ten generations
  D  best on validation, but only from generation 5 onward
Usage: select_gen.py [recording ...]. Writes select_gen[_<recordings>].csv."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sim import Market, Trader, touch, hard_cap, make_cbf
from learn import Population

HERE = os.path.dirname(os.path.abspath(__file__))
RECS = sys.argv[1:] or [f"{HERE}/data/btcusd_l3_part2.csv", f"{HERE}/data/btcusd_l3_part3.csv"]
TAG = "" if len(sys.argv) <= 1 else "_" + "+".join(os.path.basename(r).replace(".csv", "") for r in RECS)
RULES = {"A best validation": lambda h: max(h, key=lambda g: g["val"])["weights"],
         "B last generation": lambda h: h[-1]["weights"],
         "C mean of last 10": lambda h: np.mean([g["weights"] for g in h[-10:]], axis=0).tolist(),
         "D best validation, gen >= 5": lambda h: max(h[5:], key=lambda g: g["val"])["weights"]}


class Batches:
    def __init__(self, pops): self.pops = pops
    def assign(self, base):
        for p in self.pops: p.assign(base)


def entries():
    out, pops = [("simple", "-", Trader("", touch, guard=hard_cap))], []
    for f in sorted(glob.glob(f"{HERE}/seeds_hist/history_cbf_s*.json")):
        h = json.load(open(f)); seed = f[-6]
        for rule, pick in RULES.items():
            p = Population([pick(h)], make_cbf(0.2)); pops.append(p); out.append((rule, seed, p.traders[0]))
    return out, pops


reps, hours = [], []
for rec in RECS:
    ent, pops = entries()
    r = pd.DataFrame(Market([e[2] for e in ent], batch=Batches(pops)).run(rec))
    r["rule"] = [e[0] for e in ent]; r["seed"] = [e[1] for e in ent]
    reps.append(r); hours.append(np.diff(np.array([e[2].equity for e in ent])[:, ::60], axis=1))
    print(os.path.basename(rec), "done", flush=True)
H = np.concatenate(hours, axis=1); rep = pd.concat(reps)
first = reps[0]
s = rep.groupby(["rule", "seed"], sort=False).agg(total=("total", "sum"), mean_inv=("mean_abs_inv_units", "mean"), max_inv=("max_inv", "max")).reset_index()
s["hourly_sd"] = H.std(1, ddof=1).round(3)
d = H - H[0]; s["vs_simple_t"] = (d.mean(1) / (d.std(1, ddof=1) / np.sqrt(H.shape[1]) + 1e-12)).round(2)
s.to_csv(f"{HERE}/select_gen{TAG}.csv", index=False)
pd.set_option("display.width", 200); print(s.to_string(index=False))
print("\nby rule (mean and range of total profit over the five seeds):")
print(s[s.rule != "simple"].groupby("rule").total.agg(["mean", "min", "max"]).round(2).to_string())
