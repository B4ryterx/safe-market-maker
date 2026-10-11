"""Final comparison on recording 2, which no bot has seen: simple bot, tuned textbook bot, and the learning
bot with no guard, a hard cap, and the barrier-function guard. Zero maker fee unless stated."""
import itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sim import Market, Trader, touch, make_as, hard_cap, make_cbf
from learn import Population, REC1
pd.set_option("display.width", 250)
TEST = sys.argv[1] if len(sys.argv) > 1 else "data/btcusd_l3_part2.csv"


class Batches:
    def __init__(self, pops): self.pops = pops
    def assign(self, base):
        for p in self.pops: p.assign(base)


# textbook bot: tune on all of recording 1
grid = [Trader(f"{g},{k},{a}", make_as(g, k, a), guard=hard_cap) for g, k, a in itertools.product((0.02, 0.1, 0.5, 2.0), (0.5, 1.5, 5.0), (0.0, 1.0, 3.0))]
tune = pd.DataFrame(Market(grid).run(REC1)).sort_values("total", ascending=False)
g, k, a = map(float, tune.name.iloc[0].split(","))
print(f"textbook bot tuned on recording 1: gamma {g}, k {k}, imbalance weight {a} (profit there {tune.total.iloc[0]:+.2f}; {int((tune.total > 0).sum())} of {len(tune)} settings profitable)")

pol = {n: json.load(open(f"policy_{n}.json")) for n in ("unguarded", "capped", "cbf") if os.path.exists(f"policy_{n}.json")}
entries = [("simple: join best bid and ask", Trader("", touch, guard=hard_cap), None),
           ("textbook (Avellaneda-Stoikov), tuned", Trader("", make_as(g, k, a), guard=hard_cap), None),
           ("textbook, no imbalance signal", Trader("", make_as(g, k, 0.0), guard=hard_cap), None)]
pops = []
def add(label, key, guard):
    if key not in pol: return
    p = Population([pol[key]["weights"]], guard); pops.append(p); entries.append((label, p.traders[0], key))
add("learned, no guard", "unguarded", None)
add("learned (no-guard policy) + hard cap", "unguarded", hard_cap)
for al in (0.05, 0.2, 0.5): add(f"learned (no-guard policy) + barrier guard a={al}", "unguarded", make_cbf(al))
add("learned with hard cap in training", "capped", hard_cap)
add("learned with barrier guard in training", "cbf", make_cbf(0.2))

m = Market([e[1] for e in entries], batch=Batches(pops))
rep = pd.DataFrame(m.run(TEST)); rep["name"] = [e[0] for e in entries]
eq = np.array([e[1].equity for e in entries])                       # per-minute equity
hourly = np.diff(eq[:, ::60], axis=1)                                # profit per hour
rep["per_hour"] = hourly.mean(1).round(3); rep["se"] = (hourly.std(1, ddof=1) / np.sqrt(hourly.shape[1])).round(3)
d = hourly - hourly[0]
rep["vs_simple"] = d.mean(1).round(3); rep["vs_simple_se"] = (d.std(1, ddof=1) / np.sqrt(d.shape[1])).round(3)
rep["bps"] = (rep.total / rep.volume * 1e4).round(2)
print(f"\nTEST: recording 2, {rep.hours.iloc[0]} hours, mid {m.mids[0]:.0f} -> {m.mids[-1]:.0f}, {hourly.shape[1]} full hours | limit = 0.01 BTC (10 order sizes)")
cols = ["name", "fills", "volume", "spread", "adverse", "inventory", "total", "bps", "per_hour", "se", "vs_simple", "vs_simple_se"]
print(rep[cols].to_string(index=False))
print("\nRISK")
print(rep[["name", "max_inv", "breaches", "pct_time_at_limit", "mean_abs_inv_units", "overrides", "check"]].rename(columns={"breaches": "seconds_over_limit"}).to_string(index=False))
rep.drop(columns=["pen"]).to_csv("comparison.csv", index=False)
for n, p in pol.items(): print(f"policy {n}: chosen at generation {p['gen']}, weights (bid row, ask row; inputs inv, imbalance, momentum, spread, vol, const):", np.round(np.array(p["weights"]).reshape(2, 6), 2).tolist())
