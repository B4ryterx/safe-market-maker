"""Tune the textbook market maker on the first 6 hours and test it, untouched, on the last 6."""
import itertools, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sim import Market, Trader, touch, make_as, hard_cap
pd.set_option("display.width", 250)
PATH = sys.argv[1] if len(sys.argv) > 1 else "data/btcusd_l3.csv"

first = last = None
with open(PATH) as f:
    for line in f:
        if line[0] in "OD":
            ts = int(line.rstrip().rsplit(",", 1)[1]); first = first or ts; last = ts
mid_ts = (first + last) // 2

grid = []
for gamma, k, alpha, cross in itertools.product((0.02, 0.1, 0.5, 2.0), (0.5, 1.5, 5.0), (0.0, 1.0, 3.0), (False, True)):
    grid.append(Trader(f"g{gamma} k{k} a{alpha} {'cross' if cross else 'passive'}", make_as(gamma, k, alpha), cross_at_limit=cross, guard=hard_cap))
t0 = time.time()
train = pd.DataFrame(Market(grid, t_to=mid_ts).run(PATH))
print(f"tuning pass: {len(grid)} settings on the first {train.hours.iloc[0]} hours ({time.time() - t0:.0f}s)")
train[["gamma", "k", "alpha", "mode"]] = train.name.str.extract(r"g([\d.]+) k([\d.]+) a([\d.]+) (\w+)")
print("profitable settings in the tuning half:", int((train.total > 0).sum()), "of", len(train))
print(train.sort_values("total", ascending=False).head(5)[["name", "fills", "volume", "total", "spread", "adverse", "inventory", "pct_time_at_limit"]].to_string(index=False))
print("average total by setting:", {c: train.groupby(c).total.mean().round(2).to_dict() for c in ("gamma", "k", "alpha", "mode")})

best = train.sort_values("total", ascending=False).iloc[0]
g, k, a, cross = float(best.gamma), float(best.k), float(best.alpha), best["mode"] == "cross"
test_traders = [Trader("join the best bid and ask", touch, guard=hard_cap),
                Trader("textbook, default settings", make_as(), guard=hard_cap),
                Trader(f"textbook, tuned ({best['name']})", make_as(g, k, a), cross_at_limit=cross, guard=hard_cap),
                Trader("tuned, without the imbalance signal", make_as(g, k, 0.0), cross_at_limit=cross, guard=hard_cap),
                Trader("tuned, at standard fees (10 bps maker, 20 taker)", make_as(g, k, a), cross_at_limit=cross, guard=hard_cap, maker_fee_bps=10, taker_fee_bps=20)]
m = Market(test_traders, t_from=mid_ts)
test = pd.DataFrame(m.run(PATH)); test["after_fees"] = (test.total - test.fees).round(3)
print(f"\nTEST on the last {test.hours.iloc[0]} hours (never used for tuning) | mid {m.mids[0]:.0f} -> {m.mids[-1]:.0f} | prints with no book change: {m.hidden}")
print(test[["name", "fills", "taker_fills", "volume", "spread", "adverse", "inventory", "total", "after_fees", "pct_time_at_limit", "breaches", "check"]].to_string(index=False))
train.to_csv("baseline_tuning.csv", index=False); test.to_csv("baseline_test.csv", index=False)
