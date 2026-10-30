"""Regression check after code changes: the simple bot and the guard-trained policy on recording 2 must
reproduce the published numbers (2,444 fills / +$3.487 and +$5.704)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import Market, Trader, touch, hard_cap, make_cbf
from learn import Population
HERE = os.path.dirname(os.path.abspath(__file__))
pop = Population([json.load(open(f"{HERE}/policy_cbf.json"))["weights"]], make_cbf(0.2))
simple = Trader("simple", touch, guard=hard_cap)
rep = Market([simple] + pop.traders, batch=pop).run(f"{HERE}/data/btcusd_l3_part2.csv")
for r in rep: print(r["name"], r["fills"], round(r["total"], 3))
