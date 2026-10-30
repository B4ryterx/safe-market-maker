"""Volatility reference levels for the volatility-aware barrier, from the training hours only.
Prints the median and 90th percentile of the rolling 5-minute std of 1-second mid changes (in dollars).
Usage: MM_INSTR=ETH python3 calibrate_vol.py"""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from instrument import SPEC, NAME
from sim import Market, Trader
from learn import span

sds = []
def recorder(state):
    sds.append(math.sqrt(state["var"]))
    return None, None

t0, _ = span(SPEC["rec1"])
Market([Trader("rec", recorder)], t_to=t0 + SPEC["train_h"] * 3600 * 1000).run(SPEC["rec1"])
s = np.array(sds[300:])                     # skip the first 5 minutes while the window fills
print(f"{NAME}: {len(s)} seconds, median {np.median(s):.4f}, p90 {np.percentile(s, 90):.4f} (dollars; tick {SPEC['tick']})")
