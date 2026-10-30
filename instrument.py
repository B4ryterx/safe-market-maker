"""Per-market settings. Pick the market with the environment variable MM_INSTR (default BTC).

Order size and inventory limit are set to roughly the same dollar amount in both markets (about $110 per quote
and $1,100 of inventory), so results are comparable. Policies see prices in ticks, so one policy shape can be
tried on either market.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SPECS = {
    # train_h / val_h: training and validation hours at the start of the training recording.
    # val_h None = validate on everything after the training hours (the original BTC setup).
    "BTC": dict(tick=1.0, size=0.001, max_inv=0.01, ref_sd=2.36, rec1="data/btcusd_l3.csv", train_h=8, val_h=None),
    "ETH": dict(tick=0.1, size=0.05, max_inv=0.5, ref_sd=None, rec1="data/ethusd_l3.csv", train_h=8, val_h=4),
}
NAME = os.environ.get("MM_INSTR", "BTC").upper()
SPEC = dict(SPECS[NAME], rec1=os.path.join(HERE, SPECS[NAME]["rec1"]))
