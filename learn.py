"""Learning market maker: a small policy found by trial and error (cross-entropy method, a form of
evolution strategy), trained on recorded order-by-order data with honest queue fills.

Policy: each second, from 6 inputs (inventory, book imbalance, 10-second price move, spread, volatility, constant)
it chooses how many ticks behind the best bid and best ask to quote: -1 = improve the price by one tick,
0 = join the best price, 1-3 = sit deeper, 4 = do not quote that side.
Objective: profit minus ETA x sum over seconds of (inventory in order units)^2.

Data: recording 1, first 8 hours = training; recording 1, last 4 hours = validation (picks the generation);
recording 2 is never seen here.
Usage: learn.py <name> [none|cap|cbf]      guard used during training
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sim import Market, Trader, hard_cap, make_cbf

ETA = 2e-6
F = 6
REC1 = "data/btcusd_l3.csv"


def span(path):
    first = last = None
    with open(path) as f:
        for line in f:
            if line[0] in "OD":
                ts = int(line.rstrip().rsplit(",", 1)[1]); first = first or ts; last = ts
    return first, last


class Population:
    """Evaluates many policies in one pass: weights has shape (n, 2, F)."""
    def __init__(self, weights, guard=None, max_inv=0.01):
        self.w = np.asarray(weights, float).reshape(-1, 2, F)
        self.traders = [Trader(f"p{i}", None, guard=guard, max_inv=max_inv) for i in range(len(self.w))]
        self.Q = 0.01 * 1e8
        self.x = np.ones((len(self.w), F))

    def assign(self, base):
        x = self.x
        x[:, 0] = [t.inv for t in self.traders]; x[:, 0] /= self.Q
        x[:, 1] = base["imbalance"]; x[:, 2] = np.tanh(base["mom"] / 3.0)
        x[:, 3] = min(base["spread"], 5.0) / 5.0; x[:, 4] = min(base["var"] ** 0.5, 3.0) / 3.0
        off = np.clip(np.rint(np.einsum("nkf,nf->nk", self.w, x)), -1, 4)
        bid, ask = base["bid"], base["ask"]
        for t, (ob, oa) in zip(self.traders, off):
            t.act = (None if ob >= 4 else bid - ob, None if oa >= 4 else ask + oa)


def evaluate(weights, path, t_from=None, t_to=None, guard=None, max_inv=0.01):
    pop = Population(weights, guard, max_inv)
    rep = Market(pop.traders, t_from=t_from, t_to=t_to, batch=pop).run(path)
    return np.array([r["total"] - ETA * r["pen"] for r in rep]), rep


if __name__ == "__main__":
    name, guard_name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "none")
    guard = {"none": None, "cap": hard_cap, "cbf": make_cbf(0.2)}[guard_name]
    max_inv = 1.0 if guard is None else 0.01          # no guard: the limit is not enforced, only counted
    t0, t1 = span(REC1); t_val = t0 + 8 * 3600 * 1000
    rng = np.random.default_rng(0)
    mean, std = np.zeros(2 * F), np.full(2 * F, 1.0)
    POP, ELITE, GENS = 96, 16, 25
    best = dict(val=-1e9)
    log = open(f"train_{name}.log", "w")
    for gen in range(GENS):
        tic = time.time()
        cand = mean + std * rng.standard_normal((POP, 2 * F)); cand[0] = mean
        J, _ = evaluate(cand, REC1, t_to=t_val, guard=guard, max_inv=max_inv)
        elite = cand[np.argsort(J)[-ELITE:]]
        mean, std = elite.mean(0), elite.std(0) + 0.05
        Jv, rv = evaluate([mean], REC1, t_from=t_val, guard=guard, max_inv=max_inv)
        if Jv[0] > best["val"]: best = dict(val=float(Jv[0]), gen=gen, weights=mean.tolist(), guard=guard_name)
        msg = f"gen {gen:2d} | train best {J.max():+.2f} median {np.median(J):+.2f} | mean policy on validation {Jv[0]:+.2f} (profit {rv[0]['total']:+.2f}, fills {rv[0]['fills']}) | {time.time() - tic:.0f}s"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        json.dump(best, open(f"policy_{name}.json", "w"))
    print("chosen generation", best["gen"], "validation objective", round(best["val"], 3))
