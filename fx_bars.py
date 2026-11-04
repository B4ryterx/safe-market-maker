"""A rough forex version of the market-maker test, on 1-minute bars with a spread per bar.

The data (data/fx/<PAIR>_M1.json, Dukascopy through the Forex Software data app) has no orders, no trades and no
queue, so this is much weaker than sim.py. One decision a minute, on the bar's open:
  - Quotes are never better than the current best: bid <= bid open, ask >= ask open = bid open + spread.
  - A bid fills only if the bar's ask low (bid low + spread) reaches it; an ask only if the bar's bid high reaches it.
    A bid price merely dipping is not a fill: a spread widening looks the same, and nobody traded. (An earlier version
    counted such dips, and training found profit that was almost all from the daily rollover, when spreads blow out.)
  - No new quotes from 20:55 to 22:05 UTC, the rollover, when the spread column cannot be trusted within a bar.
  - A bar does not show which came first when both sides are reached, so every result is a bracket:
      both     both quotes fill and we earn the round trip (optimistic)
      adverse  only the fill that loses fills: the ask if the bar closed up, the bid if it closed down, neither if flat
Quotes rest for a whole minute, where a real forex market maker reprices in milliseconds, so adverse selection is overstated.

First 60% of the bars = training and tuning, last 40% = test. Training uses the adverse bracket and rule C from the
README (average of the last ten generations), so no validation window is needed.

Usage: fx_bars.py train <pair> <none|cbf> <seed>      writes fx_policies/<pair>_<guard>_s<seed>.json
       fx_bars.py test                                writes fx_results.csv and fx_results_hourly.npz
"""
import glob, json, math, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = ("EURUSD", "GBPUSD", "USDJPY")
EPOCH = pd.Timestamp("2000-01-01")     # the JSON's time field counts minutes from here
TRAIN_FRAC, ETA, ALPHA = 0.6, 0.01, 0.2
SIZE, Q = 1.0, 10.0                    # in orders, as in the Bitcoin test
F, POP, ELITE, GENS = 5, 96, 16, 25
GRID = [(g, k) for g in (0.02, 0.1, 0.5, 2.0) for k in (0.5, 1.5, 5.0)]


class Bars:
    """One pair's bars with everything a decision may see at the bar's open (past-only volatility and momentum)."""
    def __init__(self, time, o, h, l, c, spread, point):
        self.time, self.o, self.h, self.l, self.c = time, np.asarray(o, float), np.asarray(h, float), np.asarray(l, float), np.asarray(c, float)
        self.s, self.point = np.asarray(spread, float) * point, point
        self.n = len(self.o)
        mins = time.hour * 60 + time.minute
        self.quiet = np.asarray((mins >= 20 * 60 + 55) & (mins < 22 * 60 + 5))
        d = np.diff(self.o + self.s / 2, prepend=self.o[0] + self.s[0] / 2)
        self.sd = pd.Series(d).rolling(60, min_periods=2).std().shift(1).fillna(0.0).to_numpy(copy=True)   # copy: pandas 3 returns a read-only view
        self.mom = self.o - np.r_[np.full(10, self.o[0]), self.o[:-10]] if self.n > 10 else np.zeros(self.n)
        self.u = max(float(np.median(self.h - self.l)), point)   # the pair's scale: a typical bar's range

    def slice(self, i, j):
        b = Bars.__new__(Bars)
        b.__dict__.update(self.__dict__)
        for k in ("time", "o", "h", "l", "c", "s", "sd", "mom", "quiet"): setattr(b, k, getattr(self, k)[i:j])
        b.n = j - i
        return b


def load(pair):
    d = json.load(open(f"{HERE}/data/fx/{pair}_M1.json"))
    return Bars(EPOCH + pd.to_timedelta(np.array(d["time"]), unit="m"), d["open"], d["high"], d["low"], d["close"], d["spreads"], d["point"])


def cbf_size(q, alpha=ALPHA):
    """make_cbf in sim.py, for a whole population: bid and ask sizes that keep |q| <= Q if a side fills completely."""
    b = np.sqrt(alpha * Q * Q + (1 - alpha) * q * q)
    sb, sa = np.minimum(SIZE, b - q), np.minimum(SIZE, b + q)
    return np.where(sb * 10 < SIZE, 0.0, np.maximum(sb, 0.0)), np.where(sa * 10 < SIZE, 0.0, np.maximum(sa, 0.0))


def cap_size(q):
    return np.where(q + SIZE > Q, 0.0, SIZE), np.where(q - SIZE < -Q, 0.0, SIZE)


def free_size(q):
    return np.full(q.shape, SIZE), np.full(q.shape, SIZE)


def simple_quotes(b, t, q):
    return np.zeros(q.shape), np.zeros(q.shape)


def make_as(gamma, k):
    """Avellaneda-Stoikov around the mid, in units of b.u, as offsets behind the best prices."""
    half0 = math.log(1 + gamma / k) / gamma
    def quotes(b, t, q):
        var, half_s = (b.sd[t] / b.u) ** 2, b.s[t] / 2 / b.u
        half = gamma * var / 2 + half0 - half_s
        return half + q * gamma * var, half - q * gamma * var
    return quotes


def make_linear(w):
    """w has shape (n, 2, F): offsets behind the best bid and ask, in units of b.u, from the same inputs as learn.py
    without book imbalance, which these bars cannot show."""
    w = np.asarray(w, float).reshape(-1, 2, F)
    def quotes(b, t, q):
        x = np.ones((len(w), F))
        x[:, 0] = q / Q; x[:, 1] = np.tanh(b.mom[t] / b.u / 3); x[:, 2] = min(b.s[t] / b.u, 5) / 5; x[:, 3] = min(b.sd[t] / b.u, 3) / 3
        off = np.einsum("nkf,nf->nk", w, x)
        return off[:, 0], off[:, 1]
    return quotes


def run(b, quotes, sizes, n=1, mode="adverse"):
    """Returns total profit, volume and inventory penalty per policy (quote currency per order of base currency),
    plus the equity curve at each bar's close, shape (n, bars), and per-policy risk statistics."""
    q, cash, vol, pen, abs_q = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    max_q, over = np.zeros(n), np.zeros(n)
    eq = np.empty((n, b.n))
    for t in range(b.n):
        ob, oa = quotes(b, t, q)
        ask_o = b.o[t] + b.s[t]
        pb = np.floor((b.o[t] - np.maximum(ob, 0.0) * b.u) / b.point + 1e-9) * b.point
        pa = np.ceil((ask_o + np.maximum(oa, 0.0) * b.u) / b.point - 1e-9) * b.point
        sb, sa = sizes(q)
        tol = b.point * 1e-6
        hit_b, hit_a = b.l[t] + b.s[t] <= pb + tol, b.h[t] >= pa - tol
        if b.quiet[t]: hit_b, hit_a = np.zeros(n, bool), np.zeros(n, bool)
        if mode == "adverse":
            both = hit_b & hit_a
            up, down = b.c[t] > b.o[t], b.c[t] < b.o[t]
            hit_b = hit_b & (~both | down)       # closed down: we bought and it fell
            hit_a = hit_a & (~both | up)         # closed up: we sold and it rose
        fb, fa = np.where(hit_b, sb, 0.0), np.where(hit_a, sa, 0.0)
        q = q + fb - fa
        cash += fa * pa - fb * pb
        vol += fa * pa + fb * pb
        eq[:, t] = cash + q * (b.c[t] + b.s[t] / 2)
        pen += (q / SIZE) ** 2; abs_q += np.abs(q)
        max_q = np.maximum(max_q, np.abs(q)); over += np.abs(q) > Q + 1e-9
    return dict(total=eq[:, -1].copy(), volume=vol, pen=pen, eq=eq, mean_inv=abs_q / b.n, max_inv=max_q, minutes_over=over)


def train(pair, guard, seed):
    b = load(pair); b = b.slice(0, int(b.n * TRAIN_FRAC))
    sizes = cbf_size if guard == "cbf" else free_size
    rng = np.random.default_rng(seed)
    mean, std = np.zeros(2 * F), np.ones(2 * F)
    hist = []
    for gen in range(GENS):
        cand = mean + std * rng.standard_normal((POP, 2 * F)); cand[0] = mean
        r = run(b, make_linear(cand), sizes, n=POP)
        J = r["total"] / b.u - ETA * r["pen"]
        elite = cand[np.argsort(J)[-ELITE:]]
        mean, std = elite.mean(0), elite.std(0) + 0.05
        hist.append(mean.tolist())
        print(f"{pair} {guard} s{seed} gen {gen:2d} median J {np.median(J):+.1f} best {J.max():+.1f}", flush=True)
    os.makedirs(f"{HERE}/fx_policies", exist_ok=True)
    json.dump(dict(weights=np.mean(hist[-10:], axis=0).tolist()), open(f"{HERE}/fx_policies/{pair}_{guard}_s{seed}.json", "w"))


def tune_as(b):
    """Best Avellaneda-Stoikov parameters on the training bars, adverse bracket."""
    scores = {gk: float(run(b, make_as(*gk), cap_size)["total"][0]) for gk in GRID}
    return max(scores, key=scores.get)


def hourly(b, eq):
    s = pd.Series(eq, index=b.time)
    return s.groupby(s.index.floor("h")).last().diff().dropna().to_numpy()


def daily(b, eq):
    s = pd.Series(eq, index=b.time)
    return s.groupby(s.index.floor("D")).last().diff().dropna().to_numpy()


def t_stat(x):
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12))


def test():
    rows = []; saved = {}
    for pair in PAIRS:
        full = load(pair); cut = int(full.n * TRAIN_FRAC)
        tr, te = full.slice(0, cut), full.slice(cut, full.n)
        gk = tune_as(tr)
        bots = [("simple", simple_quotes, cap_size), (f"textbook {gk}", make_as(*gk), cap_size)]
        for f in sorted(glob.glob(f"{HERE}/fx_policies/{pair}_*.json")):
            tag = os.path.basename(f)[len(pair) + 1:-5]; w = json.load(open(f))["weights"]
            if tag.startswith("cbf"): bots.append((f"learned, guard in training {tag[-2:]}", make_linear(w), cbf_size))
            else:
                bots.append((f"learned, no guard {tag[-2:]}", make_linear(w), free_size))
                bots.append((f"no-guard policy + guard {tag[-2:]}", make_linear(w), cbf_size))
        for mode in ("both", "adverse"):
            res = [(name, run(te, qt, sz, mode=mode)) for name, qt, sz in bots]
            ref = {k: (hourly(te, r["eq"][0]), daily(te, r["eq"][0])) for k, r in ((res[0][0], res[0][1]), (res[1][0], res[1][1]))}
            for name, r in res:
                h, d = hourly(te, r["eq"][0]), daily(te, r["eq"][0])
                row = dict(pair=pair, mode=mode, bot=name, fills_volume=round(float(r["volume"][0]), 1),
                           bps_of_volume=round(float(r["total"][0] / max(r["volume"][0], 1e-9) * 1e4), 3), total_in_u=round(float(r["total"][0] / te.u), 1),
                           mean_inv=round(float(r["mean_inv"][0]), 2), max_inv=round(float(r["max_inv"][0]), 2), minutes_over=int(r["minutes_over"][0]),
                           hours=len(h), days=len(d))
                for k, (rh, rd) in ref.items():
                    lab = "simple" if k == "simple" else "textbook"
                    row[f"t_h_vs_{lab}"], row[f"t_d_vs_{lab}"] = round(t_stat(h - rh), 2) if name != k else 0.0, round(t_stat(d - rd), 2) if name != k else 0.0
                rows.append(row); saved[f"{pair}|{mode}|{name}"] = h
        print(pair, "test", te.time[0], "to", te.time[-1], "bars", te.n, "u", te.u, "median spread", float(np.median(te.s)), "as", gk, flush=True)
    rep = pd.DataFrame(rows); rep.to_csv(f"{HERE}/fx_results.csv", index=False)
    np.savez(f"{HERE}/fx_results_hourly.npz", **saved)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(rep.drop(columns=["fills_volume"]).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "train": train(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else: test()
