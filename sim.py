"""Market-making simulator on recorded order-by-order data, with honest fills.

The market's book is rebuilt from the recording. Our quotes are virtual: they do not move the market.
A quote joins the back of the queue at its price. It fills only when
  - a trade prints at a worse price than ours (the aggressor would have reached us first), or
  - a trade at our price hits an order that joined the queue after we did.
On this feed a trade prints first and the book change follows about 0.1-0.2 s later. A book reduction
counts as traded up to the size of the prints waiting at that price; the rest of it is a cancel.
Prints with no book change at all (hidden orders, about 2%) never fill us.

Profit is split per fill:  spread    = distance from the mid at the moment of the fill
                           adverse   = mid move over the next HORIZON seconds (who we traded with)
                           inventory = mid move after that until the end (what we were left holding)

Many traders can share one pass over the data, because none of them moves the market.
"""
import math
from bisect import bisect_left
from collections import deque

from sortedcontainers import SortedDict

U = 10**8                      # file units per 1.0 of price or size
TICK = 1 * U                   # BTCUSD moves in whole dollars at this price level
HORIZON_MS = 10_000
VAR_WINDOW = 300               # seconds of mid changes in the volatility estimate


class Quote:
    __slots__ = ("price", "rem", "ahead")
    def __init__(self, price, rem, ahead): self.price, self.rem, self.ahead = price, rem, ahead


class Trader:
    """One agent's quotes, inventory and fills. agent(state) -> (bid, ask) in dollars, None = no quote."""
    def __init__(self, name, agent, size=0.001, max_inv=0.01, cross_at_limit=False, taker_fee_bps=0.0, maker_fee_bps=0.0, guard=None):
        self.name, self.agent, self.size, self.max_inv = name, agent, int(size * U), int(max_inv * U)
        self.cross, self.taker_fee, self.maker_fee, self.guard = cross_at_limit, taker_fee_bps / 1e4, maker_fee_bps / 1e4, guard
        self.q = {1: None, -1: None}
        self.inv, self.cash, self.fees = 0, 0.0, 0.0
        self.fills = []                # (ts, side, price, qty, mid, taker)
        self.max_abs_inv, self.breaches, self.steps_at_limit, self.overrides = 0, 0, 0, 0
        self.pen, self.abs_inv_sum, self.act = 0.0, 0.0, None      # act: quotes set by a population policy
        self.equity = []

    def book_fill(self, ts, side, price, qty, mid, taker=False):
        self.inv += side * qty
        notional = price / U * qty / U
        self.cash -= side * notional
        self.fees += notional * (self.taker_fee if taker else self.maker_fee)
        self.fills.append((ts, side, price / U, qty / U, mid / U, taker))
        self.max_abs_inv = max(self.max_abs_inv, abs(self.inv))


class Market:
    def __init__(self, traders, step_ms=1000, t_from=None, t_to=None, batch=None):
        self.traders, self.step_ms, self.t_from, self.t_to, self.batch = traders, step_ms, t_from, t_to, batch
        self.orders = {}                                   # id -> [price, signed amount]
        self.lv = {1: SortedDict(), -1: SortedDict()}      # side -> price -> {id: size}; dict order = arrival order
        self.pending = deque()                             # trade prints awaiting their book change: [ts, price, qty, side]
        self.mids_t, self.mids, self.next_step, self.mid_now = [], [], None, None
        self.sq = deque(); self.sq_sum = 0.0               # rolling squared mid changes
        self.hidden = 0

    def best(self, side):
        d = self.lv[side]
        return (d.peekitem(-1)[0] if side == 1 else d.peekitem(0)[0]) if d else None

    def depth(self, side, n=5):
        d = self.lv[side]; keys = d.keys()[-n:] if side == 1 else d.keys()[:n]
        return sum(sum(d[k].values()) for k in keys)

    def _remove(self, oid):
        price, amt = self.orders.pop(oid); side = 1 if amt > 0 else -1
        level = self.lv[side][price]; del level[oid]
        if not level: del self.lv[side][price]

    def _add(self, oid, price, amt):
        self.orders[oid] = [price, amt]
        self.lv[1 if amt > 0 else -1].setdefault(price, {})[oid] = abs(amt)

    def _fill(self, tr, ts, side, qty):
        q = tr.q[side]; qty = min(qty, q.rem)
        if qty <= 0: return
        q.rem -= qty
        tr.book_fill(ts, side, q.price, qty, self.mid_now)
        if q.rem == 0: tr.q[side] = None

    def _on_trade(self, ts, price, amt):
        side = -1 if amt > 0 else 1            # resting side that was hit: a buy aggressor hits asks
        for tr in self.traders:
            q = tr.q[side]
            if q and ((side == 1 and price < q.price) or (side == -1 and price > q.price)):
                self._fill(tr, ts, side, abs(amt))          # printed through our price
        self.pending.append([ts, price, abs(amt), side])

    def _on_reduction(self, ts, oid, price, side, dq, gone):
        while self.pending and ts - self.pending[0][0] > 5000:
            self.pending.popleft(); self.hidden += 1
        traded, left = 0, dq
        for pr in self.pending:
            if left <= 0: break
            if pr[1] == price and pr[3] == side and pr[2] > 0:
                take = min(pr[2], left); pr[2] -= take; left -= take; traded += take
        if traded:
            self.pending = deque(p for p in self.pending if p[2] > 1)
        for tr in self.traders:
            q = tr.q[side]
            if not q: continue
            if oid in q.ahead:
                if gone: q.ahead.discard(oid)
            elif traded and price == q.price:
                self._fill(tr, ts, side, traded)             # an order behind us traded: we would have traded first

    def _step(self, ts):
        b, a = self.best(1), self.best(-1)
        if not b or not a or b >= a: return
        m = (b + a) / 2
        if self.mids:
            d2 = (m / U - self.mids[-1]) ** 2
            self.sq.append(d2); self.sq_sum += d2
            if len(self.sq) > VAR_WINDOW: self.sq_sum -= self.sq.popleft()
        self.mid_now = m; self.mids_t.append(ts); self.mids.append(m / U)
        db, da = self.depth(1), self.depth(-1)
        base = dict(ts=ts, mid=m / U, bid=b / U, ask=a / U, spread=(a - b) / U, imbalance=(db - da) / (db + da) if db + da else 0.0,
                    var=self.sq_sum / len(self.sq) if self.sq else 1.0,
                    mom=self.mids[-1] - self.mids[-11] if len(self.mids) > 10 else 0.0)
        if self.batch: self.batch.assign(base)
        for tr in self.traders:
            if abs(tr.inv) + tr.size > tr.max_inv: tr.steps_at_limit += 1
            if abs(tr.inv) > tr.max_inv: tr.breaches += 1
            u = tr.inv / tr.size; tr.pen += u * u; tr.abs_inv_sum += abs(u)
            if len(self.mids) % 60 == 1: tr.equity.append(tr.cash + tr.inv / U * (m / U))     # one sample a minute
            if tr.cross and abs(tr.inv) + tr.size > tr.max_inv:      # at the limit: pay the spread to get one unit flatter
                side = -1 if tr.inv > 0 else 1
                tr.book_fill(ts, side, a if side == 1 else b, tr.size, m, taker=True)
            state = dict(base, inv=tr.inv / U, size=tr.size / U, max_inv=tr.max_inv / U)
            bid, ask = tr.act if tr.act is not None else tr.agent(state)
            sb = sa = tr.size
            if tr.guard: bid, ask, sb, sa = tr.guard(tr, state, bid, ask)
            pb = None if bid is None else min(int(math.floor(bid)) * U, a - TICK)     # passive only: never cross
            pa = None if ask is None else max(int(math.ceil(ask)) * U, b + TICK)
            for side, price, sz in ((1, pb, sb), (-1, pa, sa)):
                cur = tr.q[side]
                if price is None or sz * 10 < tr.size: tr.q[side] = None             # no quote, or guard cut it to nothing
                elif cur and cur.price == price and cur.rem > 0: cur.rem = min(cur.rem, sz)   # unchanged quote keeps its queue place
                else: tr.q[side] = Quote(price, sz, set(self.lv[side].get(price, ())))

    def run(self, path):
        active_before = False
        with open(path) as f:
            for line in f:
                p = line.rstrip("\n").split(",")
                k = p[0]
                if k == "O":
                    oid, price, amt, ts = p[1], int(p[2]), int(p[3]), int(p[6])
                    old = self.orders.get(oid)
                    if old is None:
                        self._add(oid, price, amt)
                    elif old[0] == price and abs(amt) < abs(old[1]):
                        dq = abs(old[1]) - abs(amt); side = 1 if amt > 0 else -1
                        old[1] = amt; self.lv[side][price][oid] = abs(amt)
                        if active_before: self._on_reduction(ts, oid, price, side, dq, False)
                    elif old[0] != price or abs(amt) != abs(old[1]):
                        self._remove(oid); self._add(oid, price, amt)       # repriced or enlarged: back of the queue
                elif k == "D":
                    oid, ts = p[1], int(p[3])
                    old = self.orders.get(oid)
                    if old is None: continue
                    side = 1 if old[1] > 0 else -1
                    self._remove(oid)
                    if active_before: self._on_reduction(ts, oid, old[0], side, abs(old[1]), True)
                elif k == "T":
                    ts = int(p[5])
                    if active_before and self.mid_now: self._on_trade(ts, int(p[2]), int(p[3]))
                elif k == "S":                                               # reconnect: fresh book, our quotes are gone
                    self.orders.clear(); self.lv[1].clear(); self.lv[-1].clear(); self.pending.clear()
                    for tr in self.traders: tr.q = {1: None, -1: None}
                    continue
                else:
                    continue
                active = (self.t_from is None or ts >= self.t_from) and (self.t_to is None or ts < self.t_to)
                if active and (self.next_step is None or ts >= self.next_step):
                    self._step(ts); self.next_step = ts + self.step_ms
                if active_before and not active:
                    for tr in self.traders: tr.q = {1: None, -1: None}
                active_before = active
        return [self.report(tr) for tr in self.traders]

    def report(self, tr):
        end_mid = self.mids[-1]; spread = adverse = carry = 0.0
        for ts, side, price, qty, mid, taker in tr.fills:
            j = min(bisect_left(self.mids_t, ts + HORIZON_MS), len(self.mids) - 1)
            spread += side * (mid - price) * qty
            adverse += side * (self.mids[j] - mid) * qty
            carry += side * (end_mid - self.mids[j]) * qty
        total = tr.cash + tr.inv / U * end_mid
        vol = sum(f[3] * f[2] for f in tr.fills)
        return dict(name=tr.name, hours=round((self.mids_t[-1] - self.mids_t[0]) / 3.6e6, 2), fills=len(tr.fills),
                    taker_fills=sum(1 for f in tr.fills if f[5]), volume=round(vol), total=round(total, 3), spread=round(spread, 3),
                    adverse=round(adverse, 3), inventory=round(carry, 3), check=round(total - (spread + adverse + carry), 6),
                    fees=round(tr.fees, 3), end_inv=tr.inv / U, max_inv=tr.max_abs_inv / U, breaches=tr.breaches,
                    pct_time_at_limit=round(100 * tr.steps_at_limit / max(len(self.mids), 1), 1), overrides=tr.overrides,
                    pen=tr.pen, mean_abs_inv_units=round(tr.abs_inv_sum / max(len(self.mids), 1), 2))


# ---- agents
def touch(state):
    """Always join the best bid and best ask."""
    return state["bid"], state["ask"]


def make_as(gamma=0.5, k=1.5, alpha=0.0):
    """Avellaneda-Stoikov with a rolling volatility estimate. Inventory is counted in units of order size.
    alpha > 0 adds the order-book imbalance signal: lean both quotes toward the heavier side."""
    def agent(state):
        q = state["inv"] / state["size"]
        r = state["mid"] - q * gamma * state["var"] + alpha * state["imbalance"]
        half = gamma * state["var"] / 2 + math.log(1 + gamma / k) / gamma
        return r - half, r + half
    return agent


def hard_cap(tr, state, bid, ask):
    """Simplest inventory guard: do not quote the side that would take us past the limit."""
    if tr.inv + tr.size > tr.max_inv: bid = None; tr.overrides += 1
    if tr.inv - tr.size < -tr.max_inv: ask = None; tr.overrides += 1
    return bid, ask, tr.size, tr.size


def make_cbf(alpha=0.2):
    """Control barrier function on inventory. Safe set: h(q) = Q^2 - q^2 >= 0.
    Discrete-time condition h(q_next) >= (1 - alpha) h(q), enforced against the worst case that the whole
    posted size on one side fills before the next decision:
        bid size <= sqrt(alpha Q^2 + (1 - alpha) q^2) - q      ask size <= sqrt(alpha Q^2 + (1 - alpha) q^2) + q
    The guard changes nothing while the desired size already satisfies this (minimal intervention), shrinks
    the risky side as inventory nears the limit, and guarantees |q| <= Q. alpha = 1 is the plain hard cap."""
    def guard(tr, state, bid, ask):
        Q, q = tr.max_inv, tr.inv
        bound = math.sqrt(alpha * Q * Q + (1 - alpha) * q * q)
        sb, sa = min(tr.size, int(bound - q)), min(tr.size, int(bound + q))
        if sb < tr.size or sa < tr.size: tr.overrides += 1
        return bid, ask, max(sb, 0), max(sa, 0)
    return guard
