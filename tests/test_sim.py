"""The fill model on tiny hand-written recordings: queue position, trades through our price, the pessimistic
queue, passive-only quotes, the profit split, and the guard under latency."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from sim import Market, Trader, U, touch, make_cbf

SIZE, Q = 0.001, 0.01
UNIT = int(SIZE * U)                      # one order of ours, in file units


def px(d): return int(d * U)


class Tape:
    """Writes lines in the recorder's format: O,id,price,amount,_,seq,ts  D,id,seq,ts  T,id,price,amount,seq,ts.
    Amounts are signed: a resting bid is positive; a trade with negative amount is a sell hitting the bids."""
    def __init__(self): self.lines = []
    def order(self, ts, oid, price, amt): self.lines.append(f"O,{oid},{px(price)},{amt},0,0,{ts}")
    def delete(self, ts, oid): self.lines.append(f"D,{oid},0,{ts}")
    def trade(self, ts, price, amt): self.lines.append(f"T,0,{px(price)},{amt},0,{ts}")
    def save(self, path):
        path.write_text("R\nS\n" + "\n".join(self.lines) + "\n"); return str(path)


def book(t):
    """Bid at 100 (order a, 5 of our units), ask at 102 (order b). Our first decision is at t = 1000."""
    t.order(0, "a", 100, 5 * UNIT); t.order(0, "b", 102, -5 * UNIT)


def only_bid(state): return state["bid"], None


def run(tape, tmp_path, agent=only_bid, **kw):
    tr = Trader("t", agent, size=SIZE, max_inv=Q, guard=kw.pop("guard", None))
    rep = Market([tr], **kw).run(tape.save(tmp_path / "rec.csv"))[0]
    return tr, rep


def test_order_ahead_of_us_must_trade_first(tmp_path):
    t = Tape(); book(t)
    t.order(1000, "x", 101, -UNIT)                       # any event at t = 1000 triggers our first decision: bid 100 behind a
    t.trade(1100, 100, -3 * UNIT); t.order(1200, "a", 100, 2 * UNIT)     # a trades 3: it was ahead of us, no fill
    tr, _ = run(t, tmp_path)
    assert tr.fills == []


def test_order_that_joined_after_us_trading_fills_us(tmp_path):
    t = Tape(); book(t)
    t.order(1000, "x", 101, -UNIT)                       # we join at 100 behind a
    t.order(1100, "c", 100, 3 * UNIT)                    # c joins behind us
    t.delete(1200, "a")                                  # a cancels: we are now first
    t.trade(1300, 100, -UNIT); t.order(1400, "c", 100, 2 * UNIT)        # c's size drops by the traded amount
    tr, _ = run(t, tmp_path)
    assert [f[3] for f in tr.fills] == [SIZE]


def test_pessimistic_queue_ignores_trades_at_our_price(tmp_path):
    t = Tape(); book(t)
    t.order(1000, "x", 101, -UNIT); t.order(1100, "c", 100, 3 * UNIT); t.delete(1200, "a")
    t.trade(1300, 100, -UNIT); t.order(1400, "c", 100, 2 * UNIT)
    tr, _ = run(t, tmp_path, through_only=True)
    assert tr.fills == []


def test_trade_through_our_price_fills_us(tmp_path):
    t = Tape(); book(t)
    t.order(1000, "x", 101, -UNIT)
    t.trade(1100, 99, -10 * UNIT)                        # a sell prints below our bid
    tr, _ = run(t, tmp_path)
    assert len(tr.fills) == 1 and tr.fills[0][2] == 100.0 and tr.inv == UNIT


def test_quotes_never_cross_the_spread(tmp_path):
    t = Tape(); book(t); t.order(1000, "x", 105, -UNIT)            # best ask stays 102
    t.trade(1100, 99, -10 * UNIT)
    tr, _ = run(t, tmp_path, agent=lambda s: (s["ask"] + 5, None))  # asks to bid far above the best ask
    assert tr.fills[0][2] == 101.0                       # clamped to one tick below the ask: passive only


def test_profit_split_adds_up(tmp_path):
    t = Tape(); book(t); t.order(1000, "x", 105, -UNIT)
    t.trade(1100, 99, -2 * U)                            # bought at 100 with the mid at 101
    t.delete(6000, "b")                                  # best ask 105: mid 102.5 ...
    t.order(11500, "y", 107, -UNIT)                      # ... still 102.5 ten seconds after the fill
    t.order(20000, "a3", 101, UNIT)                      # bid 101: mid 103 at the end
    tr = Trader("t", only_bid, size=1.0, max_inv=10.0)  # whole units: the report rounds to 3 decimals
    rep = Market([tr]).run(t.save(tmp_path / "rec.csv"))[0]
    assert (rep["spread"], rep["adverse"], rep["inventory"], rep["total"]) == (1.0, 1.5, 0.5, 3.0)
    assert rep["check"] == 0


def test_reconnect_drops_our_quotes(tmp_path):
    t = Tape(); book(t); t.order(1000, "x", 105, -UNIT)
    t.lines.append("S"); book(t)                         # fresh snapshot: our resting bid is gone
    t.trade(1100, 99, -10 * UNIT)
    tr, _ = run(t, tmp_path)
    assert tr.fills == []


def adversarial_tape(seconds):
    """Every second a huge sell prints through every bid, so whatever we post on the bid fills completely."""
    t = Tape(); book(t)
    for s in range(1, seconds):
        t.order(s * 1000, f"n{s}", 110 + s, -UNIT)       # keeps events coming; never changes the best ask
        t.trade(s * 1000 + 500, 50, -1000 * UNIT)
    return t


def alternating_bid(state):
    """Re-prices every second, so each new quote replaces the last one instead of keeping its place."""
    return state["bid"] - (state["ts"] // 1000) % 2, None


def test_guard_holds_the_limit_without_latency(tmp_path):
    tr, rep = run(adversarial_tape(40), tmp_path, agent=alternating_bid, guard=make_cbf(0.2))
    assert rep["max_inv"] <= Q + 1e-12 and rep["breaches"] == 0


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_latency_breaks_the_plain_guard_at_large_alpha(tmp_path, alpha):
    """The gap: with 1 s latency the outgoing quote and the incoming one were both sized against the same
    inventory, and both fill. With a large alpha each takes most of the remaining room, and together they pass Q."""
    _, rep = run(adversarial_tape(60), tmp_path, agent=alternating_bid, guard=make_cbf(alpha), latency_ms=1000)
    assert rep["max_inv"] > Q


@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.5, 1.0])
def test_counting_in_flight_size_holds_the_limit_under_latency(tmp_path, alpha):
    _, rep = run(adversarial_tape(60), tmp_path, agent=alternating_bid, guard=make_cbf(alpha, inflight=True), latency_ms=1000)
    assert rep["max_inv"] <= Q + 1e-12 and rep["breaches"] == 0


def test_published_alpha_survives_one_step_of_latency_here(tmp_path):
    """At alpha = 0.2 each quote takes only a fraction of the room left, so even two in-flight quotes stay inside.
    This is why the published latency runs never breached; it is a property of small alpha, not a guarantee."""
    _, rep = run(adversarial_tape(60), tmp_path, agent=alternating_bid, guard=make_cbf(0.2), latency_ms=1000)
    assert rep["max_inv"] <= Q
