"""The rough forex simulator on tiny hand-written bars: fill rules, the two brackets, the guard and no lookahead."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import pytest
from fx_bars import Bars, Q, SIZE, run, simple_quotes, cbf_size, cap_size, free_size, make_as

POINT = 0.01


def bars(rows, spread=2):
    """rows of (open, high, low, close) bid prices."""
    o, h, l, c = zip(*rows)
    return Bars(pd.date_range("2026-01-01", periods=len(rows), freq="min"), o, h, l, c, [spread] * len(rows), POINT)


def go(b, mode="both", sizes=cap_size): return run(b, simple_quotes, sizes, mode=mode)


def test_no_fill_without_a_move_through_our_quote():
    r = go(bars([(1.0, 1.0, 1.0, 1.0)]))         # bid never below the bid open, ask never above the ask open
    assert r["volume"][0] == 0 and r["total"][0] == 0


def test_bid_fills_when_the_low_goes_through_and_is_marked_at_the_close_mid():
    r = go(bars([(1.0, 1.0, 0.9, 0.95)]))
    assert r["max_inv"][0] == SIZE
    assert r["total"][0] == pytest.approx(0.95 + 0.01 - 1.0)      # bought at 1.00, mid at the close = 0.95 + half of 0.02


def test_both_sides_reached_earns_the_spread_in_the_optimistic_bracket():
    r = go(bars([(1.0, 1.1, 0.9, 1.0)]), mode="both")
    assert r["total"][0] == pytest.approx(0.02) and r["max_inv"][0] == 0


@pytest.mark.parametrize("row,inv", [((1.0, 1.1, 0.9, 1.05), -1), ((1.0, 1.1, 0.9, 0.95), 1), ((1.0, 1.1, 0.9, 1.0), 0)])
def test_adverse_bracket_keeps_only_the_losing_fill(row, inv):
    r = go(bars([row]), mode="adverse")
    assert r["max_inv"][0] == abs(inv)
    assert r["total"][0] <= 0                     # never earns the round trip


def falling(n):
    return bars([(100 - i, 100 - i, 99 - i, 99.5 - i) for i in range(n)])


def test_guard_holds_the_limit_in_a_one_way_market():
    assert go(falling(100), sizes=cbf_size)["max_inv"][0] <= Q + 1e-9
    assert go(falling(100), sizes=cap_size)["max_inv"][0] <= Q + 1e-9
    r = go(falling(100), sizes=free_size)
    assert r["max_inv"][0] == 100 and r["minutes_over"][0] == 90


def test_quotes_are_never_better_than_the_best_price():
    r = run(bars([(1.0, 1.0, 0.97, 1.0)]), lambda b, t, q: (np.array([-5.0]), np.array([-5.0])), cap_size)
    assert r["volume"][0] == pytest.approx(1.0)   # asks to improve; stays at 1.00, and fills there


def test_a_bid_dip_smaller_than_the_spread_is_not_a_fill():
    """The bid falls 0.01 and the spread is 0.02: the ask never reached our bid. This may just be a widening spread."""
    assert go(bars([(1.0, 1.0, 0.99, 1.0)]))["volume"][0] == 0


def test_no_fills_during_the_rollover():
    t = pd.date_range("2026-01-01 21:00", periods=1, freq="min")
    b = Bars(t, [1.0], [1.1], [0.9], [1.0], [2], POINT)
    assert go(b)["volume"][0] == 0


def test_a_decision_cannot_see_the_bar_it_trades_in():
    """Changing a bar's high, low and close must not change what the bot sees at that bar's open."""
    rows = [(1.0 + 0.01 * i, 1.02 + 0.01 * i, 0.99 + 0.01 * i, 1.01 + 0.01 * i) for i in range(80)]
    a = bars(rows)
    rows[70] = (rows[70][0], 2.0, 0.5, 1.9)
    b = bars(rows)
    assert (a.sd[:71] == b.sd[:71]).all() and (a.mom[:71] == b.mom[:71]).all()


def test_textbook_leans_away_from_inventory():
    b = bars([(1.0, 1.02, 0.98, 1.0)] * 70)
    ob, oa = make_as(0.5, 1.5)(b, 69, np.array([5.0]))
    b.sd[69] = b.u
    ob, oa = make_as(0.5, 1.5)(b, 69, np.array([5.0]))
    assert ob[0] > oa[0]                          # long: lower the bid and the ask
