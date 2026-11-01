"""The barrier guard on its own: the closed form, minimal intervention, and the invariant |q| <= Q against an
adversary that fills everything we post."""
import math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from sim import Trader, Quote, make_cbf, make_cbf_vol, hard_cap

SIZE, Q = 0.001, 0.01          # the published setting: 10 orders of inventory


def trader(inv_units=0, guard=None):
    tr = Trader("t", None, size=SIZE, max_inv=Q, guard=guard)
    tr.inv = inv_units * tr.size
    return tr


def h(q, Q): return Q * Q - q * q


@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.5, 1.0])
def test_condition_holds_for_every_inventory(alpha):
    """h(q + bid size) >= (1 - alpha) h(q), and the same for the ask, at every reachable inventory."""
    g = make_cbf(alpha)
    for inv in range(-10, 11):
        tr = trader(inv)
        _, _, sb, sa = g(tr, {"ts": 0}, 1.0, 2.0)
        q, Qi = tr.inv, tr.max_inv
        assert h(q + sb, Qi) >= (1 - alpha) * h(q, Qi) - 1e-6 * Qi * Qi
        assert h(q - sa, Qi) >= (1 - alpha) * h(q, Qi) - 1e-6 * Qi * Qi
        assert 0 <= sb <= tr.size and 0 <= sa <= tr.size


def test_minimal_intervention_far_from_the_limit():
    tr = trader(0)
    assert make_cbf(0.2)(tr, {"ts": 0}, 1.0, 2.0) == (1.0, 2.0, tr.size, tr.size)
    assert tr.overrides == 0


def test_shrinks_only_the_risky_side_near_the_limit():
    tr = trader(9)
    _, _, sb, sa = make_cbf(0.2)(tr, {"ts": 0}, 1.0, 2.0)
    assert sb < tr.size and sa == tr.size


def test_alpha_one_lets_inventory_reach_the_limit_like_the_hard_cap():
    tr = trader(9)
    _, _, sb, _ = make_cbf(1.0)(tr, {"ts": 0}, 1.0, 2.0)
    assert tr.inv + sb == tr.max_inv
    assert hard_cap(trader(10), {}, 1.0, 2.0)[0] is None


@pytest.mark.parametrize("alpha", [0.05, 0.2, 1.0])
def test_invariant_against_an_adversary(alpha):
    """No latency: whatever is posted fills completely, on a random side, every step. |q| never passes Q."""
    rng = random.Random(0)
    g = make_cbf(alpha)
    for _ in range(200):
        tr = trader(rng.randint(-10, 10))
        for _ in range(100):
            _, _, sb, sa = g(tr, {"ts": 0}, 1.0, 2.0)
            tr.inv += sb if rng.random() < 0.5 else -sa
            assert abs(tr.inv) <= tr.max_inv


def test_volatility_aware_limit_only_lets_the_position_shrink():
    """When the market is twice as jumpy as the reference, Qe = Q/2: a long position above Qe gets no bid."""
    tr = trader(7)
    _, _, sb, sa = make_cbf_vol(0.2, ref_sd=1.0)(tr, {"ts": 0, "var": 4.0}, 1.0, 2.0)
    assert sb == 0 and sa == tr.size
    tr = trader(7)                                             # calm market: the ordinary barrier
    assert make_cbf_vol(0.2, ref_sd=1.0)(tr, {"ts": 0, "var": 0.25}, 1.0, 2.0)[2:] == make_cbf(0.2)(trader(7), {"ts": 0}, 1.0, 2.0)[2:]


def test_in_flight_size_is_counted():
    """A replaced quote whose cancel has not landed can still fill, so the in-flight guard sizes against q + its size."""
    tr = trader(5)
    tr.old[1] = Quote(price=0, rem=4 * tr.size, ahead=set(), live=0, dead=500)    # cancel lands at t = 500 ms
    _, _, plain, _ = make_cbf(0.2)(tr, {"ts": 0}, 1.0, 2.0)
    _, _, safe, _ = make_cbf(0.2, inflight=True)(tr, {"ts": 0}, 1.0, 2.0)
    q = tr.inv + 4 * tr.size
    assert safe == min(tr.size, int(math.sqrt(0.2 * tr.max_inv ** 2 + 0.8 * q * q) - q)) < plain
    _, _, after, _ = make_cbf(0.2, inflight=True)(tr, {"ts": 600}, 1.0, 2.0)         # once it is dead it no longer counts
    assert after == plain
