# Safe learning market maker

Does a market-making policy found by trial and error beat the textbook formula, and what does a
barrier-function safety guard cost? Tested on real order-by-order data with queue-aware fills.

## Result

Test set: 12 hours of Bitfinex BTCUSD that no bot saw during tuning. Order size 0.001 BTC, inventory
limit 0.01 BTC (10 orders), zero maker fee.

| Bot | Fills | Spread earned | Adverse selection | Inventory | Total | Mean inventory (orders) | Time at limit |
|---|---|---|---|---|---|---|---|
| Simple: join best bid and ask | 2,444 | +$6.89 | −$3.69 | +$0.28 | +$3.49 | 8.9 | 67% |
| Avellaneda–Stoikov, tuned | 3,384 | +$3.20 | −$3.54 | +$1.06 | +$0.71 | 9.2 | 82% |
| Learned, no guard | 3,663 | +$9.47 | −$4.97 | +$0.91 | +$5.40 | 2.8 | 0.3% |
| Learned + hard cap | 3,661 | +$9.46 | −$4.97 | +$0.91 | +$5.40 | 2.8 | 0.3% |
| Learned + barrier guard, α = 0.2 | 3,662 | +$9.43 | −$4.92 | +$0.84 | +$5.36 | 2.7 | 0% |
| Learned + barrier guard, α = 0.05 | 3,671 | +$9.09 | −$4.77 | +$0.38 | +$4.71 | 2.2 | 0% |
| Learned with barrier guard in training | 3,462 | +$9.19 | −$4.45 | +$0.96 | +$5.70 | 2.4 | 0% |

- The learned policy carries about a third of the inventory of both baselines and is almost never pinned at the limit.
- Its profit lead over the simple bot is +$0.20 per hour ± $0.12 (standard error over nine hourly blocks): suggestive, not established.
- The guard with α = 0.2 cost under 1% of profit; α = 0.05 cost about 13% and cut the largest position from 0.0093 to 0.0074 BTC.
- At test time the trained policy stayed inside the limit without help. The guard mattered in training: the median
  first-generation policy scored about −$714 without a guard and about −$2.50 with one.
- At Bitfinex's standard 10 bps maker fee every bot loses heavily. The edge is about 0.5 bps of volume.

## Second test: a further 24 hours

A third recording, made after everything above was fixed: 24 hours with nine short disconnections, 21.6 hours of
data, 3.44 million order updates, 39,115 trades, all 29,397 exchange checksums matched. No bot was retuned. The
price rose 1.2% over the day and every bot lost money.

| Bot | Fills | Spread earned | Adverse selection | Inventory | Total | Mean inventory (orders) | Time at limit |
|---|---|---|---|---|---|---|---|
| Simple: join best bid and ask | 10,230 | +$31.58 | −$28.76 | −$11.24 | −$8.41 | 7.9 | 49% |
| Avellaneda–Stoikov, tuned | 14,432 | +$15.41 | −$31.69 | −$11.45 | −$27.73 | 8.3 | 59% |
| Learned, no guard | 15,020 | +$42.32 | −$36.86 | −$8.08 | −$2.61 | 4.1 | 8% |
| Learned + hard cap | 14,109 | +$39.66 | −$35.00 | −$10.20 | −$5.55 | 3.4 | 2.4% |
| Learned + barrier guard, α = 0.2 | 14,241 | +$39.01 | −$34.32 | −$9.84 | −$5.16 | 3.2 | 0.4% |
| Learned + barrier guard, α = 0.05 | 13,701 | +$35.83 | −$31.00 | −$7.52 | −$2.69 | 2.3 | 0% |
| Learned with barrier guard in training | 14,227 | +$39.76 | −$34.37 | −$8.64 | −$3.24 | 2.8 | 0% |

Both held-out recordings together (36 hours recorded, 28 full hourly blocks), for the policy trained with the guard:

| | Simple | Avellaneda–Stoikov | Learned with barrier guard |
|---|---|---|---|
| Total profit | −$4.93 | −$27.01 | +$2.46 |
| Profit per hour | −$0.20 ± 0.34 | −$0.99 ± 0.39 | +$0.05 ± 0.12 |
| Mean inventory (orders) | 8.2 | 8.6 | 2.6 |

- **What held up:** the learned policy carries 69% less inventory than the tuned textbook bot, and its hourly profit is
  less variable. It beats the textbook bot by $1.05 per hour ± $0.37 (t = 2.9).
- **What did not:** its profit is about zero, not positive, and its lead over the simple bot is $0.26 per hour ± $0.33
  (t = 0.8), which is not distinguishable from zero. The first test's $0.48 per hour did not repeat.
- **The guard earned its place on this day.** The policy with no guard went to 3.8 times the inventory limit and spent
  68 minutes over it. Every guarded version stayed inside, at a cost of $0.08 to $2.94 depending on the guard.
- Standard errors treat hourly blocks as independent, which is generous.

## Method

**Data.** `tools/record_l3.py` records Bitfinex's raw book (every order by id), every trade, and the exchange's own
checksums. Two 12-hour recordings: 2.08 million order updates, 25,352 trades. The rebuilt book matches all 30,457
exchange checksums with no sequence gaps (checked with `replay_bitfinex` from
[matching-engine](https://github.com/B4ryterx/matching-engine)).

**Fills** (`sim.py`). Quotes are virtual and do not move the market. A quote joins the back of the queue at its price
and fills only when a trade prints through its price, or a trade at its price hits an order that joined later. Trade
prints are matched to the book change that follows them (83% one-to-one, 14% partial or grouped); the 2% with no book
change, hidden orders, never fill us.

**Profit split.** Per fill: spread = distance from the mid at the fill; adverse selection = mid move over the next
10 seconds; inventory = mid move after that. The three sum exactly to total profit.

**Learning** (`learn.py`). A linear policy maps six inputs (inventory, book imbalance, 10-second price move, spread,
volatility, constant) to a quote offset in ticks for each side. It is trained by the cross-entropy method (96 policies
per generation, 25 generations) on the first 8 hours of recording 1, to maximise profit minus a quadratic inventory
penalty. The last 4 hours of recording 1 choose the generation. Recording 2 is the test set.

**Barrier guard** (`make_cbf` in `sim.py`). Safe set h(q) = Q² − q² ≥ 0 for inventory q and limit Q. The
discrete-time condition h(q⁺) ≥ (1 − α) h(q) is enforced against the worst case that one side's whole posted size
fills before the next decision, which gives a closed form:

    bid size ≤ √(αQ² + (1 − α)q²) − q        ask size ≤ √(αQ² + (1 − α)q²) + q

The guard leaves the policy alone while these hold, shrinks the risky side near the limit, and guarantees |q| ≤ Q.
α = 1 is the plain hard cap.

## Limits

- About 24 hours of one thin market (roughly one trade every three to four seconds, median trade under $20).
- No market impact, no latency, zero fees. The learned policy often improves the best price by one tick; real competitors would respond.
- Training profit rose from $3 to $8 over generations while validation stayed near $2–2.5, so the policy overfits and an early generation was kept.
- The test comparison was run once by mistake while training was in progress. Nothing was changed afterwards; generations were chosen on validation data only.
- Arrival order within a price level is inferred from the feed, not confirmed by the exchange.

## Run

```
python3 tools/record_l3.py data/btcusd_l3.csv 12           # recording 1 (needs websocket-client)
python3 tools/record_l3.py data/btcusd_l3_part2.csv 12     # recording 2
python3 baselines.py                                       # tune and test Avellaneda-Stoikov on recording 1
python3 learn.py unguarded none && python3 learn.py capped cap && python3 learn.py cbf cbf
python3 compare.py                                         # the table above
```

Needs Python 3 with numpy, pandas and sortedcontainers. Recordings are not in the repository (about 180 MB).
The trained policies used for the table are in `policy_*.json`.
