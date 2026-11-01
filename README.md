# Safe learning market maker

Does a market-making policy found by trial and error beat the textbook formula, and what does a
barrier-function safety guard cost? Tested on real order-by-order data with queue-aware fills.

## Summary

Across four held-out days (57 hours of real Bitfinex order-by-order data):

- **The safety result holds.** Every bot with the barrier guard stayed inside its inventory limit in every test,
  including with 1 s of order latency. The plain hard cap did not: under latency it went over the limit for up to
  641 s on day 3 and 1,595 s on day 4. A volatility-aware version of the barrier beat the fixed barrier on every policy.
- **The profit result does not.** The learned policy beat the tuned Avellaneda–Stoikov bot consistently, but never
  the simple join-the-best-price bot by a clear margin, and on the fourth day every bot lost money.

- **The guarantee needed one fix.** As first written the barrier's proof assumed no latency. With latency, a quote
  whose cancel has not landed can still fill; at α = 0.2 that happens to stay inside the limit, but from α = 0.4 it
  does not. Counting quotes still in flight restores the guarantee for any α, at no cost (see the last section).

The write-up ([WRITEUP.pdf](WRITEUP.pdf)) gives the whole study in three pages. The detail follows, in the order the
tests were run.

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

## Stress tests (29 October): seeds, latency, queue model, fees

All on the two held-out recordings together (28 full hours). `stress.py <scenario>` and `volguard.py [latency_ms]`
reproduce every number; results are in `stress_*.csv` and `volguard*.csv`.

**Five more training runs per variant** (`learn.py <name> <guard> <seed>`, policies in `seeds/`). Total profit over 28 h:

| Variant | Published run | Seeds 1–5 | Seconds over the limit |
|---|---|---|---|
| Trained with the barrier guard | +$2.46 | −5.46, +4.13, −2.63, +3.32, +3.19 | 0 in every run |
| Trained with no guard | +$2.79 | +0.97, +3.27, +6.75, +6.32, +3.48 | 2,898 to 10,619 (up to 4.6× the limit) |
| No-guard policy, guard added at test | +$0.20 | −0.32, +1.22, +5.10, +5.01, +1.13 | 0 in every run |

- Every one of the 18 learned policies beat the tuned textbook bot (t from 1.7 to 2.9) and carried far less inventory.
- None beat the simple join-the-best-price bot by a statistically clear margin (largest t = 0.87).
- Two of the five guard-trained seeds (1 and 3) came out worse: twice the inventory and negative profit. Both picked
  generation 0 on the 4-hour validation window, so **choosing the generation is the weak step**, not the guard.

**Latency and a pessimistic queue** (published guard-trained policy; simple bot for reference):

| Scenario | Learned + guard | Simple | Learned − textbook (t) | Simple bot's worst position |
|---|---|---|---|---|
| Published setting | +$2.46 | −$4.93 | +$1.05/h (2.9) | 0.0100 BTC |
| 250 ms to place or cancel | +$2.03 | −$3.66 | +$1.07/h (3.1) | 0.0108 BTC (over the limit for 302 s) |
| 1 s to place or cancel | −$1.34 | −$6.87 | +$1.06/h (3.4) | 0.0109 BTC (over the limit for 641 s) |
| Fill only when a trade prints through our price | −$1.21 | −$11.58 | +$0.93/h (2.4) | 0.0100 BTC |

- **The barrier guard holds the limit under latency and the plain hard cap does not.** The hard cap only stops quoting
  at the limit, but a cancel that has not yet landed can still fill. The barrier guard sizes quotes for the worst case
  in advance, so even with 1 s of latency no guarded run went past 0.0097 BTC.
- One second of latency costs the learned bots about $4 over 28 hours and takes their profit below zero.
- **Fees decide everything.** At a 1 bps maker fee every bot loses money in every scenario, $36 to $88 over the 28 hours. The edge is a fraction of a basis point.

## A volatility-aware barrier

`make_cbf_vol` in `sim.py`: the same safe set, but the limit shrinks when the market is jumpier than usual,
Q_eff = Q · clip(σ_ref / σ, 0.5, 1), with σ the 5-minute standard deviation of one-second mid changes. If Q_eff
drops below the current position, only the reducing side is quoted, so the bot unwinds through passive fills.

Bolted on at test time to the six policies trained with the fixed barrier (σ_ref = $1.18, the median on the
training hours), against the fixed barrier on the same policy, hour by hour:

| | Profit change over 28 h | Hourly profit volatility | Worst hour |
|---|---|---|---|
| No latency | better for 6 of 6 policies (+$1.03 to +$4.92; t 0.6 to 1.7) | 25% to 42% lower | better for 6 of 6 |
| 1 s latency | better for 6 of 6 policies (+$1.99 to +$7.67; t 1.5 to 3.0) | 15% to 35% lower | better for 6 of 6 |

- The gain is mainly risk: steadier hours and a smaller worst hour. Most of the extra profit comes from the two weak
  seeds, which the tighter limit rescues.
- A looser reference (σ_ref = $2.36, the 90th percentile) changed little, and training with the volatility-aware
  barrier did not beat training with the fixed one.
- Caveat: two reference levels were tried on the test data, so the median result carries a mild selection effect.
  The six policies share the same market hours, so the six comparisons are not independent.

## Choosing the generation

The two weak seeds above both kept generation 0, after one update. I retrained the five guard-trained seeds saving
every generation (`seeds_hist/`), then compared four selection rules with the fixed barrier on the 28 held-out hours
(`select_gen.py`):

| Rule | Total profit, seeds 1–5 | Mean | Range |
|---|---|---|---|
| A. Best on the 4-hour validation window (published rule) | −5.46, +4.13, −2.63, +3.32, +3.19 | +$0.51 | $9.59 |
| B. Last generation | +4.23, +1.00, +2.19, +3.05, +4.27 | +$2.95 | $3.27 |
| C. Average of the last ten generations' weights | +3.26, +2.26, +2.36, +2.86, +3.17 | +$2.78 | $1.00 |
| D. Best on validation, generation 5 or later | +3.45, +4.13, +2.88, +3.32, +3.19 | +$3.39 | $1.25 |

- Any rule that avoids the barely trained early policies makes all five seeds profitable on held-out data, with
  mean inventory about 2.5 orders.
- Rule C uses no validation data at all and gives the tightest spread, so it is the new default.
- Caveats: the rules were written after seeing that seeds 1 and 3 picked generation 0. Even the best seed only beats the
  simple bot at t = 0.99.

## Fourth day: a further 21 hours (recorded 29–30 October)

A fourth recording, made after every choice above was fixed, and not used for anything else. No bot was retuned.
`stress.py`, `volguard.py` and `select_gen.py` take the recording as an argument; outputs carry the suffix
`_btcusd_l3_part4`.

| Bot | Total, 21 h | Mean inventory (orders) | Seconds over the limit | Same, with 1 s latency |
|---|---|---|---|---|
| Simple: join best bid and ask | −$35.84 | 6.6 | 0 | 1,595 |
| Avellaneda–Stoikov, tuned | −$135.05 | 6.0 | 0 | 1,496 |
| Learned, no guard | −$106.78 | 10.2 | 21,654 (7.4× the limit) | 17,768 |
| Learned, guard in training | −$39.64 | 3.5 | 0 | 0 |
| Seeds 1–5, guard in training | −$53.12 to −$38.90 | 3.3 to 5.6 | 0 | 0 |

- **Every bot lost money.** The learned policies were level with the simple bot (t from −0.7 to −0.1) and well ahead of
  the textbook bot (t 3.5 to 5.2). The earlier profit did not persist.
- **The guard held again**, with and without latency, for all eleven guarded policies. Without it, the five no-guard
  seeds went 5 to 11 times over the limit.
- **The volatility-aware barrier won again** for all six fixed-barrier policies: published policy −$31.33 vs −$39.64,
  hourly profit volatility $2.66 vs $3.62, worst hour −$9.08 vs −$12.40.
- **Rule C (average of the last ten generations) kept the seeds together:** −$37.82 to −$39.83, against −$38.90 to
  −$53.75 for the published rule A.

## Latency and the guarantee (1 November)

Unit tests (`tests/`, `make test`) found a gap in the guard's guarantee. Without latency, |q| ≤ Q at one decision
implies it at the next. With latency, the outgoing quote (cancel not yet landed) and the incoming one were both sized
against the same inventory, and both can fill. On an adversarial tape where every posted bid fills, with 1 s latency:

| α | 0.05 | 0.2 | 0.4 | 0.5 | 0.7 | 1.0 (hard cap) |
|---|---|---|---|---|---|---|
| Largest position, plain barrier (Q = 0.01) | 0.0084 | 0.0097 | 0.0103 | 0.0105 | 0.0107 | 0.0110 |
| Largest position, in-flight barrier | 0.0082 | 0.0096 | 0.0098 | 0.0099 | 0.0099 | 0.0100 |

At the published α = 0.2 each quote takes only a fraction of the room left, which is why no latency run breached; it
is not a guarantee. `make_cbf(alpha, inflight=True)` sizes each side from q plus the size still in flight, and the
guarantee then holds for any α while latency is at most one decision interval (proof in the write-up). Rerun on the
real recordings with 1 s latency (`stress.py lat1000i`), it was as good as or better than the plain barrier for 22 of
24 policy-recording pairs (mean +$0.45 on recordings 2–3, +$1.06 on recording 4) with about 3% less inventory.

Pooled over all held-out data (recordings 2–4, 49 complete hours), the guard-trained policy lost $0.78 ± 0.37 an hour:
level with the simple bot (+$0.09/h, t = 0.2) and ahead of the textbook bot (+$2.56/h, t = 5.2), with 3.0 orders of
mean inventory against 7.6 for both baselines.

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

- About 57 hours of one thin market (roughly one trade every three to four seconds, median trade under $20).
- No market impact and zero fees in the main tables (latency and fees are in the stress tests). The learned policy often improves the best price by one tick; real competitors would respond.
- Training profit rose from $3 to $8 over generations while validation stayed near $2–2.5, so the policy overfits and an early generation was kept.
- The test comparison was run once by mistake while training was in progress. Nothing was changed afterwards; generations were chosen on validation data only.
- Arrival order within a price level is inferred from the feed, not confirmed by the exchange.

## Run

```
pip install -r requirements.txt
make test       # 27 unit tests on synthetic books, under a second, no data needed
make check      # the published recording-2 numbers (2,444 fills / +$3.487 simple, +$5.704 guarded policy)
make results    # every table here and in the write-up, from the recordings and the saved policies
make retrain    # retrain the published policies and seeds (slow)
python3 tools/writeup_pdf.py    # rebuild WRITEUP.pdf from WRITEUP.md (needs Chrome and the markdown package)
```

Recordings are made with `python3 tools/record_l3.py data/btcusd_l3_partN.csv <hours>` (needs websocket-client) and
are not in the repository (about 1 GB). The trained policies used for the tables are in `policy_*.json`, `seeds/` and
`seeds_hist/`.
