"""Record what a market-making simulator needs from Bitfinex's public feed: every individual order
(raw book, 250 per side), every trade, and the exchange's checksums. Reconnects on a dropped line.

One flat file; price and amount are in 1e-8 units, amount < 0 = sell, ts = exchange time in ms:
  R                                   new connection (sequence numbers restart)
  S                                   book snapshot follows
  O,id,price,amount,text,seq,ts       order added or changed (text = amount exactly as sent)
  D,id,seq,ts                         order removed
  T,trade_id,price,amount,seq,ts      trade; amount > 0 means the buyer was the aggressor
  C,checksum,seq,ts                   exchange CRC32 of its top 25 bids and asks
  H,seq                               any other message (keeps the sequence unbroken)
Usage: record_l3.py <out_file> [hours] [symbol]
"""
import json
import sys
import time
from decimal import Decimal

import websocket  # websocket-client

out = sys.argv[1]
hours = float(sys.argv[2]) if len(sys.argv) > 2 else 12
symbol = sys.argv[3] if len(sys.argv) > 3 else "tBTCUSD"
CHECKSUM, SEQUENCE, TIMESTAMP = 131072, 65536, 32768
units = lambda s: int(Decimal(s) * 10**8)
end = time.time() + hours * 3600
counts = {"O": 0, "D": 0, "T": 0, "C": 0, "reconnects": -1}

with open(out, "a", buffering=1 << 16) as f:
    while time.time() < end:
        try:
            ws = websocket.create_connection("wss://api-pub.bitfinex.com/ws/2", timeout=30)
            ws.send(json.dumps({"event": "conf", "flags": CHECKSUM + SEQUENCE + TIMESTAMP}))
            ws.send(json.dumps({"event": "subscribe", "channel": "book", "symbol": symbol, "prec": "R0", "len": "250"}))
            ws.send(json.dumps({"event": "subscribe", "channel": "trades", "symbol": symbol}))
            f.write("R\n"); counts["reconnects"] += 1
            chan, last_flush = {}, time.time()
            while time.time() < end:
                m = json.loads(ws.recv(), parse_float=str, parse_int=str)
                if isinstance(m, dict):
                    if m.get("event") == "subscribed": chan[str(m["chanId"])] = m["channel"]
                    continue
                kind, body, seq, ts = chan.get(m[0]), m[1], m[-2], m[-1]
                if kind == "book" and body == "cs":
                    f.write(f"C,{int(m[2]) & 0xffffffff},{seq},{ts}\n"); counts["C"] += 1
                elif kind == "book" and isinstance(body, list):
                    rows = body if isinstance(body[0], list) else [body]
                    if isinstance(body[0], list): f.write("S\n")
                    for oid, price, amount in rows:
                        if Decimal(price) == 0:
                            f.write(f"D,{oid},{seq},{ts}\n"); counts["D"] += 1
                        else:
                            f.write(f"O,{oid},{units(price)},{units(amount)},{amount},{seq},{ts}\n"); counts["O"] += 1
                elif kind == "trades" and body == "te":
                    tid, mts, amount, price = m[2]
                    f.write(f"T,{tid},{units(price)},{units(amount)},{seq},{ts}\n"); counts["T"] += 1
                else:
                    f.write(f"H,{seq}\n")
                if time.time() - last_flush > 30:
                    f.flush(); last_flush = time.time()
        except (websocket.WebSocketException, OSError, ValueError) as e:
            print(f"{time.strftime('%H:%M:%S')} connection lost ({type(e).__name__}); reconnecting", flush=True)
            time.sleep(2)
print(f"{symbol}: {counts}")
