import os
import time

import pandas as pd
import requests

from bot import (BASE, INTERVAL, INTERVAL_MS, SYMBOLS, START_BALANCE, LEVERAGE,
                 MARGIN_PER_TRADE, MAX_POSITIONS, FEE, tg)
from strategy import get_signal

TARGET = int(os.environ.get("BT_CANDLES", "8000"))  # ~28 روز در کندل ۵ دقیقه
WINDOW = 120

# (اسم، دوره EMA برای فیلتر روند یا None، حد ضرر، حد سود)
VARIANTS = [
    ("۱) پایه (بدون فیلتر)", None, 0.015, 0.03),
    ("۲) فیلتر روند EMA288 (~۱ روز)", 288, 0.015, 0.03),
    ("۳) فیلتر روند EMA600 (~۲ روز)", 600, 0.015, 0.03),
    ("۴) EMA600 + حد ضرر ۱٪ / سود ۲٪", 600, 0.010, 0.02),
    ("۵) EMA600 + حد ضرر ۲٪ / سود ۴٪", 600, 0.020, 0.04),
]


def fetch_history(sym):
    rows, end = {}, None
    for _ in range(80):
        params = {"symbol": sym, "interval": INTERVAL, "limit": 500}
        if end:
            params["endTime"] = end
        r = requests.get(f"{BASE}/future/market/v1/public/q/kline",
                         params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("result") or []
        if not data:
            break
        before = len(rows)
        rows.update({int(x["t"]): x for x in data})
        if len(rows) == before or len(rows) >= TARGET:
            break
        end = min(int(x["t"]) for x in data) - 1
        time.sleep(0.2)
    if not rows:
        return None
    df = pd.DataFrame(list(rows.values())).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["t"] = df["t"].astype("int64")
    df = df.sort_values("t").reset_index(drop=True)
    df = df[df["t"] + INTERVAL_MS <= time.time() * 1000]  # فقط کندل بسته
    return df.tail(TARGET).reset_index(drop=True)


def precompute_signals(data):
    """سیگنال هر کندل فقط یک بار حساب می‌شود و بین حالت‌ها مشترک است."""
    sigs = {}
    for s, df in data.items():
        d = {}
        for i in range(90, len(df)):
            sig = get_signal(df.iloc[max(0, i - WINDOW + 1): i + 1].reset_index(drop=True))
            if sig:
                d[i] = sig
        sigs[s] = d
    return sigs


def simulate(data, sigs, ema_span, sl_pct, tp_pct):
    balance = START_BALANCE
    positions, trades, last_signal = [], [], {}
    pos_idx = {s: {int(t): i for i, t in enumerate(d["t"])} for s, d in data.items()}
    times = sorted({int(t) for d in data.values() for t in d["t"]})
    ema = {s: d["close"].ewm(span=ema_span, adjust=False).mean().values
           for s, d in data.items()} if ema_span else None

    for ts in times:
        remaining = []
        for p in positions:
            i = pos_idx[p["symbol"]].get(ts)
            if i is None or ts <= p["after_t"]:
                remaining.append(p)
                continue
            c = data[p["symbol"]].iloc[i]
            if p["side"] == "LONG":
                hit_sl, hit_tp = c["low"] <= p["sl"], c["high"] >= p["tp"]
            else:
                hit_sl, hit_tp = c["high"] >= p["sl"], c["low"] <= p["tp"]
            if not (hit_sl or hit_tp):
                remaining.append(p)
                continue
            exit_price = p["sl"] if hit_sl else p["tp"]
            d = 1 if p["side"] == "LONG" else -1
            pnl = (exit_price - p["entry"]) / p["entry"] * p["notional"] * d
            pnl = max(pnl, -p["margin"])
            fee = p["notional"] * FEE
            balance += p["margin"] + pnl - fee
            trades.append({"symbol": p["symbol"], "side": p["side"],
                           "net": pnl - fee - p["open_fee"], "t": ts})
        positions = remaining

        held = {p["symbol"] for p in positions}
        for sym, df in data.items():
            if len(positions) >= MAX_POSITIONS:
                break
            if sym in held:
                continue
            i = pos_idx[sym].get(ts)
            if i is None:
                continue
            sig = sigs[sym].get(i)
            if not sig:
                continue
            side, sig_t = sig
            if last_signal.get(sym) == sig_t:
                continue
            price = float(df["close"].iloc[i])
            if ema_span:
                if i < ema_span:
                    continue
                e = ema[sym][i]
                if (side == "LONG" and price <= e) or (side == "SHORT" and price >= e):
                    continue
            margin = min(MARGIN_PER_TRADE, balance)
            notional = margin * LEVERAGE
            open_fee = notional * FEE
            if margin < 0.5 or balance < margin + open_fee:
                continue
            balance -= margin + open_fee
            sl = price * (1 - sl_pct) if side == "LONG" else price * (1 + sl_pct)
            tp = price * (1 + tp_pct) if side == "LONG" else price * (1 - tp_pct)
            positions.append({"symbol": sym, "side": side, "entry": price, "sl": sl,
                              "tp": tp, "margin": margin, "notional": notional,
                              "open_fee": open_fee, "after_t": ts})
            last_signal[sym] = sig_t
    return trades


def stats(tr):
    n = len(tr)
    if n == 0:
        return "0 معامله"
    wins = sum(t["net"] for t in tr if t["net"] > 0)
    loss = -sum(t["net"] for t in tr if t["net"] <= 0)
    pf = wins / loss if loss > 0 else float("inf")
    wr = sum(t["net"] > 0 for t in tr) / n
    total = sum(t["net"] for t in tr)
    return f"{n} معامله | برد {wr:.0%} | {total:+.2f} | PF {pf:.2f}"


def side_sum(tr, side):
    return sum(t["net"] for t in tr if t["side"] == side)


def main():
    data = {}
    for s in SYMBOLS:
        try:
            df = fetch_history(s)
            if df is not None and len(df) > 700:
                data[s] = df
                print(s, len(df), "candles")
            else:
                print("no data:", s)
        except Exception as e:
            print("error", s, e)
    if not data:
        tg("⚠️ بک‌تست: دیتایی از XT نگرفتم.")
        return

    days = min(len(d) for d in data.values()) * 5 / 1440
    sigs = precompute_signals(data)
    all_t = sorted({int(t) for d in data.values() for t in d["t"]})
    mid = all_t[len(all_t) // 2]

    parts = [f"🧪 مقایسه استراتژی‌ها ({days:.0f} روز، کندل ۵ دقیقه)",
             "سود خالص به USDT. PF بالای ۱ یعنی سودده."]
    for name, ema_span, sl, tp in VARIANTS:
        tr = simulate(data, sigs, ema_span, sl, tp)
        h1 = [t for t in tr if t["t"] < mid]
        h2 = [t for t in tr if t["t"] >= mid]
        parts.append(f"\n{name}\n"
                     f"کل: {stats(tr)}\n"
                     f"نیمه اول: {stats(h1)}\n"
                     f"نیمه دوم: {stats(h2)}\n"
                     f"LONG {side_sum(tr, 'LONG'):+.2f} | SHORT {side_sum(tr, 'SHORT'):+.2f}")
    msg = "\n".join(parts)
    print(msg)
    tg(msg)


if __name__ == "__main__":
    main()
