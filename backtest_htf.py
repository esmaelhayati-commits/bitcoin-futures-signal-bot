import os
import time

import pandas as pd
import requests

from bot import (BASE, SYMBOLS, LEVERAGE,
                 MARGIN_PER_TRADE, MAX_POSITIONS, FEE, tg)
from strategy import get_signal

TARGET = int(os.environ.get("BT_CANDLES", "8000"))
WINDOW = 120
TF_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}

# (اسم، تایم‌فریم، حد ضرر، حد سود)
VARIANTS = [
    ("۱) ۱۵ دقیقه | ضرر ۱.۵٪ / سود ۳٪", "15m", 0.015, 0.03),
    ("۲) ۱۵ دقیقه | ضرر ۲٪ / سود ۴٪", "15m", 0.020, 0.04),
    ("۳) ۱ ساعته | ضرر ۲٪ / سود ۴٪", "1h", 0.020, 0.04),
    ("۴) ۱ ساعته | ضرر ۳٪ / سود ۶٪", "1h", 0.030, 0.06),
]


def fetch_history(sym, interval):
    ms = TF_MS[interval]
    rows, end = {}, None
    for _ in range(80):
        params = {"symbol": sym, "interval": interval, "limit": 500}
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
    df = df[df["t"] + ms <= time.time() * 1000]
    return df.tail(TARGET).reset_index(drop=True)


def precompute_signals(data):
    sigs = {}
    for s, df in data.items():
        d = {}
        for i in range(90, len(df)):
            sig = get_signal(df.iloc[max(0, i - WINDOW + 1): i + 1].reset_index(drop=True))
            if sig:
                d[i] = sig
        sigs[s] = d
    return sigs


def simulate(data, sigs, sl_pct, tp_pct):
    balance = 1_000_000.0  # حساب تمام‌نشدنی تا نتیجه به ورشکستگی وابسته نشود
    positions, trades, last_signal = [], [], {}
    pos_idx = {s: {int(t): i for i, t in enumerate(d["t"])} for s, d in data.items()}
    times = sorted({int(t) for d in data.values() for t in d["t"]})

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
            trades.append({"symbol": p["symbol"], "side": p["side"], "t": ts,
                           "net": pnl - fee - p["open_fee"],
                           "notional": p["notional"]})
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
            margin = min(MARGIN_PER_TRADE, balance)
            notional = margin * LEVERAGE
            open_fee = notional * FEE
            if margin < 0.5 or balance < margin + open_fee:
                continue
            price = float(df["close"].iloc[i])
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
    avg_pct = sum(t["net"] / t["notional"] for t in tr) / n * 100
    return f"{n} معامله | برد {wr:.0%} | {total:+.2f} | PF {pf:.2f} | میانگین {avg_pct:+.2f}٪ حجم"


def side_sum(tr, side):
    return sum(t["net"] for t in tr if t["side"] == side)


def main():
    datasets, signals, spans = {}, {}, {}
    for tf in sorted({v[1] for v in VARIANTS}):
        data = {}
        for s in SYMBOLS:
            try:
                df = fetch_history(s, tf)
                if df is not None and len(df) > 200:
                    data[s] = df
                    print(tf, s, len(df), "candles")
                else:
                    print("no data:", tf, s)
            except Exception as e:
                print("error", tf, s, e)
        if not data:
            continue
        datasets[tf] = data
        spans[tf] = min(len(d) for d in data.values()) * TF_MS[tf] / 86400000
        signals[tf] = precompute_signals(data)

    if not datasets:
        tg("⚠️ بک‌تست: دیتایی از XT نگرفتم.")
        return

    parts = ["🧪 مقایسه تایم‌فریم‌ها",
             "سود خالص به USDT. PF بالای ۱ یعنی سودده. داده به ۳ بخش زمانی تقسیم شده."]
    for name, tf, sl, tp in VARIANTS:
        if tf not in datasets:
            continue
        tr = simulate(datasets[tf], signals[tf], sl, tp)
        all_t = sorted({int(t) for d in datasets[tf].values() for t in d["t"]})
        b1, b2 = all_t[len(all_t) // 3], all_t[2 * len(all_t) // 3]
        p1 = [t for t in tr if t["t"] < b1]
        p2 = [t for t in tr if b1 <= t["t"] < b2]
        p3 = [t for t in tr if t["t"] >= b2]
        parts.append(f"\n{name} ({spans[tf]:.0f} روز داده)\n"
                     f"کل: {stats(tr)}\n"
                     f"بخش ۱: {stats(p1)}\n"
                     f"بخش ۲: {stats(p2)}\n"
                     f"بخش ۳: {stats(p3)}\n"
                     f"LONG {side_sum(tr, 'LONG'):+.2f} | SHORT {side_sum(tr, 'SHORT'):+.2f}")
    msg = "\n".join(parts)
    print(msg)
    tg(msg)


if __name__ == "__main__":
    main()
