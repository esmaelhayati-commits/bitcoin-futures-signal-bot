import os
import time

import pandas as pd
import requests

from bot import (BASE, INTERVAL, INTERVAL_MS, SYMBOLS, START_BALANCE, LEVERAGE,
                 MARGIN_PER_TRADE, MAX_POSITIONS, SL_PCT, TP_PCT, FEE, tg)
from strategy import get_signal

TARGET = int(os.environ.get("BT_CANDLES", "8000"))  # ~28 روز در کندل ۵ دقیقه
WINDOW = 120


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


def simulate(data):
    balance = START_BALANCE
    positions, trades, last_signal = [], [], {}
    pos_idx = {s: {int(t): i for i, t in enumerate(d["t"])} for s, d in data.items()}
    times = sorted({int(t) for d in data.values() for t in d["t"]})

    for ts in times:
        # ۱) خروج
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

        # ۲) ورود
        held = {p["symbol"] for p in positions}
        for sym, df in data.items():
            if len(positions) >= MAX_POSITIONS:
                break
            if sym in held:
                continue
            i = pos_idx[sym].get(ts)
            if i is None or i < 90:
                continue
            sig = get_signal(df.iloc[max(0, i - WINDOW + 1): i + 1].reset_index(drop=True))
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
            sl = price * (1 - SL_PCT) if side == "LONG" else price * (1 + SL_PCT)
            tp = price * (1 + TP_PCT) if side == "LONG" else price * (1 - TP_PCT)
            positions.append({"symbol": sym, "side": side, "entry": price, "sl": sl,
                              "tp": tp, "margin": margin, "notional": notional,
                              "open_fee": open_fee, "after_t": ts})
            last_signal[sym] = sig_t
    return trades, balance, positions


def report(trades, balance, open_pos, days):
    n = len(trades)
    if n == 0:
        return f"🧪 بک‌تست ({days:.0f} روز): هیچ معامله بسته‌شده‌ای نبود."
    wins = [t["net"] for t in trades if t["net"] > 0]
    losses = [t["net"] for t in trades if t["net"] <= 0]
    total = sum(t["net"] for t in trades)
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    cum, peak, mdd = 0.0, 0.0, 0.0
    for t in trades:
        cum += t["net"]
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)

    def side_line(side):
        ts = [t["net"] for t in trades if t["side"] == side]
        if not ts:
            return f"{side}: -"
        return f"{side}: {len(ts)} معامله، برد {sum(x > 0 for x in ts) / len(ts):.0%}، جمع {sum(ts):+.2f}"

    by_sym = {}
    for t in trades:
        by_sym.setdefault(t["symbol"], []).append(t["net"])
    sym_lines = [f"• {s.split('_')[0].upper()}: {len(v)} معامله، جمع {sum(v):+.2f}"
                 for s, v in sorted(by_sym.items(), key=lambda x: -sum(x[1]))]

    return (f"🧪 بک‌تست ({days:.0f} روز، کندل ۵ دقیقه)\n"
            f"معاملات: {n} | نرخ برد: {len(wins) / n:.0%}\n"
            f"سود خالص: {total:+.2f} USDT (از {START_BALANCE})\n"
            f"میانگین سود: {sum(wins) / len(wins) if wins else 0:.3f} | "
            f"میانگین زیان: {sum(losses) / len(losses) if losses else 0:.3f}\n"
            f"ضریب سود (PF): {pf:.2f} | حداکثر افت: {mdd:.2f} USDT\n"
            f"{side_line('LONG')}\n{side_line('SHORT')}\n"
            + "\n".join(sym_lines)
            + f"\nپوزیشن باز در پایان: {len(open_pos)}")


def main():
    data = {}
    for s in SYMBOLS:
        try:
            df = fetch_history(s)
            if df is not None and len(df) > 100:
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
    trades, balance, open_pos = simulate(data)
    msg = report(trades, balance, open_pos, days)
    print(msg)
    tg(msg)


if __name__ == "__main__":
    main()
