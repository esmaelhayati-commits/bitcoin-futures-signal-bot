import json
import os
import time

import pandas as pd
import requests

from strategy import get_signal

BASE = "https://fapi.xt.com"
INTERVAL = "5m"
INTERVAL_MS = 5 * 60 * 1000

SYMBOLS = ["btc_usdt", "eth_usdt", "sol_usdt", "xrp_usdt", "doge_usdt",
           "bnb_usdt", "ada_usdt", "link_usdt", "avax_usdt", "ltc_usdt"]

START_BALANCE = 10.0
LEVERAGE = 20
MARGIN_PER_TRADE = 2.0
MAX_POSITIONS = 3
SL_PCT = 0.015
TP_PCT = 0.03
FEE = 0.0006  # کارمزد فرضی هر طرف

STATE_FILE = "state.json"
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def tg(text):
    if not TOKEN or not CHAT_ID:
        print("TELEGRAM:", text)
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=15)
    except Exception as e:
        print("telegram error:", e)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"balance": START_BALANCE, "positions": [], "history": [],
            "last_signal": {}, "last_summary": 0}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=1)


def get_candles(symbol):
    r = requests.get(f"{BASE}/future/market/v1/public/q/kline",
                     params={"symbol": symbol, "interval": INTERVAL, "limit": 200},
                     timeout=15)
    r.raise_for_status()
    data = r.json().get("result") or []
    df = pd.DataFrame(data)
    if df.empty:
        return None
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["t"] = df["t"].astype("int64")
    return df.sort_values("t").reset_index(drop=True)


def close_position(state, pos, exit_price, reason):
    direction = 1 if pos["side"] == "LONG" else -1
    pnl = (exit_price - pos["entry"]) / pos["entry"] * pos["notional"] * direction
    pnl = max(pnl, -pos["margin"])  # حداکثر ضرر = مارجین (لیکوییدی)
    fee = pos["notional"] * FEE
    state["balance"] += pos["margin"] + pnl - fee
    net = pnl - fee - pos["open_fee"]
    state["history"].append({"symbol": pos["symbol"], "side": pos["side"],
                             "net": round(net, 4), "reason": reason})
    state["history"] = state["history"][-200:]
    icon = "✅" if net > 0 else "❌"
    tg(f"{icon} بسته شد: {pos['symbol'].upper()} {pos['side']}\n"
       f"دلیل: {reason}\nورود: {pos['entry']} | خروج: {exit_price}\n"
       f"سود/زیان خالص: {net:.3f} USDT\nموجودی: {state['balance']:.3f} USDT")


def check_exits(state, candles):
    remaining = []
    for pos in state["positions"]:
        df = candles.get(pos["symbol"])
        if df is None:
            remaining.append(pos)
            continue
        new = df[df["t"] > pos["after_t"]]
        closed = False
        for _, c in new.iterrows():
            if pos["side"] == "LONG":
                hit_sl, hit_tp = c["low"] <= pos["sl"], c["high"] >= pos["tp"]
            else:
                hit_sl, hit_tp = c["high"] >= pos["sl"], c["low"] <= pos["tp"]
            if hit_sl:  # محافظه‌کارانه: اگر هر دو در یک کندل بودند، حد ضرر
                close_position(state, pos, pos["sl"], "حد ضرر")
                closed = True
                break
            if hit_tp:
                close_position(state, pos, pos["tp"], "حد سود")
                closed = True
                break
        if not closed:
            remaining.append(pos)
    state["positions"] = remaining


def open_position(state, symbol, side, price, sig_t, last_t):
    margin = min(MARGIN_PER_TRADE, state["balance"])
    if margin < 0.5:
        return
    notional = margin * LEVERAGE
    open_fee = notional * FEE
    if state["balance"] < margin + open_fee:
        return
    state["balance"] -= margin + open_fee
    if side == "LONG":
        sl, tp = price * (1 - SL_PCT), price * (1 + TP_PCT)
    else:
        sl, tp = price * (1 + SL_PCT), price * (1 - TP_PCT)
    state["positions"].append({
        "symbol": symbol, "side": side, "entry": price,
        "sl": round(sl, 8), "tp": round(tp, 8),
        "margin": margin, "notional": notional, "open_fee": open_fee,
        "after_t": last_t})
    state["last_signal"][symbol] = sig_t
    tg(f"🔔 پوزیشن جدید (شبیه‌سازی)\n{symbol.upper()} {side} x{LEVERAGE}\n"
       f"ورود: {price}\nحد ضرر: {sl:.6g} | حد سود: {tp:.6g}\n"
       f"مارجین: {margin} USDT | حجم: {notional} USDT")


def summary(state, prices):
    unreal = 0.0
    lines = []
    for p in state["positions"]:
        cur = prices.get(p["symbol"], p["entry"])
        d = 1 if p["side"] == "LONG" else -1
        u = (cur - p["entry"]) / p["entry"] * p["notional"] * d
        unreal += u + p["margin"]
        lines.append(f"• {p['symbol'].upper()} {p['side']} ({u:+.3f})")
    equity = state["balance"] + unreal
    h = state["history"]
    wins = sum(1 for x in h if x["net"] > 0)
    tg(f"📊 گزارش\nارزش کل: {equity:.3f} USDT (شروع {START_BALANCE})\n"
       f"معاملات بسته: {len(h)} | برد: {wins}\n"
       f"پوزیشن باز: {len(state['positions'])}\n" + "\n".join(lines))


def main():
    state = load_state()
    now_ms = int(time.time() * 1000)
    candles, prices = {}, {}

    for sym in SYMBOLS:
        try:
            df = get_candles(sym)
            if df is None:
                print("no data:", sym)
                continue
            candles[sym] = df
            prices[sym] = float(df["close"].iloc[-1])
        except Exception as e:
            print("error", sym, e)

    print("symbols with data:", list(candles))
    if not candles:
        tg("⚠️ از XT هیچ دیتایی نگرفتم. آدرس API یا نام جفت‌ارزها را چک کنید.")
        return

    check_exits(state, candles)

    held = {p["symbol"] for p in state["positions"]}
    for sym, df in candles.items():
        if len(state["positions"]) >= MAX_POSITIONS:
            break
        if sym in held:
            continue
        closed = df[df["t"] + INTERVAL_MS <= now_ms].reset_index(drop=True)
        sig = get_signal(closed)
        if not sig:
            continue
        side, sig_t = sig
        if state["last_signal"].get(sym) == sig_t:
            continue
        last_t = int(closed["t"].iloc[-1])
        open_position(state, sym, side, prices[sym], sig_t, last_t)

    # گزارش دوره‌ای هر ۶ ساعت
    if now_ms - state.get("last_summary", 0) > 6 * 3600 * 1000:
        summary(state, prices)
        state["last_summary"] = now_ms

    save_state(state)


if __name__ == "__main__":
    main()
